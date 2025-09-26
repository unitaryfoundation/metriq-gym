import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
import urllib.error
from typing import Optional, Any

from metriq_gym.exporters.base_exporter import BaseExporter


class GitHubPRExporter(BaseExporter):
    """
    Export job results as a JSON file to a GitHub repository by opening a Pull Request.

    Auth is provided via a GitHub token in the environment (e.g., GITHUB_TOKEN).
    The exporter clones the target repo, creates a branch, writes the JSON file,
    commits, pushes, and then opens a PR via the GitHub REST API.
    """

    def export(
        self,
        *,
        repo: str,
        base_branch: str = "main",
        directory: str = "",
        branch_name: Optional[str] = None,
        token: Optional[str] = None,
        commit_message: Optional[str] = None,
        pr_title: Optional[str] = None,
        pr_body: Optional[str] = None,
        clone_dir: Optional[str] = None,
        payload: dict[str, Any] | list[dict[str, Any]] | None = None,
        filename: Optional[str] = None,
        append: bool = False,
    ) -> str:
        """
        Create a PR adding a JSON result file.

        Args:
            repo: "owner/repo" string.
            base_branch: Base branch to target for the PR.
            directory: Directory inside the repo to place the file.
            branch_name: Branch to create for the changes. Defaults to "mgym/upload-<job_id>".
            token: GitHub token. If None, read from GITHUB_TOKEN.
            committer_name: Optional git commit author name.
            committer_email: Optional git commit author email.
            commit_message: Commit message. Defaults to a message with job id.
            pr_title: Pull request title. Defaults to a title with job id.
            pr_body: Pull request body. Optional.
            clone_dir: Optional directory to perform clone/work. Defaults to a temp dir.

        Returns:
            The URL of the created pull request.
        """

        if token is None:
            token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if not token:
            raise RuntimeError("GitHub token not provided. Set GITHUB_TOKEN.")

        if branch_name is None:
            branch_name = f"mgym/upload-{self.metriq_gym_job.id}"
        # Default PR title based on job context
        if pr_title is None:
            job_type = (
                self.metriq_gym_job.job_type.value
                if hasattr(self.metriq_gym_job.job_type, "value")
                else str(self.metriq_gym_job.job_type)
            )
            pr_title = (
                f"mgym upload: {job_type} on "
                f"{self.metriq_gym_job.provider_name}/{self.metriq_gym_job.device_name}"
            )
        # Align commit message with PR title so browser compare uses a meaningful default
        if commit_message is None:
            commit_message = pr_title

        owner, repo_name = repo.split("/", 1)

        # Prepare working directory
        temp_root = None
        if clone_dir is None:
            temp_root = tempfile.mkdtemp(prefix="mgym-upload-")
            workdir = temp_root
        else:
            os.makedirs(clone_dir, exist_ok=True)
            workdir = clone_dir

        # Determine authenticated user and ensure a fork exists
        login = self._get_authenticated_login(token)
        self._ensure_fork(token=token, owner=owner, repo=repo_name, login=login)

        upstream_repo_url = f"https://github.com/{repo}.git"
        fork_repo = f"{login}/{repo_name}"
        fork_push_url = f"https://x-access-token:{token}@github.com/{fork_repo}.git"

        repo_path = os.path.join(workdir, repo_name)

        try:
            # 1) Clone upstream base branch
            self._run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    base_branch,
                    upstream_repo_url,
                    repo_path,
                ]
            )

            # Add remote for fork (for checking/pushing)
            self._run(["git", "-C", repo_path, "remote", "add", "fork", fork_push_url])

            # 2) Prepare branch: if it exists on fork, generate a unique new name; otherwise use given name
            if self._remote_branch_exists(repo_path, "fork", branch_name):
                branch_name = self._next_available_branch_name(repo_path, "fork", branch_name)
            self._run(["git", "-C", repo_path, "checkout", "-b", branch_name])

            # 3) Write JSON/JSONL file
            out_dir = os.path.join(repo_path, directory) if directory else repo_path
            os.makedirs(out_dir, exist_ok=True)

            out_filename = filename or f"{self.metriq_gym_job.id}.json"
            out_path = os.path.join(out_dir, out_filename)
            data = payload if payload is not None else self.as_dict()
            # Pretty JSON array, appending by reading existing file if present
            existing: list = []
            if append and os.path.exists(out_path):
                try:
                    with open(out_path, "r", encoding="utf-8") as rf:
                        parsed = json.load(rf)
                        if isinstance(parsed, list):
                            existing = parsed
                except Exception:
                    # If file is corrupt or not a list, start fresh
                    existing = []
            # Normalize to list of items (prepend new items so newest appear first)
            if isinstance(data, list):
                existing = data + existing
            else:
                existing = [data] + existing
            with open(out_path, "w", encoding="utf-8") as wf:
                json.dump(existing, wf, indent=2)

            # 4) Use git's configured author identity

            # 5) Commit
            self._run(["git", "-C", repo_path, "add", os.path.relpath(out_path, repo_path)])
            self._run(["git", "-C", repo_path, "commit", "-m", commit_message])

            # 6) Push to fork remote
            self._run(["git", "-C", repo_path, "push", "fork", f"HEAD:{branch_name}"])

            # 7) Open PR via GitHub REST API
            # 7) Open PR on upstream using head as "<login>:<branch>"
            compare_url = (
                f"https://github.com/{owner}/{repo_name}/compare/"
                f"{base_branch}...{login}:{branch_name}?expand=1"
            )
            try:
                pr_url = self._create_pull_request(
                    token=token,
                    owner=owner,
                    repo=repo_name,
                    title=pr_title,
                    head=f"{login}:{branch_name}",
                    base=base_branch,
                    body=pr_body,
                )
                return pr_url
            except Exception:
                # Fallback: return compare URL so user can open PR manually in browser
                return compare_url
        finally:
            if temp_root and os.path.isdir(temp_root):
                shutil.rmtree(temp_root, ignore_errors=True)

    def _run(self, cmd: list[str]) -> None:
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            stderr = (
                e.stderr.decode("utf-8", errors="ignore")
                if isinstance(e.stderr, (bytes, bytearray))
                else str(e.stderr)
            )
            stdout = (
                e.stdout.decode("utf-8", errors="ignore")
                if isinstance(e.stdout, (bytes, bytearray))
                else str(e.stdout)
            )
            msg = f"Command failed: {' '.join(cmd)}\nstdout:\n{stdout}\nstderr:\n{stderr}"
            raise RuntimeError(msg) from e

    def _create_pull_request(
        self,
        *,
        token: str,
        owner: str,
        repo: str,
        title: str,
        head: str,
        base: str,
        body: Optional[str] = None,
    ) -> str:
        api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
        data = {
            "title": title,
            "head": head,
            "base": base,
        }
        if body:
            data["body"] = body

        req = urllib.request.Request(
            api_url,
            data=json.dumps(data).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "User-Agent": "metriq-gym",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"GitHub PR creation failed: {e.read().decode('utf-8')}") from e

        # Expect an 'html_url' field in the PR response
        pr_url = resp_data.get("html_url")
        if not pr_url:
            raise RuntimeError("GitHub PR creation: missing html_url in response")
        return pr_url

    def _run_out(self, cmd: list[str]) -> tuple[str, str]:
        try:
            res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            stderr = (
                e.stderr.decode("utf-8", errors="ignore")
                if isinstance(e.stderr, (bytes, bytearray))
                else str(e.stderr)
            )
            stdout = (
                e.stdout.decode("utf-8", errors="ignore")
                if isinstance(e.stdout, (bytes, bytearray))
                else str(e.stdout)
            )
            msg = f"Command failed: {' '.join(cmd)}\nstdout:\n{stdout}\nstderr:\n{stderr}"
            raise RuntimeError(msg) from e
        return (
            res.stdout.decode("utf-8", errors="ignore"),
            res.stderr.decode("utf-8", errors="ignore"),
        )

    def _remote_branch_exists(self, repo_path: str, remote: str, branch: str) -> bool:
        out, _ = self._run_out(["git", "-C", repo_path, "ls-remote", "--heads", remote, branch])
        return bool(out.strip())

    def _next_available_branch_name(self, repo_path: str, remote: str, base: str) -> str:
        # Try numeric suffixes to avoid collisions: base-2, base-3, ...
        for i in range(2, 1000):
            candidate = f"{base}-{i}"
            if not self._remote_branch_exists(repo_path, remote, candidate):
                return candidate
        # As a last resort, append a timestamp-like suffix
        import time

        candidate = f"{base}-{int(time.time())}"
        return candidate

    def _get_authenticated_login(self, token: str) -> str:
        req = urllib.request.Request(
            "https://api.github.com/user",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "metriq-gym",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"Failed to resolve authenticated user: {e.read().decode('utf-8')}"
            ) from e
        login = data.get("login")
        if not login:
            raise RuntimeError("Authenticated user login not found in response")
        return login

    def _ensure_fork(self, *, token: str, owner: str, repo: str, login: str) -> None:
        # Check if fork exists
        if self._repo_exists(token, owner=login, repo=repo):
            return
        # Create fork
        api_url = f"https://api.github.com/repos/{owner}/{repo}/forks"
        req = urllib.request.Request(
            api_url,
            data=json.dumps({"default_branch_only": True}).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "User-Agent": "metriq-gym",
            },
            method="POST",
        )
        try:
            urllib.request.urlopen(req).read()
        except urllib.error.HTTPError as e:
            # If fork already exists or immediate accept, ignore certain errors
            if e.code not in (202, 201):
                # Could be 403/404 in private repos without permission
                # We'll proceed to poll; if the fork doesn't appear, we'll raise a clear message.
                pass

        # Forking is asynchronous; poll a few times for availability
        for _ in range(10):
            if self._repo_exists(token, owner=login, repo=repo):
                return
            time.sleep(1)
        raise RuntimeError(
            "Fork creation failed or is not accessible with this token. "
            "Please fork the repository manually and re-run the command."
        )

    def _repo_exists(self, token: str, *, owner: str, repo: str) -> bool:
        api_url = f"https://api.github.com/repos/{owner}/{repo}"
        req = urllib.request.Request(
            api_url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "metriq-gym",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.getcode() == 200
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            raise
