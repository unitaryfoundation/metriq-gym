import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
import urllib.error
import urllib.parse
from typing import Optional, Any


from metriq_gym.exporters.base_exporter import BaseExporter

MAX_BRANCH_SUFFIX_ATTEMPTS = 1000


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
        pr_labels: Optional[list[str]] = None,
        clone_dir: Optional[str] = None,
        payload: dict[str, Any] | list[dict[str, Any]] | None = None,
        filename: Optional[str] = None,
        append: bool = False,
        dry_run: bool = False,
    ) -> str:
        """
        Create a PR adding a JSON result file.

        Args:
            repo: "owner/repo" string.
            base_branch: Base branch to target for the PR.
            directory: Directory inside the repo to place the file.
            branch_name: Branch to create for the changes. Defaults to "mgym/upload-<job_id>".
            token: GitHub token. If None, read from GITHUB_TOKEN.
            commit_message: Commit message. Defaults to a message with job id.
            pr_title: Pull request title. Defaults to a title with job id.
            pr_body: Pull request body. Optional.
            pr_labels: Labels to add to the created pull request. Optional.
            clone_dir: Optional directory to perform clone/work. Defaults to a temp dir.
            payload: Optional data to write instead of self.as_dict().
            filename: Filename to write. Defaults to "<job_id>.json".
            append: If True and the file exists, append to it as a JSON array.

        Returns:
            The URL of the created pull request.
        """

        if not dry_run:
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
        if dry_run:
            temp_root = tempfile.mkdtemp(prefix="mgym-dryrun-")
            workdir = temp_root
        else:
            if clone_dir is None:
                temp_root = tempfile.mkdtemp(prefix="mgym-upload-")
                workdir = temp_root
            else:
                os.makedirs(clone_dir, exist_ok=True)
                workdir = clone_dir

        # Determine authenticated user and ensure a fork exists (skip network in dry-run)
        upstream_repo_url = f"https://github.com/{repo}.git"
        if not dry_run:
            login = self._get_authenticated_login(token)  # type: ignore[arg-type]
            self._ensure_fork(token=token, owner=owner, repo=repo_name, login=login)  # type: ignore[arg-type]
            fork_repo = f"{login}/{repo_name}"
            fork_push_url = f"https://x-access-token:{token}@github.com/{fork_repo}.git"

        repo_path = os.path.join(workdir, repo_name)

        try:
            # 1) Clone upstream base branch (or simulate in dry-run)
            if not dry_run:
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

                # 2) Add remote for fork (for checking/pushing)
                self._run(["git", "-C", repo_path, "remote", "add", "fork", fork_push_url])

                # 3) Create working branch (ensure uniqueness)
                if self._remote_branch_exists(repo_path, "fork", branch_name):
                    branch_name = self._next_available_branch_name(repo_path, "fork", branch_name)
                self._run(["git", "-C", repo_path, "checkout", "-b", branch_name])
            else:
                os.makedirs(repo_path, exist_ok=True)

            # 4) Write / update results file (JSON array, newest first)
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

            if not dry_run:
                # 5) Stage changes
                self._run(["git", "-C", repo_path, "add", os.path.relpath(out_path, repo_path)])
                # 6) Commit (author identity comes from local git config / env)
                self._run(["git", "-C", repo_path, "commit", "-m", commit_message])

                # 7) Push branch to fork
                self._run(["git", "-C", repo_path, "push", "fork", f"HEAD:{branch_name}"])

                # 8) Open PR on upstream via GitHub REST API (head=<login>:<branch>)
                compare_url = (
                    f"https://github.com/{owner}/{repo_name}/compare/"
                    f"{base_branch}...{login}:{branch_name}?expand=1"
                )
                try:
                    pr_url, pr_number = self._create_pull_request(
                        token=token,  # type: ignore[arg-type]
                        owner=owner,
                        repo=repo_name,
                        title=pr_title,
                        head=f"{login}:{branch_name}",
                        base=base_branch,
                        body=pr_body,
                    )
                    if pr_labels:
                        # Labels must be added via the issues API after PR creation.
                        self._add_labels_to_pull_request(
                            token=token,  # type: ignore[arg-type]
                            owner=owner,
                            repo=repo_name,
                            pr_number=pr_number,
                            labels=pr_labels,
                        )
                    return pr_url
                except Exception:
                    # Fallback: return compare URL so user can open PR manually in browser
                    if pr_labels:
                        labels_param = urllib.parse.quote(",".join(pr_labels))
                        return f"{compare_url}&labels={labels_param}"
                    return compare_url
            else:
                return (
                    "DRY-RUN: wrote mock file at "
                    + out_path
                    + f"; would create branch '{branch_name}' and open PR to {owner}/{repo_name} (base: {base_branch}) with title: {pr_title}"
                )
        finally:
            if temp_root and os.path.isdir(temp_root):
                # Keep workspace for inspection when dry-run
                if not dry_run:
                    shutil.rmtree(temp_root, ignore_errors=True)

    def _headers(
        self,
        token: str,
        *,
        accept: str = "application/vnd.github+json",
        content_type: Optional[str] = None,
    ) -> dict:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": accept,
            "User-Agent": "metriq-gym",
        }
        if content_type:
            headers["Content-Type"] = content_type
        return headers

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
    ) -> tuple[str, int]:
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
            headers=self._headers(token, content_type="application/json"),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"GitHub PR creation failed: {e.read().decode('utf-8')}") from e

        # Expect an 'html_url' field in the PR response
        pr_url = resp_data.get("html_url")
        pr_number = resp_data.get("number")
        if not pr_url or not pr_number:
            raise RuntimeError("GitHub PR creation: missing html_url or number in response")
        return pr_url, int(pr_number)

    def _add_labels_to_pull_request(
        self, *, token: str, owner: str, repo: str, pr_number: int, labels: list[str]
    ) -> None:
        api_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/labels"
        data = {"labels": labels}
        req = urllib.request.Request(
            api_url,
            data=json.dumps(data).encode("utf-8"),
            headers=self._headers(token, content_type="application/json"),
            method="POST",
        )
        try:
            urllib.request.urlopen(req).read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"GitHub label add failed: {e.read().decode('utf-8')}") from e

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
        # Try numeric suffixes to avoid collisions: base-2, base-3, ... up to limit
        for i in range(2, MAX_BRANCH_SUFFIX_ATTEMPTS):
            candidate = f"{base}-{i}"
            if not self._remote_branch_exists(repo_path, remote, candidate):
                return candidate

        candidate = f"{base}-{int(time.time())}"
        return candidate

    def _get_authenticated_login(self, token: str) -> str:
        req = urllib.request.Request(
            "https://api.github.com/user",
            headers=self._headers(token),
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
            headers=self._headers(token, content_type="application/json"),
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
            headers=self._headers(token),
            method="GET",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.getcode() == 200
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            raise
