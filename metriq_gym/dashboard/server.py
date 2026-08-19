"""Local server for the metriq-gym jobs dashboard.

Serves index.html plus a small JSON API over the local metriq-gym job database.
Launch via the CLI:

    mgym dashboard [--port 8787]

Then open http://localhost:8787
"""

import json
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import cast

PORT = 8787
# Browsers send an Origin header on cross-site POSTs; only accept our own page
# (or clients like curl that send no Origin) so a drive-by webpage can't CSRF
# poll/upload/delete actions against the local server. Recomputed in main() so
# it tracks the serving port.
ALLOWED_ORIGINS = {f"http://localhost:{PORT}", f"http://127.0.0.1:{PORT}"}
DASHBOARD_DIR = Path(__file__).resolve().parent
INDEX_FILE = DASHBOARD_DIR / "index.html"


def state_file() -> Path:
    # Sidecar lives next to the local job db, not in the (possibly read-only)
    # installed package directory.
    return db_path().parent / "dashboard_state.json"


_state_lock = threading.Lock()
_poll_locks: dict[str, threading.Lock] = {}
_poll_locks_guard = threading.Lock()
# metriq-gym's JobManager has no file locking and rewrites localdb.jsonl wholesale,
# so two concurrent mgym subprocesses can clobber each other's writes (e.g. a poll
# finishing after a delete resurrects the deleted job). Serialize every subprocess
# that can write the db.
_mgym_write_lock = threading.Lock()


def db_path() -> Path:
    from metriq_gym.paths import get_data_db_path

    return get_data_db_path()


def mgym_cmd() -> list[str]:
    exe = shutil.which("mgym")
    if exe:
        return [exe]
    return [sys.executable, "-m", "metriq_gym.run"]


def load_state() -> dict:
    path = state_file()
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return {"uploads": {}, "polls": {}}


def save_state(state: dict) -> None:
    path = state_file()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)


def read_jobs_raw() -> list[dict]:
    path = db_path()
    jobs: list[dict] = []
    if not path.exists():
        return jobs
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                jobs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return jobs


def short_device(provider: str, name: str) -> str:
    # The same dataset key the job is stored and uploaded under.
    from metriq_gym.platform import canonical_device_name

    return canonical_device_name(provider or "", name)


def num_qubits(params: dict) -> int | None:
    from metriq_gym.job_manager import MetriqGymJob

    # Reuse MetriqGymJob.num_qubits rather than keeping a copy of its key
    # order/type handling in sync. The method only reads .params, and a real
    # MetriqGymJob can't be built here: deserialize() raises on records the
    # tolerant raw reader deliberately accepts (unknown job_type, bad dates).
    return MetriqGymJob.num_qubits(cast(MetriqGymJob, SimpleNamespace(params=params)))


QUEUE_POS_RE = re.compile(r"QUEUED(?:\s*\(position\s*(\d+)\))?", re.IGNORECASE)


def parse_poll_output(text: str) -> dict:
    """Reduce `mgym job poll` stdout to an aggregate provider status."""
    failed = bool(re.search(r"FAILED|CANCELLED", text))
    running = bool(re.search(r"\bRUNNING\b", text))
    positions = [int(m) for m in QUEUE_POS_RE.findall(text) if m]
    queued = bool(re.search(r"\bQUEUED\b", text))
    pending = "not yet completed" in text
    if failed:
        status = "failed"
    elif not pending:
        status = "completed"
    elif running:
        status = "running"
    elif queued:
        status = "queued"
    else:
        status = "pending"
    return {
        "status": status,
        # A job is done when its *slowest* task is, so across tasks the max
        # position is the meaningful one; min would show 3 while a sibling
        # task sits at 50.
        "queue_position": max(positions) if positions else None,
        "at": datetime.now(timezone.utc).isoformat(),
        "detail": text[-2000:],
    }


def job_state(raw: dict, state: dict) -> tuple[str, int | None]:
    jid = raw.get("id")
    if jid in state["uploads"]:
        return "uploaded", None
    if raw.get("result_data") is not None:
        return "ready_to_upload", None
    if raw.get("error") is not None:
        # Failure recorded on the job itself (dispatch raised, or a poll saw a
        # terminal provider failure); no need to wait for a dashboard poll.
        return "failed", None
    poll = state["polls"].get(jid)
    if poll:
        s = poll["status"]
        if s == "failed":
            return "failed", None
        if s == "running":
            return "running", None
        if s == "queued":
            return "queued", poll.get("queue_position")
        if s == "completed":
            # Provider says done but localdb has no results yet (poll parses lazily);
            # next successful poll writes result_data. Treat as running.
            return "running", None
        return "queued", poll.get("queue_position")
    return "unknown", None


def wire_jobs() -> list[dict]:
    with _state_lock:
        state = load_state()
    out = []
    for raw in read_jobs_raw():
        # Tolerate partial records the same way read_jobs_raw tolerates bad
        # lines: one row missing "id" must not 500 the whole /api/jobs list.
        # Without an id no action can target the job, so skip it entirely.
        if not raw.get("id"):
            continue
        st, qpos = job_state(raw, state)
        upload = state["uploads"].get(raw["id"], {})
        poll = state["polls"].get(raw["id"], {})
        out.append(
            {
                "id": raw["id"],
                "benchmark": raw.get("job_type") or "unknown",
                "provider": raw.get("provider_name"),
                "device": short_device(
                    raw.get("provider_name") or "", raw.get("device_name") or ""
                ),
                "device_full": raw.get("device_name"),
                "num_qubits": num_qubits(raw.get("params") or {}),
                "params": raw.get("params") or {},
                "suite_id": raw.get("suite_id"),
                "suite_name": raw.get("suite_name"),
                "dispatch_time": raw.get("dispatch_time"),
                "runtime_seconds": raw.get("runtime_seconds"),
                "app_version": raw.get("app_version"),
                "provider_job_ids": (raw.get("data") or {}).get("provider_job_ids") or [],
                "result_data": raw.get("result_data"),
                "state": st,
                "queue_position": qpos,
                "last_polled": poll.get("at"),
                "pr_url": upload.get("pr_url"),
                "uploaded_at": upload.get("uploaded_at"),
            }
        )
    out.sort(key=lambda j: j["dispatch_time"] or "", reverse=True)
    return out


def poll_job(job_id: str) -> dict:
    with _poll_locks_guard:
        lock = _poll_locks.setdefault(job_id, threading.Lock())
    with lock, _mgym_write_lock:
        proc = subprocess.run(
            mgym_cmd() + ["job", "poll", job_id],
            capture_output=True,
            text=True,
            timeout=600,
        )
        text = proc.stdout + "\n" + proc.stderr
        parsed = parse_poll_output(text)
        with _state_lock:
            state = load_state()
            state["polls"][job_id] = parsed
            save_state(state)
        return parsed


def _job_in_db(job_id: str) -> bool:
    return any(raw.get("id") == job_id for raw in read_jobs_raw())


def delete_job_record(job_id: str) -> dict:
    if not _job_in_db(job_id):
        return {"ok": False, "output": f"Job {job_id} not found in local db"}
    text = ""
    with _mgym_write_lock:
        try:
            proc = subprocess.run(
                mgym_cmd() + ["job", "delete", job_id],
                capture_output=True,
                text=True,
                timeout=300,
            )
            text = proc.stdout + "\n" + proc.stderr
        except subprocess.TimeoutExpired as e:
            text = f"(subprocess timed out; verifying against db) {e}"
    # The db is the ground truth: stdout matching is brittle (killed processes,
    # buffered output), and the rewrite may have landed even if the CLI died.
    ok = not _job_in_db(job_id)
    if ok:
        # Drop any dashboard-side state for the removed job
        with _state_lock:
            state = load_state()
            state["polls"].pop(job_id, None)
            state["uploads"].pop(job_id, None)
            save_state(state)
        with _poll_locks_guard:
            _poll_locks.pop(job_id, None)
    return {"ok": ok, "output": text[-1000:]}


# `mgym job upload` exits 0 on every path (including "✗ Upload failed"), so the
# printed URL is the only reliable success artifact: a PR URL when the PR was
# opened, or a compare URL when the branch was pushed for a manual PR. Match the
# URL shapes rather than the ✓ wording, which can change.
PR_URL_RE = re.compile(r"https://\S+/pull/\d+")
COMPARE_URL_RE = re.compile(r"https://\S+/compare/\S+")


def upload_job(job_id: str) -> dict:
    # Upload can also write the db: fetch_result() persists result_data via
    # job_manager.update_job(), so serialize with the other db-writing subprocesses.
    with _mgym_write_lock:
        proc = subprocess.run(
            mgym_cmd() + ["job", "upload", job_id],
            capture_output=True,
            text=True,
            timeout=900,
        )
    text = proc.stdout + "\n" + proc.stderr
    match = PR_URL_RE.search(text) or COMPARE_URL_RE.search(text)
    url = match.group(0) if match else None
    ok = url is not None and proc.returncode == 0
    if ok:
        with _state_lock:
            state = load_state()
            state["uploads"][job_id] = {
                "pr_url": url,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            }
            save_state(state)
    return {"ok": ok, "pr_url": url, "output": text[-3000:]}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX_FILE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/jobs":
            try:
                self._json({"jobs": wire_jobs()})
            except Exception as e:
                self._json({"error": str(e)}, 500)
        else:
            self._send(404, b"not found", "text/plain")

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return True  # non-browser clients (curl, scripts) send no Origin
        return origin in ALLOWED_ORIGINS

    def do_POST(self):
        if not self._origin_ok():
            self._json({"error": "cross-origin request rejected"}, 403)
            return
        try:
            if self.path.startswith("/api/poll/"):
                self._json(poll_job(self.path.rsplit("/", 1)[-1]))
            elif self.path.startswith("/api/upload/"):
                self._json(upload_job(self.path.rsplit("/", 1)[-1]))
            elif self.path.startswith("/api/delete/"):
                self._json(delete_job_record(self.path.rsplit("/", 1)[-1]))
            else:
                self._send(404, b"not found", "text/plain")
        except subprocess.TimeoutExpired:
            self._json({"error": "mgym subprocess timed out"}, 504)
        except Exception as e:
            self._json({"error": str(e)}, 500)

    def log_message(self, fmt, *args):
        pass  # keep the terminal quiet


def main(port: int = PORT) -> None:
    global ALLOWED_ORIGINS
    ALLOWED_ORIGINS = {f"http://localhost:{port}", f"http://127.0.0.1:{port}"}
    print(f"metriq-gym jobs dashboard: http://localhost:{port}")
    print(f"reading jobs from: {db_path()}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
