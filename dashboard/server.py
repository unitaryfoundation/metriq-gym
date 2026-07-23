"""Local server for the metriq-gym jobs dashboard.

Serves index.html plus a small JSON API over the local metriq-gym job database.
Run from the repo root so the metriq-gym environment is available:

    UV_PROJECT_ENVIRONMENT=~/.venvs/metriq-gym uv run python dashboard/server.py

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

PORT = 8787
DASHBOARD_DIR = Path(__file__).resolve().parent
STATE_FILE = DASHBOARD_DIR / "state.json"
INDEX_FILE = DASHBOARD_DIR / "index.html"

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
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {"uploads": {}, "polls": {}}


def save_state(state: dict) -> None:
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_FILE)


def read_jobs_raw() -> list[dict]:
    path = db_path()
    jobs = []
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


def short_device(name: str) -> str:
    # Braket device ARNs -> trailing segment, e.g. "Cepheus-1-108Q"
    if name.startswith("arn:"):
        return name.rsplit("/", 1)[-1]
    return name


def num_qubits(params: dict) -> int | None:
    for key in ("num_qubits", "qubits", "width", "max_qubits"):
        if key in params:
            v = params[key]
            if isinstance(v, bool):
                return None
            if isinstance(v, int):
                return v
            if isinstance(v, float) and v.is_integer():
                return int(v)
    return None


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
        "queue_position": min(positions) if positions else None,
        "at": datetime.now(timezone.utc).isoformat(),
        "detail": text[-2000:],
    }


def job_state(raw: dict, state: dict) -> tuple[str, int | None]:
    jid = raw["id"]
    if jid in state["uploads"]:
        return "uploaded", None
    if raw.get("result_data") is not None:
        return "ready_to_upload", None
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
        st, qpos = job_state(raw, state)
        upload = state["uploads"].get(raw["id"], {})
        poll = state["polls"].get(raw["id"], {})
        out.append(
            {
                "id": raw["id"],
                "benchmark": raw["job_type"],
                "provider": raw.get("provider_name"),
                "device": short_device(raw.get("device_name") or ""),
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
    return {"ok": ok, "output": text[-1000:]}


PR_URL_RE = re.compile(r"(?:pull request|create the PR): (\S+)")


def upload_job(job_id: str) -> dict:
    proc = subprocess.run(
        mgym_cmd() + ["job", "upload", job_id],
        capture_output=True,
        text=True,
        timeout=900,
    )
    text = proc.stdout + "\n" + proc.stderr
    match = PR_URL_RE.search(text)
    ok = ("Opened pull request" in text or "Branch pushed" in text) and proc.returncode == 0
    if ok:
        with _state_lock:
            state = load_state()
            state["uploads"][job_id] = {
                "pr_url": match.group(1) if match else None,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            }
            save_state(state)
    return {"ok": ok, "pr_url": match.group(1) if match else None, "output": text[-3000:]}


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

    def do_POST(self):
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


def main() -> None:
    print(f"metriq-gym jobs dashboard: http://localhost:{PORT}")
    print(f"reading jobs from: {db_path()}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
