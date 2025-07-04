import json
import os
from pathlib import Path
from typing import Any
from platformdirs import user_cache_dir

_base = (
    Path(os.environ["MGR_LOCAL_JOB_DIR"]).expanduser()
    if "MGR_LOCAL_JOB_DIR" in os.environ
    else Path(user_cache_dir(appname="metriq_gym", appauthor="metriq"))
)

_DIR = _base / "jobs"
_DIR.mkdir(parents=True, exist_ok=True)


def write(job_id: str, payload: dict[str, Any]) -> None:
    (_DIR / f"{job_id}.json").write_text(json.dumps(payload))


def read(job_id: str) -> dict[str, Any] | None:
    p = _DIR / f"{job_id}.json"
    return json.loads(p.read_text()) if p.exists() else None
