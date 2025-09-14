from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from metriq_gym.paths import get_local_simulator_cache_dir


REF_PREFIX = "qnexus_ref_"
META_PREFIX = "qnexus_meta_"


def _ref_path(job_id: str) -> Path:
    return get_local_simulator_cache_dir() / f"{REF_PREFIX}{job_id}.bin"


def _meta_path(job_id: str) -> Path:
    return get_local_simulator_cache_dir() / f"{META_PREFIX}{job_id}.json"


def write_meta(job_id: str, payload: dict[str, Any]) -> None:
    try:
        p = _meta_path(job_id)
        p.write_text(json.dumps(payload))
    except Exception:
        pass


def read_meta(job_id: str) -> dict[str, Any] | None:
    try:
        p = _meta_path(job_id)
        if not p.exists():
            return None
        return json.loads(p.read_text())
    except Exception:
        return None
