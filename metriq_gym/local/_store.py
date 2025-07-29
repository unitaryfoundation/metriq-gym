import json
from typing import Any

from metriq_gym.paths import get_local_simulator_cache_dir


def write(job_id: str, payload: dict[str, Any]) -> None:
    (get_local_simulator_cache_dir() / f"{job_id}.json").write_text(json.dumps(payload))


def read(job_id: str) -> dict[str, Any] | None:
    p = get_local_simulator_cache_dir() / f"{job_id}.json"
    return json.loads(p.read_text()) if p.exists() else None
