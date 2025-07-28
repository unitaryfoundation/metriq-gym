import os
from pathlib import Path
from importlib.metadata import distribution, PackageNotFoundError
from platformdirs import user_data_dir, user_cache_dir


def _load_app_meta(dist_name: str) -> tuple[str, str]:
    try:
        dist = distribution(dist_name)
        meta = dist.metadata
        name = meta["Name"] or dist_name
        author = meta.get("Author", "unknown-author")
    except PackageNotFoundError:
        name, author = dist_name, "unknown-author"
    return name, author


_APP_NAME, _APP_AUTHOR = _load_app_meta("metriq-gym")


def get_data_db_path(filename: str = "localdb.jsonl") -> Path:
    base = Path(os.environ.get("MGYM_LOCAL_DB_DIR", user_data_dir(_APP_NAME, _APP_AUTHOR)))
    base.mkdir(parents=True, exist_ok=True)
    return base / filename


def get_local_simulator_cache_dir(subfolder: str = "jobs") -> Path:
    base = Path(
        os.environ.get("MGYM_LOCAL_SIMULATOR_CACHE_DIR", user_cache_dir(_APP_NAME, _APP_AUTHOR))
    )
    jobs = base / subfolder
    jobs.mkdir(parents=True, exist_ok=True)
    return jobs


# Run as a script to test paths
if __name__ == "__main__":
    print("Persistent DB path:  ", get_data_db_path())
    print("Jobs cache directory:", get_local_simulator_cache_dir())
