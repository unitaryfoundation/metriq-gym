import os
from pathlib import Path
from importlib.metadata import distribution, PackageNotFoundError
from platformdirs import user_data_dir, user_cache_dir

AUTHOR_FALLBACK = "Unitary Foundation"
DEFAULT_DB_FILENAME = "localdb.jsonl"
DEFAULT_JOBS_SUBFOLDER = "jobs"


def _load_app_meta(dist_name: str) -> tuple[str, str]:
    try:
        dist = distribution(dist_name)
        meta = dist.metadata
        name = meta["Name"] or dist_name
        author = meta.get("Author") or meta.get("Author-email") or AUTHOR_FALLBACK
    except PackageNotFoundError:
        name, author = dist_name, AUTHOR_FALLBACK
    return name, author


_APP_NAME, _APP_AUTHOR = _load_app_meta("metriq-gym")


def get_data_db_path(filename: str = DEFAULT_DB_FILENAME) -> Path:
    """
    Returns the full path to the local database file, creating the parent directory if it does not exist.

    If the environment variable 'MGYM_LOCAL_DB_DIR' is set, its value is used as the base directory.
    Otherwise, a user-specific data directory is used. The function ensures the base directory exists.

    Args:
        filename: The name of the database file. Defaults to DEFAULT_DB_FILENAME.

    Returns:
        The full path to the database file.
    """
    base = Path(os.environ.get("MGYM_LOCAL_DB_DIR") or user_data_dir(_APP_NAME, _APP_AUTHOR))
    base.mkdir(parents=True, exist_ok=True)
    return base / filename


def get_local_simulator_cache_dir(subfolder: str = DEFAULT_JOBS_SUBFOLDER) -> Path:
    """
    Returns the path to the local simulator cache directory, creating it if it does not exist.

    The base directory is determined by the `MGYM_LOCAL_SIMULATOR_CACHE_DIR` environment variable if set,
    otherwise it defaults to the user's cache directory for the application. A subfolder can be specified,
    which will be created under the base directory.

    Args:
        subfolder: Name of the subfolder within the cache directory. Defaults to DEFAULT_JOBS_SUBFOLDER.

    Returns:
        The full path to the cache subfolder, guaranteed to exist.
    """
    base = Path(
        os.environ.get("MGYM_LOCAL_SIMULATOR_CACHE_DIR") or user_cache_dir(_APP_NAME, _APP_AUTHOR)
    )
    jobs = base / subfolder
    jobs.mkdir(parents=True, exist_ok=True)
    return jobs


# Run as a script to test paths
if __name__ == "__main__":
    print("Persistent DB path:  ", get_data_db_path())
    print("Jobs cache directory:", get_local_simulator_cache_dir())
