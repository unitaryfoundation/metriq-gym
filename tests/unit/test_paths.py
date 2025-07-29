import tempfile
from pathlib import Path
from metriq_gym.paths import get_data_db_path, _APP_NAME, _APP_AUTHOR
from platformdirs import user_data_dir
from metriq_gym.paths import (
    get_local_simulator_cache_dir,
    DEFAULT_DB_FILENAME,
    DEFAULT_JOBS_SUBFOLDER,
)
from platformdirs import user_cache_dir


def test_get_data_db_path_default(monkeypatch):
    monkeypatch.delenv("MGYM_LOCAL_DB_DIR", raising=False)
    expected_base = Path(user_data_dir(_APP_NAME, _APP_AUTHOR))
    path = get_data_db_path()
    assert isinstance(path, Path)
    assert path.name == DEFAULT_DB_FILENAME
    assert str(path.parent).startswith(str(expected_base))
    assert path.parent.exists()


def test_get_data_db_path_env(monkeypatch):
    test_file_name = "testfile.jsonl"
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("MGYM_LOCAL_DB_DIR", tmpdir)
        path = get_data_db_path(test_file_name)
        assert isinstance(path, Path)
        assert path.name == test_file_name
        assert str(path.parent) == tmpdir
        assert path.parent.exists()


def test_get_local_simulator_cache_dir_default(monkeypatch):
    monkeypatch.delenv("MGYM_LOCAL_SIMULATOR_CACHE_DIR", raising=False)
    expected_base = Path(user_cache_dir(_APP_NAME, _APP_AUTHOR))
    path = get_local_simulator_cache_dir()
    assert isinstance(path, Path)
    assert path.name == DEFAULT_JOBS_SUBFOLDER
    assert str(path.parent) == str(expected_base)
    assert path.exists()
    assert path.is_dir()


def test_get_local_simulator_cache_dir_env(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("MGYM_LOCAL_SIMULATOR_CACHE_DIR", tmpdir)
        subfolder = "custom_jobs"
        path = get_local_simulator_cache_dir(subfolder)
        assert isinstance(path, Path)
        assert path.name == subfolder
        assert str(path.parent) == tmpdir
        assert path.exists()
        assert path.is_dir()
