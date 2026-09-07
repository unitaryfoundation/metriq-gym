"""Concurrency tests for JobManager's on-disk database.

The interesting case is cross-process rather than cross-thread: metriq-gym
spawns `mgym` subprocesses (the dashboard does this per job), and each builds
its own JobManager, so an in-process lock alone would serialize nothing.
"""

import multiprocessing as mp
import threading
from datetime import datetime
from pathlib import Path

import pytest

from metriq_gym.constants import JobType
from metriq_gym.helpers.file_lock import exclusive_lock, lock_path_for
from metriq_gym.job_manager import JobManager, MetriqGymJob


def _job(job_id: str) -> MetriqGymJob:
    return MetriqGymJob(
        id=job_id,
        job_type=JobType("BSEQ"),
        params={},
        data={},
        provider_name="local",
        device_name="aer_simulator",
        dispatch_time=datetime.now(),
    )


@pytest.fixture
def jobs_file(tmp_path: Path) -> Path:
    return tmp_path / "localdb.jsonl"


def test_reload_does_not_duplicate_line_entries(jobs_file):
    """Regression: _load_jobs used to append to the previous entries.

    One job then became N duplicate lines on disk after N reloads plus a
    rewrite. Latent while _load_jobs was only called from __init__, and live
    as soon as the write paths reload under the lock.
    """
    manager = JobManager(jobs_file=jobs_file)
    manager.add_job(_job("a"))

    for _ in range(3):
        manager._load_jobs()

    assert len(manager._line_entries) == 1
    manager._rewrite_jobs_file()
    assert len(jobs_file.read_text().strip().splitlines()) == 1


def test_add_job_picks_up_writes_from_another_manager(jobs_file):
    """A stale in-memory view must not drop another writer's records."""
    first = JobManager(jobs_file=jobs_file)
    second = JobManager(jobs_file=jobs_file)

    first.add_job(_job("from-first"))
    second.add_job(_job("from-second"))

    on_disk = {line for line in jobs_file.read_text().strip().splitlines()}
    assert len(on_disk) == 2
    assert {j.id for j in JobManager(jobs_file=jobs_file).get_jobs()} == {
        "from-first",
        "from-second",
    }


def test_delete_does_not_discard_a_concurrent_append(jobs_file):
    """delete_job rewrites the whole file, so it must reload first."""
    deleter = JobManager(jobs_file=jobs_file)
    deleter.add_job(_job("doomed"))

    other = JobManager(jobs_file=jobs_file)
    other.add_job(_job("survivor"))

    deleter.delete_job("doomed")

    assert {j.id for j in JobManager(jobs_file=jobs_file).get_jobs()} == {"survivor"}


def test_update_does_not_discard_a_concurrent_append(jobs_file):
    updater = JobManager(jobs_file=jobs_file)
    updater.add_job(_job("target"))

    other = JobManager(jobs_file=jobs_file)
    other.add_job(_job("survivor"))

    job = updater.get_job("target")
    job.error = "boom"
    updater.update_job(job)

    reloaded = JobManager(jobs_file=jobs_file)
    assert {j.id for j in reloaded.get_jobs()} == {"target", "survivor"}
    # error is normalized to a dict on serialization round-trip
    assert reloaded.get_job("target").error == {"message": "boom"}


def test_threads_do_not_interleave_writes(jobs_file):
    JobManager(jobs_file=jobs_file)

    def worker(index: int) -> None:
        JobManager(jobs_file=jobs_file).add_job(_job(f"job-{index}"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len({j.id for j in JobManager(jobs_file=jobs_file).get_jobs()}) == 8


def _child_add(path_str: str, index: int) -> None:
    JobManager(jobs_file=Path(path_str)).add_job(_job(f"proc-{index}"))


def test_separate_processes_do_not_clobber_each_other(jobs_file):
    """The case an in-process lock cannot cover."""
    JobManager(jobs_file=jobs_file)

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_child_add, args=(str(jobs_file), i)) for i in range(6)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)

    assert all(p.exitcode == 0 for p in procs)
    assert len({j.id for j in JobManager(jobs_file=jobs_file).get_jobs()}) == 6


def test_lock_is_held_against_a_second_holder(tmp_path):
    target = tmp_path / "localdb.jsonl"
    with exclusive_lock(target):
        with pytest.raises(TimeoutError):
            with exclusive_lock(target, timeout=0.2, poll_interval=0.01):
                pass


def test_lock_file_sits_beside_the_data_file(tmp_path):
    """The lock must not be the data file, which rewrites replace."""
    target = tmp_path / "localdb.jsonl"
    assert lock_path_for(target) == tmp_path / "localdb.jsonl.lock"
