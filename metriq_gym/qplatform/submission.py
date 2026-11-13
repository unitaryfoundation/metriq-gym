"""Record accepted provider jobs before submitting more work.

Some SDKs submit a circuit list as several independent jobs but return their IDs
only after the whole list succeeds. Submit those circuits individually so a later
failure cannot hide earlier jobs. Braket submissions are consequently sequential;
the submitted jobs still execute asynchronously. An SDK failure before returning
a job ID remains ambiguous and must not be retried automatically.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from qbraid import QuantumDevice, QuantumJob
from qbraid.runtime import AzureQuantumDevice, BraketDevice
from qbraid.runtime.origin import OriginDevice


_recorder: ContextVar[Callable[[list[str]], None] | None] = ContextVar(
    "submission_recorder", default=None
)


@contextmanager
def track_submissions(callback: Callable[[list[str]], None]) -> Iterator[None]:
    """Record provider IDs in this dispatch context, restoring it on exit."""
    token = _recorder.set(callback)
    try:
        yield
    finally:
        _recorder.reset(token)


def record_submission[T: QuantumJob | list[QuantumJob]](jobs: T) -> T:
    """Checkpoint a successful submission and return its job objects unchanged.

    A checkpoint failure propagates immediately, preventing the caller from
    submitting the next batch. Outside a dispatch context this is a no-op.
    """
    callback = _recorder.get()
    if callback is not None:
        ids = [job.id for job in jobs] if isinstance(jobs, list) else [jobs.id]
        if ids:
            callback(ids)
    return jobs


def _submits_individual_jobs(device: QuantumDevice) -> bool:
    return isinstance(device, (AzureQuantumDevice, BraketDevice)) or (
        isinstance(device, OriginDevice) and device.profile.simulator is True
    )


def _flatten_programs(programs: list[Any]) -> list[Any]:
    # Runtime conversions can expand one source circuit into several programs.
    return [
        program for item in programs for program in (item if isinstance(item, list) else [item])
    ]


def submit_circuits(
    device: QuantumDevice, run_input: Any, *args: Any, **kwargs: Any
) -> QuantumJob | list[QuantumJob]:
    """Submit prepared circuits, checkpointing each independent provider job.

    Native batch providers retain their batch API. The input must already have
    passed the provider's runtime preparation and validation.
    """
    if not _submits_individual_jobs(device) or not isinstance(run_input, list):
        return record_submission(device.submit(run_input, *args, **kwargs))

    jobs = []
    for program in _flatten_programs(run_input):
        submitted = record_submission(device.submit(program, *args, **kwargs))
        jobs.extend(submitted if isinstance(submitted, list) else [submitted])
    return jobs


def run_circuits(
    device: QuantumDevice, run_input: Any, *args: Any, **kwargs: Any
) -> QuantumJob | list[QuantumJob]:
    """Prepare and submit circuits without losing IDs after a partial failure.

    Complete preparation for all circuits before the first submission. Providers
    with native batches keep their own ``run`` implementations and optimizations.
    """
    if not _submits_individual_jobs(device):
        return record_submission(device.run(run_input, *args, **kwargs))

    is_batch = isinstance(run_input, list)
    programs = run_input if is_batch else [run_input]
    prepared = [device.apply_runtime_profile(program) for program in programs]
    return submit_circuits(device, prepared if is_batch else prepared[0], *args, **kwargs)
