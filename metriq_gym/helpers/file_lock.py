"""Cross-process advisory locking for the local jobs database."""

from __future__ import annotations

import errno
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:  # POSIX
    import fcntl
except ImportError:  # pragma: no cover - platform dependent
    fcntl = None  # type: ignore[assignment]

try:  # Windows
    import msvcrt
except ImportError:  # pragma: no cover - platform dependent
    msvcrt = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

LOCK_SUFFIX = ".lock"
# Raised when the lock is held elsewhere. Anything else (an unsupported
# filesystem, a bad descriptor) is permanent and must not be retried.
CONTENTION_ERRNOS = frozenset(
    {
        errno.EACCES,
        errno.EAGAIN,
        errno.EWOULDBLOCK,
        errno.EDEADLK,
    }
)
DEFAULT_TIMEOUT = 30.0
DEFAULT_POLL_INTERVAL = 0.05


def lock_path_for(path: Path) -> Path:
    """Return the sidecar lock file for ``path``.

    The lock lives beside the data file rather than on it. Rewrites replace the
    data file via ``Path.replace``, so a lock held on the data file would be a
    lock on an inode that no longer has that name, and a process opening the
    new file would not see it.
    """
    return path.with_name(path.name + LOCK_SUFFIX)


def _try_acquire(fd: int) -> None:
    """Attempt a non-blocking exclusive lock, raising OSError if held."""
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    elif msvcrt is not None:  # pragma: no cover - Windows only
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)


def _release(fd: int) -> None:
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
    elif msvcrt is not None:  # pragma: no cover - Windows only
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


@contextmanager
def exclusive_lock(
    path: Path,
    timeout: float = DEFAULT_TIMEOUT,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
) -> Iterator[None]:
    """Hold an exclusive advisory lock covering ``path`` for the duration.

    Polls rather than blocking so a stuck holder surfaces as a TimeoutError
    instead of hanging the CLI indefinitely.

    Raises:
        TimeoutError: if the lock is still held after ``timeout`` seconds.
    """
    if fcntl is None and msvcrt is None:  # pragma: no cover - platform dependent
        logger.warning(
            "No file locking available on this platform; concurrent metriq-gym "
            "processes may corrupt %s",
            path,
        )
        yield
        return

    lock_file = lock_path_for(path)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_file, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                _try_acquire(fd)
                break
            except OSError as exc:
                if exc.errno not in CONTENTION_ERRNOS:
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out after {timeout}s waiting for the lock on {lock_file}. "
                        "Another metriq-gym process may be stuck."
                    )
                time.sleep(poll_interval)
        try:
            yield
        finally:
            _release(fd)
    finally:
        os.close(fd)
