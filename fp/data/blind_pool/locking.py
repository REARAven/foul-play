"""Bounded standard-library file locking for bag-state transactions."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from types import TracebackType
from typing import BinaryIO

from .errors import BlindPoolValidationError


_MUTEX_INDEX_GUARD = threading.Lock()
_MUTEXES: dict[Path, threading.Lock] = {}


def _mutex_for(path: Path) -> threading.Lock:
    with _MUTEX_INDEX_GUARD:
        return _MUTEXES.setdefault(path, threading.Lock())


class BlindPoolStateLock:
    """An in-process mutex plus a Windows/POSIX advisory file lock."""

    def __init__(
        self,
        lock_path: Path,
        *,
        timeout_seconds: float = 5.0,
        poll_seconds: float = 0.025,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise BlindPoolValidationError(
                "state_lock_timeout_invalid",
                "Blind Ladder state lock timeout must be positive",
            )
        try:
            self._path = Path(lock_path).resolve(strict=False)
        except (TypeError, OSError, RuntimeError):
            raise BlindPoolValidationError(
                "state_lock_path_invalid",
                "Blind Ladder state lock path is invalid",
            ) from None
        self._timeout = float(timeout_seconds)
        self._poll = max(0.001, float(poll_seconds))
        self._mutex = _mutex_for(self._path)
        self._stream: BinaryIO | None = None
        self._mutex_held = False

    def __repr__(self) -> str:
        return "BlindPoolStateLock(configured=True)"

    def __enter__(self) -> BlindPoolStateLock:
        deadline = time.monotonic() + self._timeout
        if not self._mutex.acquire(timeout=self._timeout):
            self._timeout_error()
        self._mutex_held = True
        try:
            self._stream = self._path.open("a+b")
            self._prepare_lock_byte()
            while True:
                try:
                    self._acquire_os_lock()
                    return self
                except (BlockingIOError, PermissionError):
                    if time.monotonic() >= deadline:
                        self._timeout_error()
                    time.sleep(min(self._poll, max(0.0, deadline - time.monotonic())))
                except OSError:
                    raise BlindPoolValidationError(
                        "state_lock_failed",
                        "Blind Ladder state lock could not be acquired",
                    ) from None
        except BlindPoolValidationError:
            self._close_and_release_mutex()
            raise
        except OSError:
            self._close_and_release_mutex()
            raise BlindPoolValidationError(
                "state_lock_failed",
                "Blind Ladder state lock could not be acquired",
            ) from None

    def _prepare_lock_byte(self) -> None:
        assert self._stream is not None
        self._stream.seek(0, os.SEEK_END)
        if self._stream.tell() == 0:
            self._stream.write(b"\0")
            self._stream.flush()
        self._stream.seek(0)

    def _acquire_os_lock(self) -> None:
        assert self._stream is not None
        if os.name == "nt":
            import msvcrt

            self._stream.seek(0)
            msvcrt.locking(self._stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _release_os_lock(self) -> None:
        assert self._stream is not None
        if os.name == "nt":
            import msvcrt

            self._stream.seek(0)
            msvcrt.locking(self._stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)

    def _timeout_error(self) -> None:
        raise BlindPoolValidationError(
            "state_lock_timeout",
            "Blind Ladder state lock acquisition timed out",
        ) from None

    def _close_and_release_mutex(self) -> bool:
        close_failed = False
        if self._stream is not None:
            try:
                self._stream.close()
            except OSError:
                close_failed = True
            finally:
                self._stream = None
        if self._mutex_held:
            self._mutex.release()
            self._mutex_held = False
        return close_failed

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        release_failed = False
        if self._stream is not None:
            try:
                self._release_os_lock()
            except OSError:
                release_failed = True
        close_failed = self._close_and_release_mutex()
        if (release_failed or close_failed) and exc_type is None:
            raise BlindPoolValidationError(
                "state_lock_failed",
                "Blind Ladder state lock could not be released",
            ) from None
        return False
