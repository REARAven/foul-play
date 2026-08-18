"""Shared strict file primitives for independent team-era state stores."""

from __future__ import annotations

from dataclasses import dataclass, field
import errno
import os
from pathlib import Path
import stat
import tempfile
from typing import NoReturn

from .errors import BlindPoolValidationError


_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS = frozenset(
    value
    for value in (
        errno.EINVAL,
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
    )
    if value is not None
)


def _fail(prefix: str, suffix: str, message: str) -> NoReturn:
    raise BlindPoolValidationError("{}_{}".format(prefix, suffix), message) from None


def _is_within(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def _is_link_or_reparse(path: Path, *, prefix: str) -> bool:
    try:
        info = os.lstat(path)
    except OSError:
        _fail(prefix, "path_unavailable", "Team ladder path is unavailable")
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT
    )


def _validate_regular_file(path: Path, *, prefix: str) -> None:
    if _is_link_or_reparse(path, prefix=prefix):
        _fail(prefix, "path_unsafe", "Team ladder state must not be a link")
    try:
        info = os.lstat(path)
    except OSError:
        _fail(prefix, "path_unavailable", "Team ladder state is unavailable")
    if not stat.S_ISREG(info.st_mode):
        _fail(prefix, "path_invalid", "Team ladder state must be a regular file")
    if getattr(info, "st_nlink", 1) != 1:
        _fail(prefix, "path_unsafe", "Team ladder state must not be hard linked")


@dataclass(frozen=True, slots=True, repr=False)
class TeamStateFileConfig:
    path: Path = field(repr=False)
    private_root: Path = field(repr=False)
    repository_root: Path = field(repr=False)
    collision_paths: tuple[Path, ...] = field(repr=False)
    code_prefix: str

    @property
    def lock_path(self) -> Path:
        return self.path.with_name(self.path.name + ".lock")

    def __repr__(self) -> str:
        return "TeamStateFileConfig(configured=True)"


def validate_team_state_file_config(
    path: str | Path,
    *,
    private_root: str | Path,
    collision_paths: tuple[str | Path, ...],
    code_prefix: str,
    repository_root: str | Path | None = None,
) -> TeamStateFileConfig:
    """Resolve one explicit external state path and reject link/collision tricks."""

    if not isinstance(code_prefix, str) or not code_prefix:
        raise TypeError("code_prefix must be a non-empty string")
    try:
        candidate = Path(path)
        configured_private_root = Path(private_root)
        configured_collisions = tuple(Path(item) for item in collision_paths)
    except (TypeError, ValueError):
        _fail(code_prefix, "config_invalid", "Team ladder configuration is invalid")
    if not candidate.is_absolute() or not configured_private_root.is_absolute():
        _fail(code_prefix, "path_not_absolute", "Team ladder path must be absolute")
    if any(not item.is_absolute() for item in configured_collisions):
        _fail(code_prefix, "config_invalid", "Team ladder collision path is invalid")
    try:
        repository = Path(
            repository_root or Path(__file__).resolve().parents[3]
        ).resolve(strict=True)
        private = configured_private_root.resolve(strict=True)
        parent = candidate.parent.resolve(strict=True)
        collisions = tuple(item.resolve(strict=False) for item in configured_collisions)
    except (OSError, RuntimeError):
        _fail(code_prefix, "path_unavailable", "Team ladder path is unavailable")
    raw_ancestor = candidate.parent
    while raw_ancestor != raw_ancestor.parent:
        if raw_ancestor.exists() and _is_link_or_reparse(
            raw_ancestor, prefix=code_prefix
        ):
            _fail(code_prefix, "parent_unsafe", "Team ladder parent is unsafe")
        raw_ancestor = raw_ancestor.parent
    if not parent.is_dir():
        _fail(code_prefix, "parent_unsafe", "Team ladder parent is unsafe")
    resolved = parent / candidate.name
    if candidate.exists() or candidate.is_symlink():
        _validate_regular_file(candidate, prefix=code_prefix)
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            _fail(code_prefix, "path_unavailable", "Team ladder path is unavailable")
    lock = parent / (candidate.name + ".lock")
    raw_lock = candidate.with_name(candidate.name + ".lock")
    if raw_lock.exists() or raw_lock.is_symlink():
        _validate_regular_file(raw_lock, prefix=code_prefix)
        lock = raw_lock.resolve(strict=True)
    if _is_within(resolved, repository):
        _fail(code_prefix, "path_not_external", "Team ladder state must be external")
    if _is_within(resolved, private):
        _fail(
            code_prefix,
            "path_in_deployment",
            "Team ladder state must be outside deployments",
        )
    if _is_within(lock, repository) or _is_within(lock, private):
        _fail(code_prefix, "lock_path_unsafe", "Team ladder lock path is unsafe")
    forbidden = set(collisions)
    forbidden.update(item.with_name(item.name + ".lock") for item in collisions)
    forbidden.update(item.with_name(item.name + ".owner.lock") for item in collisions)
    if resolved == lock or resolved in forbidden or lock in forbidden:
        _fail(code_prefix, "path_collision", "Team ladder state path collides")
    return TeamStateFileConfig(resolved, private, repository, collisions, code_prefix)


def _identity(info: os.stat_result, *, opened: bool = False) -> tuple[int, ...]:
    values = (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_size),
        int(info.st_mtime_ns),
    )
    if opened:
        return (*values, int(getattr(info, "st_file_attributes", 0)))
    return (
        *values,
        int(info.st_ctime_ns),
        int(getattr(info, "st_file_attributes", 0)),
    )


def stable_read_bytes(config: TeamStateFileConfig) -> bytes:
    path = config.path
    if not path.exists() and not path.is_symlink():
        _fail(
            config.code_prefix,
            "not_initialized",
            "Team ladder state is not initialized",
        )
    _validate_regular_file(path, prefix=config.code_prefix)
    try:
        before = os.lstat(path)
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or _identity(
                before, opened=True
            ) != _identity(opened, opened=True):
                _fail(config.code_prefix, "file_changed", "Team ladder state changed")
            raw = stream.read(before.st_size + 1)
            if len(raw) != before.st_size:
                _fail(config.code_prefix, "file_changed", "Team ladder state changed")
        after = os.lstat(path)
    except BlindPoolValidationError:
        raise
    except OSError:
        _fail(config.code_prefix, "file_unreadable", "Team ladder state is unreadable")
    if _is_link_or_reparse(path, prefix=config.code_prefix) or _identity(
        before
    ) != _identity(after):
        _fail(config.code_prefix, "file_changed", "Team ladder state changed")
    return raw


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
            return
        raise
    try:
        os.fsync(descriptor)
    except OSError as error:
        if error.errno not in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
            raise
    finally:
        os.close(descriptor)


def write_atomic(
    config: TeamStateFileConfig,
    payload: bytes,
    *,
    replace_existing: bool,
) -> None:
    descriptor: int | None = None
    temporary: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=".{}-".format(config.path.name),
            suffix=".tmp",
            dir=config.path.parent,
        )
        temporary = Path(raw_path)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = None
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        if replace_existing:
            os.replace(temporary, config.path)
            temporary = None
        else:
            try:
                os.link(temporary, config.path)
            except FileExistsError:
                _fail(
                    config.code_prefix,
                    "target_exists",
                    "Team ladder state already exists",
                )
            temporary.unlink()
            temporary = None
        _fsync_directory(config.path.parent)
    except BlindPoolValidationError:
        raise
    except OSError:
        _fail(
            config.code_prefix,
            "atomic_write_failed",
            "Team ladder state could not be persisted",
        )
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
