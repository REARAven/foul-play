"""Process-lifetime ownership for one external Blind Ladder deployment."""

from __future__ import annotations

import os
from pathlib import Path
import stat
from types import TracebackType
from typing import NoReturn

from .config import validate_blind_pool_state_config
from .errors import BlindPoolValidationError
from .locking import BlindPoolStateLock
from .models import BlindPoolStateConfig


_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _fail(code: str, message: str) -> NoReturn:
    raise BlindPoolValidationError(code, message) from None


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _same_path(left: Path, right: Path) -> bool:
    return _path_key(left) == _path_key(right)


def _is_within(path: Path, directory: Path) -> bool:
    path_key = _path_key(path)
    directory_key = _path_key(directory)
    try:
        return os.path.commonpath((path_key, directory_key)) == directory_key
    except ValueError:
        return False


def _is_link_or_reparse(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & _REPARSE_POINT
    )


def _is_safe_owner_file(info: os.stat_result) -> bool:
    return (
        stat.S_ISREG(info.st_mode)
        and not _is_link_or_reparse(info)
        and int(info.st_nlink) == 1
    )


def _stat_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_size),
        int(getattr(info, "st_mtime_ns", int(info.st_mtime * 1_000_000_000))),
        int(getattr(info, "st_ctime_ns", int(info.st_ctime * 1_000_000_000))),
        int(getattr(info, "st_file_attributes", 0)),
        int(info.st_nlink),
    )


def _same_file_object(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        int(left.st_dev),
        int(left.st_ino),
        stat.S_IFMT(left.st_mode),
        int(getattr(left, "st_file_attributes", 0)),
    ) == (
        int(right.st_dev),
        int(right.st_ino),
        stat.S_IFMT(right.st_mode),
        int(getattr(right, "st_file_attributes", 0)),
    )


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _owner_path_candidate(state_path: Path) -> Path:
    """Derive the one committed deployment-owner control path."""

    return state_path.with_name(state_path.name + ".owner.lock")


def _resolve_owner_path(
    config: BlindPoolStateConfig,
    *,
    repository_root: str | Path | None = None,
    allow_missing_registry: bool = False,
) -> Path:
    """Derive and validate a sibling control file without exposing its path."""

    state_path = config.state_path
    candidate = _owner_path_candidate(state_path)
    invalid = False
    try:
        if candidate.exists() or candidate.is_symlink():
            info = os.lstat(candidate)
            if not _is_safe_owner_file(info):
                invalid = True
                resolved = None
            else:
                resolved = candidate.resolve(strict=True)
        else:
            resolved = candidate.parent.resolve(strict=True) / candidate.name
    except (OSError, RuntimeError, ValueError, TypeError):
        invalid = True
        resolved = None
    if invalid or resolved is None:
        _fail(
            "deployment_owner_path_invalid",
            "Blind Ladder deployment owner control is invalid",
        )

    protected_paths_invalid = False
    try:
        private_root = config.pool_config.private_root.resolve(strict=True)
        registry_candidate = config.pool_config.registry_path
        if registry_candidate.exists() or registry_candidate.is_symlink():
            registry_path = registry_candidate.resolve(strict=True)
        elif allow_missing_registry:
            registry_path = (
                registry_candidate.parent.resolve(strict=True) / registry_candidate.name
            )
        else:
            registry_path = registry_candidate.resolve(strict=True)
        transaction_lock = config.lock_path.resolve(strict=False)
        canonical_root = (private_root / "canonical").resolve(strict=False)
        repository = Path(repository_root or _repository_root()).resolve(strict=True)
    except (OSError, RuntimeError, ValueError, TypeError):
        protected_paths_invalid = True
        private_root = registry_path = transaction_lock = canonical_root = None
        repository = None
    if protected_paths_invalid:
        _fail(
            "deployment_owner_path_invalid",
            "Blind Ladder deployment owner control is invalid",
        )
    assert (
        private_root is not None
        and registry_path is not None
        and transaction_lock is not None
        and canonical_root is not None
        and repository is not None
    )
    if (
        not _is_within(resolved, private_root)
        or _is_within(resolved, canonical_root)
        or _is_within(resolved, repository)
        or _same_path(resolved, state_path)
        or _same_path(resolved, transaction_lock)
        or _same_path(resolved, registry_path)
    ):
        _fail(
            "deployment_owner_path_collision",
            "Blind Ladder deployment owner control conflicts with protected data",
        )
    return resolved


def _validate_acquired_owner_lock(
    expected_path: Path,
    lock: BlindPoolStateLock,
) -> None:
    """Bind the acquired descriptor to the validated non-link control path."""

    stream = lock._stream
    invalid = stream is None or not _same_path(lock._path, expected_path)
    try:
        if invalid:
            raise OSError
        path_before = os.lstat(expected_path)
        descriptor_before = os.fstat(stream.fileno())
        resolved = expected_path.resolve(strict=True)
        descriptor_after = os.fstat(stream.fileno())
        path_after = os.lstat(expected_path)
    except (OSError, RuntimeError, ValueError, TypeError):
        invalid = True
    else:
        invalid = (
            not _is_safe_owner_file(path_before)
            or not _is_safe_owner_file(descriptor_before)
            or not _is_safe_owner_file(descriptor_after)
            or not _is_safe_owner_file(path_after)
            or _stat_identity(path_before) != _stat_identity(path_after)
            or _stat_identity(descriptor_before) != _stat_identity(descriptor_after)
            or not _same_file_object(path_before, descriptor_before)
            or not _same_file_object(descriptor_after, path_after)
            or not _same_path(resolved, expected_path)
        )
    if invalid:
        _fail(
            "deployment_owner_path_changed",
            "Blind Ladder deployment owner control changed during acquisition",
        )


class BlindPoolDeploymentOwnerGuard:
    """One held OS-backed owner lock, separate from state transactions."""

    __slots__ = ("_held", "_lock")

    def __init__(self, lock: BlindPoolStateLock) -> None:
        self._lock = lock
        self._held = True

    @classmethod
    def acquire(
        cls,
        config: BlindPoolStateConfig,
        *,
        timeout_seconds: float = 0.25,
        repository_root: str | Path | None = None,
    ) -> BlindPoolDeploymentOwnerGuard:
        if not isinstance(config, BlindPoolStateConfig):
            _fail(
                "deployment_owner_config_invalid",
                "Blind Ladder deployment owner configuration is invalid",
            )
        config_invalid = False
        try:
            validated_config = validate_blind_pool_state_config(
                config,
                repository_root=repository_root,
            )
        except BlindPoolValidationError:
            config_invalid = True
            validated_config = None
        if config_invalid:
            _fail(
                "deployment_owner_config_invalid",
                "Blind Ladder deployment owner configuration is invalid",
            )
        assert validated_config is not None
        return _acquire_validated_blind_pool_deployment_owner(
            validated_config,
            timeout_seconds=timeout_seconds,
            repository_root=repository_root,
        )

    @classmethod
    def _acquire_path(
        cls,
        owner_path: Path,
        *,
        timeout_seconds: float,
    ) -> BlindPoolDeploymentOwnerGuard:
        lock = BlindPoolStateLock(owner_path, timeout_seconds=timeout_seconds)
        failed = False
        entered = False
        try:
            lock.__enter__()
            entered = True
            _validate_acquired_owner_lock(owner_path, lock)
        except BlindPoolValidationError:
            failed = True
        except BaseException:
            if entered:
                try:
                    lock.__exit__(None, None, None)
                except BlindPoolValidationError:
                    pass
            raise
        if failed and entered:
            try:
                lock.__exit__(None, None, None)
            except BlindPoolValidationError:
                pass
        if failed:
            _fail(
                "deployment_ownership_unavailable",
                "Blind Ladder deployment is already active or unavailable",
            )
        return cls(lock)

    @property
    def held(self) -> bool:
        return self._held

    def close(self) -> None:
        if not self._held:
            return
        failed = False
        try:
            self._lock.__exit__(None, None, None)
        except BlindPoolValidationError:
            failed = True
        self._held = False
        if failed:
            _fail(
                "deployment_owner_release_failed",
                "Blind Ladder deployment ownership could not be released",
            )

    def __enter__(self) -> BlindPoolDeploymentOwnerGuard:
        if not self._held:
            _fail(
                "deployment_owner_inactive",
                "Blind Ladder deployment ownership is not active",
            )
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> bool:
        self.close()
        return False

    def __repr__(self) -> str:
        return "BlindPoolDeploymentOwnerGuard(held={!r})".format(self._held)

    def __str__(self) -> str:
        return repr(self)

    def __reduce__(self) -> NoReturn:
        raise TypeError("BlindPoolDeploymentOwnerGuard serialization is disabled")


def acquire_blind_pool_deployment_owner(
    config: BlindPoolStateConfig,
    *,
    timeout_seconds: float = 0.25,
    repository_root: str | Path | None = None,
) -> BlindPoolDeploymentOwnerGuard:
    """Acquire the reusable runtime/maintenance ownership boundary."""

    return BlindPoolDeploymentOwnerGuard.acquire(
        config,
        timeout_seconds=timeout_seconds,
        repository_root=repository_root,
    )


def _acquire_validated_blind_pool_deployment_owner(
    config: BlindPoolStateConfig,
    *,
    timeout_seconds: float = 0.25,
    repository_root: str | Path | None = None,
    allow_missing_registry: bool = False,
) -> BlindPoolDeploymentOwnerGuard:
    """Acquire the standard owner for already resolved internal path policy."""

    owner_path = _resolve_owner_path(
        config,
        repository_root=repository_root,
        allow_missing_registry=allow_missing_registry,
    )
    return BlindPoolDeploymentOwnerGuard._acquire_path(
        owner_path,
        timeout_seconds=timeout_seconds,
    )
