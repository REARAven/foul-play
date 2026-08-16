"""Deterministic metadata-only canonical registry generation."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, NoReturn

from .canonical_models import (
    CANONICAL_ARTIFACT_SCHEMA_VERSION,
    CANONICAL_FORMAT_ID,
    CANONICAL_METADATA_SCHEMA_VERSION,
    CANONICAL_REGISTRY_SCHEMA_VERSION,
    CanonicalArtifactError,
    _require_exact_fields,
    _require_opaque_team_id,
    _strict_json_bytes,
)
from .canonical_registry import (
    _is_link_or_reparse,
    _is_within,
    _resolve_existing_directory,
    _resolve_private_root,
    _same_file_object,
    _stable_read_bytes,
    _stat_identity,
    is_valid_canonical_registry_version,
    load_canonical_runtime_registry,
    _load_validated_canonical_metadata_bytes,
)
from .models import BlindPoolConfig, BlindPoolStateConfig
from .errors import BlindPoolValidationError
from .ownership import (
    _acquire_validated_blind_pool_deployment_owner,
    _owner_path_candidate,
)
from .startup import (
    BlindCanonicalActivationError,
    BlindCanonicalErrorCategory,
)
from .state import _fsync_directory


CANONICAL_DEPLOYMENT_PLAN_SCHEMA_VERSION = 1

_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "registry_version",
        "format_id",
        "entries",
    }
)
_PLAN_ENTRY_FIELDS = frozenset({"team_id", "active"})


def _error(
    category: BlindCanonicalErrorCategory,
    code: str,
    message: str,
) -> BlindCanonicalActivationError:
    return BlindCanonicalActivationError(category, code, message)


def _raise(
    category: BlindCanonicalErrorCategory,
    code: str,
    message: str,
) -> NoReturn:
    raise _error(category, code, message) from None


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _same_path(left: Path, right: Path) -> bool:
    return _path_key(left) == _path_key(right)


@dataclass(frozen=True, slots=True, repr=False)
class CanonicalDeploymentPlanEntry:
    """One explicit opaque membership decision."""

    team_id: str = field(repr=False)
    active: bool

    def __repr__(self) -> str:
        return "CanonicalDeploymentPlanEntry(active={!r})".format(self.active)


@dataclass(frozen=True, slots=True, repr=False)
class CanonicalDeploymentPlan:
    """Strict membership authority without team content or filesystem paths."""

    schema_version: int
    registry_version: str
    format_id: str
    entries: tuple[CanonicalDeploymentPlanEntry, ...] = field(repr=False)

    @property
    def active_count(self) -> int:
        return sum(entry.active for entry in self.entries)

    def __repr__(self) -> str:
        return (
            "CanonicalDeploymentPlan(schema_version={!r}, "
            "registry_version={!r}, format_id={!r}, entry_count={!r}, "
            "active_count={!r})"
        ).format(
            self.schema_version,
            self.registry_version,
            self.format_id,
            len(self.entries),
            self.active_count,
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError("CanonicalDeploymentPlan serialization is disabled")


@dataclass(frozen=True, slots=True, repr=False)
class CanonicalRegistryBuildConfig:
    """Explicit offline build paths with a fully redacted representation."""

    private_root: Path = field(repr=False)
    plan_path: Path = field(repr=False)
    output_registry_path: Path = field(repr=False)
    state_path: Path = field(repr=False)

    def __post_init__(self) -> None:
        try:
            paths = tuple(
                Path(value)
                for value in (
                    self.private_root,
                    self.plan_path,
                    self.output_registry_path,
                    self.state_path,
                )
            )
        except (TypeError, ValueError):
            paths = ()
        if len(paths) != 4 or not all(path.is_absolute() for path in paths):
            _raise(
                BlindCanonicalErrorCategory.CONFIGURATION,
                "blind_canonical_registry_build_config_invalid",
                "Canonical registry build paths must be explicit absolute paths",
            )
        object.__setattr__(self, "private_root", paths[0])
        object.__setattr__(self, "plan_path", paths[1])
        object.__setattr__(self, "output_registry_path", paths[2])
        object.__setattr__(self, "state_path", paths[3])

    def __repr__(self) -> str:
        return "CanonicalRegistryBuildConfig(configured=True)"

    def __reduce__(self) -> NoReturn:
        raise TypeError("CanonicalRegistryBuildConfig serialization is disabled")


@dataclass(frozen=True, slots=True, repr=False)
class CanonicalRegistryGenerationResult:
    """Safe aggregate result of one completed registry publication."""

    registry_version: str
    entry_count: int
    active_count: int

    def __repr__(self) -> str:
        return (
            "CanonicalRegistryGenerationResult(registry_version={!r}, "
            "entry_count={!r}, active_count={!r})"
        ).format(self.registry_version, self.entry_count, self.active_count)


@dataclass(frozen=True, slots=True, repr=False)
class _ResolvedBuildPaths:
    private_root: Path = field(repr=False)
    plan_path: Path = field(repr=False)
    output_registry_path: Path = field(repr=False)
    state_path: Path = field(repr=False)
    repository_root: Path = field(repr=False)


def parse_canonical_deployment_plan_bytes(raw: bytes) -> CanonicalDeploymentPlan:
    """Parse strict UTF-8 membership JSON without retaining the input bytes."""

    parse_failure: str | None = None
    try:
        document = _strict_json_bytes(
            raw,
            encoding_code="PLAN_ENCODING_INVALID",
            duplicate_code="PLAN_DUPLICATE_KEY",
            json_code="PLAN_JSON_INVALID",
        )
        value = _require_exact_fields(
            document,
            _PLAN_FIELDS,
            code="PLAN_INVALID",
            context="Canonical deployment plan",
        )
    except CanonicalArtifactError as error:
        parse_failure = (
            "blind_canonical_plan_duplicate_key"
            if error.code == "PLAN_DUPLICATE_KEY"
            else "blind_canonical_plan_invalid"
        )
        value = None
    if parse_failure is not None:
        _raise(
            BlindCanonicalErrorCategory.CONFIGURATION,
            parse_failure,
            "Canonical deployment plan is invalid",
        )
    assert value is not None

    schema_version = value["schema_version"]
    if (
        type(schema_version) is not int
        or schema_version != CANONICAL_DEPLOYMENT_PLAN_SCHEMA_VERSION
    ):
        _raise(
            BlindCanonicalErrorCategory.CONFIGURATION,
            "blind_canonical_plan_schema_unsupported",
            "Canonical deployment plan schema is unsupported",
        )
    registry_version = value["registry_version"]
    if not is_valid_canonical_registry_version(registry_version):
        _raise(
            BlindCanonicalErrorCategory.CONFIGURATION,
            "blind_canonical_plan_registry_version_invalid",
            "Canonical deployment registry version is invalid",
        )
    if value["format_id"] != CANONICAL_FORMAT_ID:
        _raise(
            BlindCanonicalErrorCategory.CONFIGURATION,
            "blind_canonical_plan_format_invalid",
            "Canonical deployment plan format is incompatible",
        )
    raw_entries = value["entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        _raise(
            BlindCanonicalErrorCategory.CONFIGURATION,
            "blind_canonical_plan_invalid",
            "Canonical deployment plan entries are invalid",
        )

    entries: list[CanonicalDeploymentPlanEntry] = []
    seen_ids: set[str] = set()
    for raw_entry in raw_entries:
        entry_invalid = False
        try:
            entry = _require_exact_fields(
                raw_entry,
                _PLAN_ENTRY_FIELDS,
                code="PLAN_ENTRY_INVALID",
                context="Canonical deployment plan entry",
            )
            team_id = _require_opaque_team_id(
                entry["team_id"],
                code="PLAN_ENTRY_INVALID",
            )
        except CanonicalArtifactError:
            entry_invalid = True
            entry = None
            team_id = None
        if entry_invalid:
            _raise(
                BlindCanonicalErrorCategory.CONFIGURATION,
                "blind_canonical_plan_entry_invalid",
                "Canonical deployment plan entry is invalid",
            )
        assert entry is not None and team_id is not None
        if team_id in seen_ids:
            _raise(
                BlindCanonicalErrorCategory.CONFIGURATION,
                "blind_canonical_plan_duplicate_team",
                "Canonical deployment plan contains a duplicate team ID",
            )
        active = entry["active"]
        if type(active) is not bool:
            _raise(
                BlindCanonicalErrorCategory.CONFIGURATION,
                "blind_canonical_plan_entry_invalid",
                "Canonical deployment plan entry is invalid",
            )
        seen_ids.add(team_id)
        entries.append(CanonicalDeploymentPlanEntry(team_id, active))
    if not any(entry.active for entry in entries):
        _raise(
            BlindCanonicalErrorCategory.CONFIGURATION,
            "blind_canonical_plan_no_active_entries",
            "Canonical deployment plan requires an active entry",
        )
    return CanonicalDeploymentPlan(
        schema_version,
        registry_version,
        CANONICAL_FORMAT_ID,
        tuple(sorted(entries, key=lambda item: item.team_id)),
    )


def _resolve_existing_file(path: Path, *, code: str) -> Path:
    invalid = False
    try:
        before = os.lstat(path)
        if (
            not stat.S_ISREG(before.st_mode)
            or _is_link_or_reparse(before)
            or int(before.st_nlink) != 1
        ):
            invalid = True
            resolved = None
        else:
            resolved = path.resolve(strict=True)
            after = os.lstat(path)
            if _stat_identity(before) != _stat_identity(after):
                invalid = True
    except (OSError, RuntimeError, ValueError, TypeError):
        invalid = True
        resolved = None
    if invalid or resolved is None:
        _raise(
            BlindCanonicalErrorCategory.CONFIGURATION,
            code,
            "Canonical registry build path is invalid",
        )
    return resolved


def _resolve_state_path_for_build(
    path: Path,
    *,
    private_root: Path,
    canonical_root: Path,
    repository_root: Path,
) -> Path:
    invalid = False
    try:
        parent = _resolve_existing_directory(
            path.parent,
            code="CANONICAL_STATE_PATH_INVALID",
        )
        if path.exists() or path.is_symlink():
            info = os.lstat(path)
            if (
                not stat.S_ISREG(info.st_mode)
                or _is_link_or_reparse(info)
                or int(info.st_nlink) != 1
            ):
                invalid = True
                resolved = None
            else:
                resolved = path.resolve(strict=True)
        else:
            resolved = parent / path.name
    except (CanonicalArtifactError, OSError, RuntimeError, ValueError, TypeError):
        invalid = True
        resolved = None
    if (
        invalid
        or resolved is None
        or not _is_within(resolved, private_root, allow_equal=False)
        or _is_within(resolved, canonical_root)
        or _is_within(resolved, repository_root)
    ):
        _raise(
            BlindCanonicalErrorCategory.CONFIGURATION,
            "blind_canonical_registry_build_state_invalid",
            "Canonical registry build state path is invalid",
        )
    return resolved


def _resolve_absent_output_path(
    path: Path,
    *,
    private_root: Path,
    canonical_root: Path,
    repository_root: Path,
    plan_path: Path,
    state_path: Path,
) -> Path:
    if path.exists() or path.is_symlink():
        _raise(
            BlindCanonicalErrorCategory.CONFIGURATION,
            "blind_canonical_registry_output_exists",
            "Canonical registry output already exists",
        )
    invalid = False
    try:
        parent = _resolve_existing_directory(
            path.parent,
            code="CANONICAL_REGISTRY_OUTPUT_INVALID",
        )
        resolved = parent / path.name
        state_config = BlindPoolStateConfig(
            BlindPoolConfig(private_root, resolved),
            state_path,
        )
        transaction_lock = state_config.lock_path
        owner_lock = _owner_path_candidate(state_config.state_path)
    except (CanonicalArtifactError, OSError, RuntimeError, ValueError, TypeError):
        invalid = True
        resolved = transaction_lock = owner_lock = None
    if (
        invalid
        or resolved is None
        or not _is_within(resolved, private_root, allow_equal=False)
        or _is_within(resolved, canonical_root)
        or _is_within(resolved, repository_root)
        or any(
            _same_path(resolved, other)
            for other in (plan_path, state_path, transaction_lock, owner_lock)
            if other is not None
        )
    ):
        _raise(
            BlindCanonicalErrorCategory.CONFIGURATION,
            "blind_canonical_registry_output_invalid",
            "Canonical registry output path is invalid",
        )
    return resolved


def _resolve_build_config(
    config: CanonicalRegistryBuildConfig,
    *,
    repository_root: str | Path | None,
) -> _ResolvedBuildPaths:
    if not isinstance(config, CanonicalRegistryBuildConfig):
        _raise(
            BlindCanonicalErrorCategory.CONFIGURATION,
            "blind_canonical_registry_build_config_invalid",
            "Canonical registry build configuration is invalid",
        )
    config_invalid = False
    try:
        repository = Path(repository_root or _repository_root()).resolve(strict=True)
        private_root = _resolve_private_root(
            config.private_root,
            repository_root=repository,
        )
        canonical_root = _resolve_existing_directory(
            private_root / "canonical",
            code="CANONICAL_ARTIFACT_PATH_INVALID",
            parent=private_root,
        )
    except (CanonicalArtifactError, OSError, RuntimeError, ValueError, TypeError):
        config_invalid = True
        repository = private_root = canonical_root = None
    if config_invalid:
        _raise(
            BlindCanonicalErrorCategory.CONFIGURATION,
            "blind_canonical_registry_build_config_invalid",
            "Canonical registry build configuration is invalid",
        )
    assert (
        repository is not None
        and private_root is not None
        and canonical_root is not None
    )
    plan_path = _resolve_existing_file(
        config.plan_path,
        code="blind_canonical_plan_path_invalid",
    )
    state_path = _resolve_state_path_for_build(
        config.state_path,
        private_root=private_root,
        canonical_root=canonical_root,
        repository_root=repository,
    )
    if _same_path(plan_path, state_path):
        _raise(
            BlindCanonicalErrorCategory.CONFIGURATION,
            "blind_canonical_registry_build_path_collision",
            "Canonical registry build paths conflict",
        )
    output_path = _resolve_absent_output_path(
        config.output_registry_path,
        private_root=private_root,
        canonical_root=canonical_root,
        repository_root=repository,
        plan_path=plan_path,
        state_path=state_path,
    )
    return _ResolvedBuildPaths(
        private_root,
        plan_path,
        output_path,
        state_path,
        repository,
    )


def load_canonical_deployment_plan(
    plan_path: str | Path,
) -> CanonicalDeploymentPlan:
    """Load one exact stable plan file without exposing its path or bytes."""

    try:
        path = Path(plan_path)
    except (TypeError, ValueError):
        path = None
    if path is None or not path.is_absolute():
        _raise(
            BlindCanonicalErrorCategory.CONFIGURATION,
            "blind_canonical_plan_path_invalid",
            "Canonical deployment plan path is invalid",
        )
    resolved = _resolve_existing_file(
        path,
        code="blind_canonical_plan_path_invalid",
    )
    read_failed = False
    try:
        raw = _stable_read_bytes(
            resolved,
            code="CANONICAL_PLAN_INVALID",
        )
    except CanonicalArtifactError:
        read_failed = True
        raw = None
    if read_failed:
        _raise(
            BlindCanonicalErrorCategory.CONFIGURATION,
            "blind_canonical_plan_invalid",
            "Canonical deployment plan could not be read",
        )
    assert raw is not None
    return parse_canonical_deployment_plan_bytes(raw)


def _registry_document(
    plan: CanonicalDeploymentPlan,
    metadata_digests: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": CANONICAL_REGISTRY_SCHEMA_VERSION,
        "registry_version": plan.registry_version,
        "format_id": CANONICAL_FORMAT_ID,
        "artifact_schema_version": CANONICAL_ARTIFACT_SCHEMA_VERSION,
        "metadata_schema_version": CANONICAL_METADATA_SCHEMA_VERSION,
        "entries": [
            {
                "team_id": entry.team_id,
                "active": entry.active,
                "metadata_sha256": metadata_digests[entry.team_id],
            }
            for entry in plan.entries
        ],
    }


def _serialize_registry(document: dict[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                document,
                ensure_ascii=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        _raise(
            BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
            "blind_canonical_registry_serialization_failed",
            "Canonical registry could not be serialized",
        )


def _prepare_registry_bytes(
    paths: _ResolvedBuildPaths,
    plan: CanonicalDeploymentPlan,
) -> bytes:
    metadata_digests: dict[str, str] = {}
    active_provenance: tuple[object, ...] | None = None
    for entry in plan.entries:
        metadata_failed = False
        try:
            raw, metadata = _load_validated_canonical_metadata_bytes(
                paths.private_root,
                entry.team_id,
                repository_root=paths.repository_root,
            )
        except CanonicalArtifactError:
            metadata_failed = True
            raw = metadata = None
        if metadata_failed:
            _raise(
                BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
                "blind_canonical_registry_metadata_invalid",
                "Canonical registry metadata validation failed",
            )
        assert raw is not None and metadata is not None
        metadata_digests[entry.team_id] = hashlib.sha256(raw).hexdigest()
        if entry.active:
            if active_provenance is None:
                active_provenance = metadata.active_provenance
            elif metadata.active_provenance != active_provenance:
                _raise(
                    BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
                    "blind_canonical_registry_provenance_mismatch",
                    "Active canonical metadata provenance is inconsistent",
                )
        del metadata
        del raw
    return _serialize_registry(_registry_document(plan, metadata_digests))


def _atomic_boundary(_name: str) -> None:
    """No-op failure-injection seam for atomic publication tests."""


def _is_safe_published_file(path: Path, identity: tuple[int, ...]) -> bool:
    try:
        info = os.lstat(path)
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError, TypeError):
        return False
    return (
        stat.S_ISREG(info.st_mode)
        and not _is_link_or_reparse(info)
        and int(info.st_nlink) == 1
        and _stat_identity(info) == identity
        and _same_path(resolved, path)
    )


def _unlink_owned_temporary(
    path: Path,
    *,
    owned_object: os.stat_result,
    expected_parent: Path,
    parent_object: os.stat_result,
) -> None:
    """Remove a temp name only while it still identifies this operation's file."""

    try:
        path_info = os.lstat(path)
        parent_info = os.lstat(expected_parent)
        resolved = path.resolve(strict=True)
        if (
            not stat.S_ISREG(path_info.st_mode)
            or _is_link_or_reparse(path_info)
            or not _same_file_object(path_info, owned_object)
            or not _same_file_object(parent_info, parent_object)
            or not _same_path(resolved, path)
            or not _is_within(resolved, expected_parent, allow_equal=False)
        ):
            return
        path.unlink()
    except (OSError, RuntimeError, ValueError, TypeError):
        return


def _publish_registry_atomic(
    output_path: Path,
    payload: bytes,
) -> tuple[int, ...]:
    descriptor: int | None = None
    temporary_path: Path | None = None
    temporary_object: os.stat_result | None = None
    parent_object: os.stat_result | None = None
    published = False
    published_identity: tuple[int, ...] | None = None
    failure_code: str | None = None
    try:
        parent_before = os.lstat(output_path.parent)
        parent_object = parent_before
        resolved_parent = _resolve_existing_directory(
            output_path.parent,
            code="CANONICAL_REGISTRY_OUTPUT_INVALID",
        )
        if not _same_path(resolved_parent, output_path.parent):
            raise OSError
        _atomic_boundary("temp_create")
        descriptor, raw_path = tempfile.mkstemp(
            prefix=".{}-".format(output_path.name),
            suffix=".tmp",
            dir=output_path.parent,
        )
        temporary_path = Path(raw_path)
        created_path_info = os.lstat(temporary_path)
        created_descriptor_info = os.fstat(descriptor)
        temporary_object = created_descriptor_info
        if (
            not stat.S_ISREG(created_path_info.st_mode)
            or not stat.S_ISREG(created_descriptor_info.st_mode)
            or _is_link_or_reparse(created_path_info)
            or _is_link_or_reparse(created_descriptor_info)
            or int(created_path_info.st_nlink) != 1
            or int(created_descriptor_info.st_nlink) != 1
            or not _same_file_object(created_path_info, created_descriptor_info)
        ):
            raise OSError
        with os.fdopen(descriptor, "wb") as destination:
            descriptor = None
            path_before = os.lstat(temporary_path)
            descriptor_before = os.fstat(destination.fileno())
            if (
                not stat.S_ISREG(path_before.st_mode)
                or not stat.S_ISREG(descriptor_before.st_mode)
                or _is_link_or_reparse(path_before)
                or _is_link_or_reparse(descriptor_before)
                or int(path_before.st_nlink) != 1
                or int(descriptor_before.st_nlink) != 1
                or not _same_file_object(path_before, descriptor_before)
            ):
                raise OSError
            _atomic_boundary("temp_write")
            written = destination.write(payload)
            if written != len(payload):
                raise OSError
            _atomic_boundary("flush")
            destination.flush()
            _atomic_boundary("fsync")
            os.fsync(destination.fileno())
            descriptor_after = os.fstat(destination.fileno())
            path_after = os.lstat(temporary_path)
            if (
                not _same_file_object(path_before, path_after)
                or not _same_file_object(descriptor_before, descriptor_after)
                or not _same_file_object(descriptor_after, path_after)
                or descriptor_after.st_size != len(payload)
                or path_after.st_size != len(payload)
                or _is_link_or_reparse(descriptor_after)
                or _is_link_or_reparse(path_after)
                or int(descriptor_after.st_nlink) != 1
                or int(path_after.st_nlink) != 1
            ):
                raise OSError

        temp_path_info = os.lstat(temporary_path)
        if (
            not stat.S_ISREG(temp_path_info.st_mode)
            or _is_link_or_reparse(temp_path_info)
            or int(temp_path_info.st_nlink) != 1
            or temp_path_info.st_size != len(payload)
            or not _is_within(
                temporary_path.resolve(strict=True),
                output_path.parent,
                allow_equal=False,
            )
        ):
            raise OSError
        if output_path.exists() or output_path.is_symlink():
            raise FileExistsError
        parent_before_publish = os.lstat(output_path.parent)
        if not _same_file_object(
            parent_before, parent_before_publish
        ) or not _same_path(
            _resolve_existing_directory(
                output_path.parent,
                code="CANONICAL_REGISTRY_OUTPUT_INVALID",
            ),
            output_path.parent,
        ):
            raise OSError
        _atomic_boundary("publish")
        os.link(temporary_path, output_path)
        published = True
        published_info = os.lstat(output_path)
        temporary_info = os.lstat(temporary_path)
        if not _same_file_object(published_info, temporary_info):
            raise OSError
        published_identity = _stat_identity(published_info)
        temporary_path.unlink()
        temporary_path = None
        published_after_unlink = os.lstat(output_path)
        if (
            not _same_file_object(published_info, published_after_unlink)
            or not stat.S_ISREG(published_after_unlink.st_mode)
            or _is_link_or_reparse(published_after_unlink)
            or int(published_after_unlink.st_nlink) != 1
            or published_after_unlink.st_size != len(payload)
        ):
            raise OSError
        published_identity = _stat_identity(published_after_unlink)
        _atomic_boundary("parent_fsync")
        _fsync_directory(output_path.parent)

        published_info = os.lstat(output_path)
        parent_after_publish = os.lstat(output_path.parent)
        if (
            not stat.S_ISREG(published_info.st_mode)
            or _is_link_or_reparse(published_info)
            or int(published_info.st_nlink) != 1
            or published_info.st_size != len(payload)
            or not _same_file_object(parent_before, parent_after_publish)
            or not _same_path(output_path.resolve(strict=True), output_path)
        ):
            raise OSError
        published_identity = _stat_identity(published_info)
        _atomic_boundary("readback")
        readback = _stable_read_bytes(
            output_path,
            code="CANONICAL_REGISTRY_INVALID",
        )
        if readback != payload:
            raise OSError
    except FileExistsError:
        if not published:
            failure_code = "blind_canonical_registry_output_exists"
        else:
            failure_code = "blind_canonical_registry_publication_failed"
    except (CanonicalArtifactError, OSError):
        failure_code = "blind_canonical_registry_publication_failed"
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if (
            temporary_path is not None
            and temporary_object is not None
            and parent_object is not None
        ):
            _unlink_owned_temporary(
                temporary_path,
                owned_object=temporary_object,
                expected_parent=output_path.parent,
                parent_object=parent_object,
            )
    if failure_code is not None:
        if (
            published
            and published_identity is not None
            and _is_safe_published_file(output_path, published_identity)
        ):
            try:
                output_path.unlink()
            except OSError:
                pass
        if failure_code == "blind_canonical_registry_output_exists":
            _raise(
                BlindCanonicalErrorCategory.CONFIGURATION,
                failure_code,
                "Canonical registry output already exists",
            )
        _raise(
            BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
            failure_code,
            "Canonical registry publication failed",
        )
    assert published_identity is not None
    return published_identity


def _state_path_exists_safely(path: Path) -> bool:
    invalid = False
    try:
        if not path.exists() and not path.is_symlink():
            return False
        info = os.lstat(path)
    except OSError:
        invalid = True
        info = None
    if invalid:
        _raise(
            BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
            "blind_canonical_registry_state_invalid",
            "Canonical registry build state path is invalid",
        )
    assert info is not None
    if (
        not stat.S_ISREG(info.st_mode)
        or _is_link_or_reparse(info)
        or int(info.st_nlink) != 1
    ):
        _raise(
            BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
            "blind_canonical_registry_state_invalid",
            "Canonical registry build state path is invalid",
        )
    return True


def build_canonical_runtime_registry(
    config: CanonicalRegistryBuildConfig,
    *,
    repository_root: str | Path | None = None,
    owner_timeout_seconds: float = 0.25,
) -> CanonicalRegistryGenerationResult:
    """Publish one absent registry from explicit metadata-only membership."""

    paths = _resolve_build_config(config, repository_root=repository_root)
    plan = load_canonical_deployment_plan(paths.plan_path)
    payload = _prepare_registry_bytes(paths, plan)
    state_config = BlindPoolStateConfig(
        BlindPoolConfig(paths.private_root, paths.output_registry_path),
        paths.state_path,
    )

    ownership_failed = False
    try:
        owner = _acquire_validated_blind_pool_deployment_owner(
            state_config,
            timeout_seconds=owner_timeout_seconds,
            repository_root=paths.repository_root,
            allow_missing_registry=True,
        )
    except BlindPoolValidationError:
        ownership_failed = True
        owner = None
    if ownership_failed:
        _raise(
            BlindCanonicalErrorCategory.DEPLOYMENT_OWNERSHIP,
            "blind_canonical_deployment_in_use",
            "Canonical deployment is already active or unavailable",
        )
    assert owner is not None

    operation_error: BlindCanonicalActivationError | None = None
    result: CanonicalRegistryGenerationResult | None = None
    try:
        _resolve_absent_output_path(
            paths.output_registry_path,
            private_root=paths.private_root,
            canonical_root=paths.private_root / "canonical",
            repository_root=paths.repository_root,
            plan_path=paths.plan_path,
            state_path=paths.state_path,
        )
        if _state_path_exists_safely(paths.state_path):
            _raise(
                BlindCanonicalErrorCategory.RECOVERY_REQUIRED,
                "blind_canonical_registry_existing_state",
                "Canonical registry generation does not update existing state",
            )
        published_identity = _publish_registry_atomic(
            paths.output_registry_path,
            payload,
        )
        self_validation_failed = False
        try:
            registry = load_canonical_runtime_registry(
                paths.private_root,
                paths.output_registry_path,
                repository_root=paths.repository_root,
            )
        except CanonicalArtifactError:
            self_validation_failed = True
            registry = None
        if self_validation_failed:
            if _is_safe_published_file(
                paths.output_registry_path,
                published_identity,
            ):
                try:
                    paths.output_registry_path.unlink()
                except OSError:
                    pass
            _raise(
                BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
                "blind_canonical_registry_self_validation_failed",
                "Generated canonical registry failed self-validation",
            )
        assert registry is not None
        expected_membership = tuple(
            (entry.team_id, entry.active) for entry in plan.entries
        )
        actual_membership = tuple(
            (entry.team_id, entry.active) for entry in registry.entries
        )
        if (
            registry.registry_version != plan.registry_version
            or registry.format_id != plan.format_id
            or actual_membership != expected_membership
        ):
            if _is_safe_published_file(
                paths.output_registry_path,
                published_identity,
            ):
                try:
                    paths.output_registry_path.unlink()
                except OSError:
                    pass
            _raise(
                BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
                "blind_canonical_registry_self_validation_failed",
                "Generated canonical registry failed self-validation",
            )
        result = CanonicalRegistryGenerationResult(
            registry.registry_version,
            len(registry.entries),
            len(registry.active_ids),
        )
    except BlindCanonicalActivationError as error:
        operation_error = error
    finally:
        try:
            owner.close()
        except BlindPoolValidationError:
            operation_error = _error(
                BlindCanonicalErrorCategory.DEPLOYMENT_OWNERSHIP,
                "blind_canonical_owner_release_failed",
                "Canonical deployment ownership could not be released",
            )
    if operation_error is not None:
        raise operation_error from None
    assert result is not None
    return result
