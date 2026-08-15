"""Immutable pool-selection inputs for the persistent Blind Ladder bag."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import NoReturn

from .canonical_models import CanonicalRuntimeRegistry
from .errors import BlindPoolValidationError
from .fingerprint import compute_registry_fingerprint
from .models import BlindPoolRegistry, is_valid_opaque_team_id


_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SELECTION_CONSTRUCTION_TOKEN = object()
_RAW_SELECTION_KIND = "raw"
_CANONICAL_SELECTION_KIND = "canonical"


def _fail(code: str, message: str) -> NoReturn:
    raise BlindPoolValidationError(code, message) from None


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _is_within(path: Path, directory: Path) -> bool:
    path_key = _path_key(path)
    directory_key = _path_key(directory)
    try:
        return os.path.commonpath((path_key, directory_key)) == directory_key
    except ValueError:
        return False


def _resolved_path(path: Path, *, code: str) -> Path:
    failed = False
    try:
        resolved = Path(path).resolve(strict=True)
    except (OSError, RuntimeError, TypeError):
        failed = True
        resolved = None
    if failed:
        _fail(code, "Blind Ladder protected path is unavailable")
    assert resolved is not None
    return resolved


class BlindPoolSelectionSnapshot:
    """Minimal immutable bag identity with private collision boundaries."""

    __slots__ = (
        "_active_ids",
        "_kind",
        "_protected_files",
        "_protected_roots",
        "_registry_fingerprint",
        "_registry_path",
    )

    def __init__(
        self,
        *,
        active_ids: tuple[str, ...],
        registry_fingerprint: str,
        kind: str,
        registry_path: Path | None,
        protected_files: tuple[Path, ...],
        protected_roots: tuple[Path, ...],
        construction_token: object,
    ) -> None:
        if construction_token is not _SELECTION_CONSTRUCTION_TOKEN:
            _fail(
                "selection_construction_invalid",
                "Blind Ladder selection snapshot construction is restricted",
            )
        active_ids = tuple(active_ids)
        if (
            not active_ids
            or len(set(active_ids)) != len(active_ids)
            or not all(is_valid_opaque_team_id(team_id) for team_id in active_ids)
        ):
            _fail(
                "selection_active_ids_invalid",
                "Blind Ladder selection contains invalid active IDs",
            )
        if (
            not isinstance(registry_fingerprint, str)
            or _FINGERPRINT_PATTERN.fullmatch(registry_fingerprint) is None
        ):
            _fail(
                "selection_fingerprint_invalid",
                "Blind Ladder selection fingerprint is invalid",
            )
        if kind not in {_RAW_SELECTION_KIND, _CANONICAL_SELECTION_KIND}:
            _fail(
                "selection_kind_invalid",
                "Blind Ladder selection kind is invalid",
            )
        object.__setattr__(self, "_active_ids", active_ids)
        object.__setattr__(self, "_registry_fingerprint", registry_fingerprint)
        object.__setattr__(self, "_kind", kind)
        object.__setattr__(self, "_registry_path", registry_path)
        object.__setattr__(self, "_protected_files", tuple(protected_files))
        object.__setattr__(self, "_protected_roots", tuple(protected_roots))

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("BlindPoolSelectionSnapshot is immutable")

    def __delattr__(self, _name: str) -> None:
        raise AttributeError("BlindPoolSelectionSnapshot is immutable")

    def __reduce__(self) -> NoReturn:
        raise TypeError("BlindPoolSelectionSnapshot serialization is disabled")

    @property
    def active_ids(self) -> tuple[str, ...]:
        return self._active_ids

    @property
    def registry_fingerprint(self) -> str:
        return self._registry_fingerprint

    def __repr__(self) -> str:
        return "BlindPoolSelectionSnapshot(active_count={!r}, kind={!r})".format(
            len(self.active_ids),
            self._kind,
        )

    def __str__(self) -> str:
        return repr(self)

    def _validate_collision_paths(
        self,
        *,
        configured_registry_path: Path,
        state_path: Path,
        lock_path: Path,
    ) -> None:
        if (
            self._registry_path is not None
            and configured_registry_path != self._registry_path
        ):
            _fail(
                "selection_registry_path_mismatch",
                "Blind Ladder selection and configured registry do not match",
            )

        protected_files = frozenset(self._protected_files)
        if self._kind == _RAW_SELECTION_KIND:
            if configured_registry_path in protected_files:
                _fail(
                    "registry_team_file_collision",
                    "Blind Ladder registry must not replace a registered team file",
                )
            if state_path in protected_files:
                _fail(
                    "state_team_file_collision",
                    "Blind Ladder state must not replace a registered team file",
                )
            if lock_path in protected_files:
                _fail(
                    "state_lock_team_file_collision",
                    "Blind Ladder state lock must not use a registered team file",
                )
            return

        if any(_is_within(state_path, root) for root in self._protected_roots):
            _fail(
                "state_canonical_artifact_collision",
                "Blind Ladder state must remain outside canonical artifacts",
            )
        if any(_is_within(lock_path, root) for root in self._protected_roots):
            _fail(
                "state_lock_canonical_artifact_collision",
                "Blind Ladder state lock must remain outside canonical artifacts",
            )


def create_raw_selection_snapshot(
    registry: BlindPoolRegistry,
    *,
    registry_path: Path | None = None,
) -> BlindPoolSelectionSnapshot:
    """Adapt a raw registry while reusing its historical fingerprint exactly."""

    if not isinstance(registry, BlindPoolRegistry):
        _fail("registry_type_invalid", "Blind Ladder registry has an invalid type")
    resolved_registry = None
    if registry_path is not None:
        resolved_registry = _resolved_path(
            registry_path,
            code="selection_registry_path_invalid",
        )
    return BlindPoolSelectionSnapshot(
        active_ids=tuple(entry.team_id for entry in registry.active_entries),
        registry_fingerprint=compute_registry_fingerprint(registry),
        kind=_RAW_SELECTION_KIND,
        registry_path=resolved_registry,
        protected_files=tuple(entry.resolved_team_path for entry in registry.entries),
        protected_roots=(),
        construction_token=_SELECTION_CONSTRUCTION_TOKEN,
    )


def create_canonical_selection_snapshot(
    registry: CanonicalRuntimeRegistry,
) -> BlindPoolSelectionSnapshot:
    """Adapt a verified canonical registry without recomputing its fingerprint."""

    if not isinstance(registry, CanonicalRuntimeRegistry):
        _fail(
            "canonical_registry_type_invalid",
            "Canonical Blind Ladder registry has an invalid type",
        )
    canonical_root = _resolved_path(
        registry._private_root / "canonical",
        code="selection_canonical_root_invalid",
    )
    return BlindPoolSelectionSnapshot(
        active_ids=registry.active_ids,
        registry_fingerprint=registry.registry_fingerprint,
        kind=_CANONICAL_SELECTION_KIND,
        registry_path=registry._registry_path,
        protected_files=(),
        protected_roots=(canonical_root,),
        construction_token=_SELECTION_CONSTRUCTION_TOKEN,
    )


def coerce_selection_snapshot(
    value: BlindPoolSelectionSnapshot | BlindPoolRegistry,
) -> BlindPoolSelectionSnapshot:
    """Preserve direct raw-registry callers while state code consumes snapshots."""

    if isinstance(value, BlindPoolSelectionSnapshot):
        return value
    if isinstance(value, BlindPoolRegistry):
        return create_raw_selection_snapshot(value)
    _fail("registry_type_invalid", "Blind Ladder registry has an invalid type")
