"""Explicit offline migration for strict canonical pool expansions."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import hmac
import os
from pathlib import Path
import random
from typing import NoReturn

from .bag import BlindPoolBagStore, ShuffleSource
from .canonical_artifacts import load_canonical_team_artifact
from .canonical_models import CanonicalArtifactError, CanonicalRuntimeRegistry
from .canonical_registry import load_canonical_runtime_registry
from .config import validate_blind_pool_state_config
from .errors import BlindPoolValidationError
from .locking import BlindPoolStateLock
from .models import BlindPoolBagState, BlindPoolConfig, BlindPoolStateConfig
from .ownership import acquire_blind_pool_deployment_owner
from .selection import BlindPoolSelectionSnapshot, create_canonical_selection_snapshot
from .startup import BlindCanonicalStartupConfig
from .state import (
    ACCEPT_SENT_PHASE,
    STATE_SCHEMA_VERSION,
    load_blind_pool_bag_state,
    write_blind_pool_bag_state_atomic,
)


@dataclass(frozen=True, slots=True, repr=False)
class BlindPoolExpansionMigrationResult:
    """Content-neutral aggregate result of one completed expansion."""

    source_active_count: int
    target_active_count: int
    consumed_count_preserved: int
    remaining_count: int
    target_registry_version: str

    def __repr__(self) -> str:
        return (
            "BlindPoolExpansionMigrationResult(source_active_count={!r}, "
            "target_active_count={!r}, consumed_count_preserved={!r}, "
            "remaining_count={!r}, target_registry_version={!r})"
        ).format(
            self.source_active_count,
            self.target_active_count,
            self.consumed_count_preserved,
            self.remaining_count,
            self.target_registry_version,
        )


@dataclass(frozen=True, slots=True)
class _MigrationDeployment:
    registry: CanonicalRuntimeRegistry
    selection: BlindPoolSelectionSnapshot
    state_config: BlindPoolStateConfig


def _fail(code: str, message: str) -> NoReturn:
    raise BlindPoolValidationError(code, message) from None


def _path_key(path: Path) -> str:
    value = str(path)
    return value.casefold() if os.name == "nt" else value


def _state_config(
    config: BlindCanonicalStartupConfig,
    *,
    repository_root: str | Path | None,
) -> BlindPoolStateConfig:
    if not isinstance(config, BlindCanonicalStartupConfig):
        _fail(
            "expansion_config_invalid",
            "Blind Ladder expansion configuration is invalid",
        )
    try:
        return validate_blind_pool_state_config(
            BlindPoolStateConfig(
                BlindPoolConfig(config.private_root, config.canonical_registry_path),
                config.state_path,
            ),
            repository_root=repository_root,
        )
    except BlindPoolValidationError:
        _fail(
            "expansion_config_invalid",
            "Blind Ladder expansion configuration is invalid",
        )


def _load_deployment(
    config: BlindCanonicalStartupConfig,
    state_config: BlindPoolStateConfig,
    *,
    repository_root: str | Path | None,
) -> _MigrationDeployment:
    try:
        registry = load_canonical_runtime_registry(
            config.private_root,
            config.canonical_registry_path,
            repository_root=repository_root,
        )
        selection = create_canonical_selection_snapshot(registry)
        BlindPoolBagStore.from_selection_snapshot(state_config, selection)
    except (CanonicalArtifactError, BlindPoolValidationError):
        _fail(
            "expansion_deployment_invalid",
            "Blind Ladder expansion deployment validation failed",
        )
    return _MigrationDeployment(registry, selection, state_config)


def _validate_expansion_bindings(
    source: CanonicalRuntimeRegistry,
    target: CanonicalRuntimeRegistry,
) -> tuple[str, ...]:
    source_active = frozenset(source.active_ids)
    target_active = frozenset(target.active_ids)
    if not source_active < target_active:
        _fail(
            "expansion_membership_invalid",
            "Blind Ladder migration requires a strict active-pool expansion",
        )

    for source_entry in source.entries:
        target_entry = target.get_entry(source_entry.team_id)
        if (
            target_entry is None
            or target_entry.active is not source_entry.active
            or not hmac.compare_digest(
                target_entry.metadata_sha256,
                source_entry.metadata_sha256,
            )
        ):
            _fail(
                "expansion_existing_binding_changed",
                "Blind Ladder existing canonical bindings changed",
            )
        try:
            source_artifact = load_canonical_team_artifact(
                source,
                source_entry.team_id,
            )
            target_artifact = load_canonical_team_artifact(
                target,
                source_entry.team_id,
            )
        except CanonicalArtifactError:
            _fail(
                "expansion_existing_binding_changed",
                "Blind Ladder existing canonical bindings changed",
            )
        del source_artifact
        del target_artifact

    return tuple(
        team_id for team_id in target.active_ids if team_id not in source_active
    )


def _expanded_state(
    source_state: BlindPoolBagState,
    target: _MigrationDeployment,
    added_ids: tuple[str, ...],
    random_source: ShuffleSource,
) -> BlindPoolBagState:
    consumed_prefix = source_state.cycle_order[: source_state.next_index]
    old_unconsumed = source_state.cycle_order[source_state.next_index :]
    remaining = list(old_unconsumed + added_ids)
    random_source.shuffle(remaining)
    if (
        source_state.cycle_number > 1
        and source_state.next_index == 0
        and source_state.last_consumed_id is not None
        and remaining[0] == source_state.last_consumed_id
    ):
        replacement_position = random_source.randrange(1, len(remaining))
        remaining[0], remaining[replacement_position] = (
            remaining[replacement_position],
            remaining[0],
        )
    return BlindPoolBagState(
        schema_version=STATE_SCHEMA_VERSION,
        registry_fingerprint=target.selection.registry_fingerprint,
        cycle_number=source_state.cycle_number,
        cycle_order=consumed_prefix + tuple(remaining),
        next_index=source_state.next_index,
        last_consumed_id=source_state.last_consumed_id,
        reservation=None,
    )


def _migrate_locked(
    source: _MigrationDeployment,
    target: _MigrationDeployment,
    *,
    random_source: ShuffleSource,
) -> BlindPoolExpansionMigrationResult:
    try:
        source_state = load_blind_pool_bag_state(
            source.state_config,
            source.selection,
        )
    except BlindPoolValidationError:
        _fail(
            "expansion_source_state_invalid",
            "Blind Ladder source state is invalid",
        )
    if source_state.reservation is not None:
        if source_state.reservation.phase == ACCEPT_SENT_PHASE:
            _fail(
                "expansion_accept_sent_unresolved",
                "Blind Ladder source state requires reconciliation",
            )
        _fail(
            "expansion_reserved_unresolved",
            "Blind Ladder source state contains an unresolved reservation",
        )
    if (
        target.state_config.state_path.exists()
        or target.state_config.state_path.is_symlink()
    ):
        _fail(
            "expansion_target_state_exists",
            "Blind Ladder target state already exists",
        )

    added_ids = _validate_expansion_bindings(source.registry, target.registry)
    migrated = _expanded_state(source_state, target, added_ids, random_source)
    try:
        write_blind_pool_bag_state_atomic(
            target.state_config,
            migrated,
            target.selection,
            replace_existing=False,
        )
        validated = load_blind_pool_bag_state(
            target.state_config,
            target.selection,
        )
    except BlindPoolValidationError as error:
        if error.code == "state_target_exists":
            _fail(
                "expansion_target_state_exists",
                "Blind Ladder target state already exists",
            )
        _fail(
            "expansion_target_state_invalid",
            "Blind Ladder migrated state validation failed",
        )
    if validated != migrated:
        _fail(
            "expansion_target_state_invalid",
            "Blind Ladder migrated state validation failed",
        )
    return BlindPoolExpansionMigrationResult(
        source_active_count=len(source.selection.active_ids),
        target_active_count=len(target.selection.active_ids),
        consumed_count_preserved=source_state.next_index,
        remaining_count=len(target.selection.active_ids) - source_state.next_index,
        target_registry_version=target.registry.registry_version,
    )


def migrate_blind_canonical_pool_expansion(
    source_config: BlindCanonicalStartupConfig,
    target_config: BlindCanonicalStartupConfig,
    *,
    repository_root: str | Path | None = None,
    random_source: ShuffleSource | None = None,
    owner_timeout_seconds: float = 0.25,
    lock_timeout_seconds: float = 5.0,
) -> BlindPoolExpansionMigrationResult:
    """Atomically publish a new state for one strict canonical pool expansion."""

    source_state_config = _state_config(
        source_config,
        repository_root=repository_root,
    )
    target_state_config = _state_config(
        target_config,
        repository_root=repository_root,
    )
    if (
        source_state_config.pool_config.private_root
        == target_state_config.pool_config.private_root
        or source_state_config.state_path == target_state_config.state_path
        or source_state_config.lock_path == target_state_config.lock_path
    ):
        _fail(
            "expansion_config_invalid",
            "Blind Ladder source and target deployments must be distinct",
        )

    state_configs = sorted(
        (source_state_config, target_state_config),
        key=lambda item: _path_key(item.state_path),
    )
    shuffle = random.SystemRandom() if random_source is None else random_source
    if not callable(getattr(shuffle, "shuffle", None)) or not callable(
        getattr(shuffle, "randrange", None)
    ):
        _fail(
            "expansion_random_source_invalid",
            "Blind Ladder expansion randomness is invalid",
        )

    with ExitStack() as stack:
        for state_config in state_configs:
            stack.enter_context(
                acquire_blind_pool_deployment_owner(
                    state_config,
                    timeout_seconds=owner_timeout_seconds,
                    repository_root=repository_root,
                )
            )
        source = _load_deployment(
            source_config,
            source_state_config,
            repository_root=repository_root,
        )
        target = _load_deployment(
            target_config,
            target_state_config,
            repository_root=repository_root,
        )
        for state_config in state_configs:
            stack.enter_context(
                BlindPoolStateLock(
                    state_config.lock_path,
                    timeout_seconds=lock_timeout_seconds,
                )
            )
        return _migrate_locked(source, target, random_source=shuffle)
