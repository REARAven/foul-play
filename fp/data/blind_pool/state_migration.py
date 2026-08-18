"""Explicit, network-free Blind Ladder selection-state schema migrations."""

from __future__ import annotations

from dataclasses import replace

from .config import validate_blind_pool_state_config
from .errors import BlindPoolValidationError
from .locking import BlindPoolStateLock
from .models import BlindPoolBagState, BlindPoolRegistry, BlindPoolStateConfig
from .selection import BlindPoolSelectionSnapshot
from .state import (
    LEGACY_STATE_SCHEMA_VERSION,
    STATE_SCHEMA_VERSION,
    load_blind_pool_bag_state,
    write_blind_pool_bag_state_atomic,
)


def migrate_blind_pool_state_schema_2_to_3(
    config: BlindPoolStateConfig,
    registry: BlindPoolSelectionSnapshot | BlindPoolRegistry,
    *,
    lock_timeout_seconds: float = 5.0,
) -> BlindPoolBagState:
    """Upgrade one idle state in place without changing any bag semantics."""

    validated_config = validate_blind_pool_state_config(config)
    with BlindPoolStateLock(
        validated_config.lock_path,
        timeout_seconds=lock_timeout_seconds,
    ):
        legacy = load_blind_pool_bag_state(
            validated_config,
            registry,
            required_schema_version=LEGACY_STATE_SCHEMA_VERSION,
        )
        if legacy.reservation is not None:
            raise BlindPoolValidationError(
                "state_schema_migration_reservation_unresolved",
                "Selection state with a reservation cannot be migrated",
            ) from None
        migrated = replace(legacy, schema_version=STATE_SCHEMA_VERSION)
        write_blind_pool_bag_state_atomic(
            validated_config,
            migrated,
            registry,
            required_schema_version=STATE_SCHEMA_VERSION,
        )
        persisted = load_blind_pool_bag_state(
            validated_config,
            registry,
            required_schema_version=STATE_SCHEMA_VERSION,
        )
        if persisted != migrated:
            raise BlindPoolValidationError(
                "state_schema_migration_verification_failed",
                "Selection state migration verification failed",
            ) from None
        return persisted
