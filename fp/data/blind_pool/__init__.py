"""External private-registry and persistent shuffled-bag contracts.

This package validates external state and exposes a dormant, dependency-injected
challenge lifecycle. It deliberately does not provision packed teams, activate
the sealed pool, or connect lifecycle behavior to the production entry point.

Schema v1 is exactly ``schema_version``, ``registry_version``, ``format_id``,
and ``entries``. Each entry is exactly ``team_id``, ``active``, ``team_file``,
and ``sha256``; the digest covers the referenced file's raw bytes.
"""

from .bag import BlindPoolBagStore
from .config import (
    PRIVATE_ROOT_ENV,
    REGISTRY_PATH_ENV,
    STATE_PATH_ENV,
    load_blind_pool_config,
    load_blind_pool_state_config,
    validate_blind_pool_config,
    validate_blind_pool_state_config,
)
from .errors import (
    BlindPoolLifecycleError,
    BlindPoolReconciliationRequired,
    BlindPoolValidationError,
)
from .fingerprint import compute_registry_fingerprint
from .lifecycle import (
    BLIND_LADDER_FORMAT,
    BLIND_LADDER_MODE,
    BlindPoolLifecycleCoordinator,
    BlindRoomCorrelator,
    normalize_showdown_identity,
    parse_incoming_challenge,
)
from .loader import (
    get_active_blind_pool_entries,
    get_blind_pool_entry_by_id,
    load_blind_pool_registry,
    validate_blind_pool_registry,
)
from .models import (
    SCHEMA_VERSION,
    SUPPORTED_FORMAT_ID,
    BlindPoolBagState,
    BlindPoolBattleRoom,
    BlindPoolChallenge,
    BlindPoolConfig,
    BlindPoolEntry,
    BlindPoolRegistry,
    BlindPoolReservation,
    BlindPoolStateConfig,
)
from .state import STATE_SCHEMA_VERSION, validate_blind_pool_bag_state

__all__ = (
    "PRIVATE_ROOT_ENV",
    "REGISTRY_PATH_ENV",
    "STATE_PATH_ENV",
    "BLIND_LADDER_FORMAT",
    "BLIND_LADDER_MODE",
    "SCHEMA_VERSION",
    "STATE_SCHEMA_VERSION",
    "SUPPORTED_FORMAT_ID",
    "BlindPoolBagState",
    "BlindPoolBagStore",
    "BlindPoolBattleRoom",
    "BlindPoolChallenge",
    "BlindPoolConfig",
    "BlindPoolEntry",
    "BlindPoolLifecycleError",
    "BlindPoolLifecycleCoordinator",
    "BlindPoolReconciliationRequired",
    "BlindPoolRegistry",
    "BlindPoolReservation",
    "BlindPoolStateConfig",
    "BlindPoolValidationError",
    "BlindRoomCorrelator",
    "compute_registry_fingerprint",
    "get_active_blind_pool_entries",
    "get_blind_pool_entry_by_id",
    "load_blind_pool_config",
    "load_blind_pool_state_config",
    "load_blind_pool_registry",
    "normalize_showdown_identity",
    "parse_incoming_challenge",
    "validate_blind_pool_config",
    "validate_blind_pool_bag_state",
    "validate_blind_pool_registry",
    "validate_blind_pool_state_config",
)
