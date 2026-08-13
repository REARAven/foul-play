"""External private-registry and persistent shuffled-bag contracts.

This package validates configuration and file integrity only. It deliberately
does not track battles, react to challenges, or provision packed teams.

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
from .errors import BlindPoolValidationError
from .fingerprint import compute_registry_fingerprint
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
    "SCHEMA_VERSION",
    "STATE_SCHEMA_VERSION",
    "SUPPORTED_FORMAT_ID",
    "BlindPoolBagState",
    "BlindPoolBagStore",
    "BlindPoolConfig",
    "BlindPoolEntry",
    "BlindPoolRegistry",
    "BlindPoolReservation",
    "BlindPoolStateConfig",
    "BlindPoolValidationError",
    "compute_registry_fingerprint",
    "get_active_blind_pool_entries",
    "get_blind_pool_entry_by_id",
    "load_blind_pool_config",
    "load_blind_pool_state_config",
    "load_blind_pool_registry",
    "validate_blind_pool_config",
    "validate_blind_pool_bag_state",
    "validate_blind_pool_registry",
    "validate_blind_pool_state_config",
)
