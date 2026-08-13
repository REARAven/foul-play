"""External private-registry contract for future Blind Ladder phases.

This package validates configuration and file integrity only. It deliberately
does not select teams, reserve entries, track battles, or provision packed
teams.

Schema v1 is exactly ``schema_version``, ``registry_version``, ``format_id``,
and ``entries``. Each entry is exactly ``team_id``, ``active``, ``team_file``,
and ``sha256``; the digest covers the referenced file's raw bytes.
"""

from .config import (
    PRIVATE_ROOT_ENV,
    REGISTRY_PATH_ENV,
    load_blind_pool_config,
    validate_blind_pool_config,
)
from .errors import BlindPoolValidationError
from .loader import (
    get_active_blind_pool_entries,
    get_blind_pool_entry_by_id,
    load_blind_pool_registry,
    validate_blind_pool_registry,
)
from .models import (
    SCHEMA_VERSION,
    SUPPORTED_FORMAT_ID,
    BlindPoolConfig,
    BlindPoolEntry,
    BlindPoolRegistry,
)

__all__ = (
    "PRIVATE_ROOT_ENV",
    "REGISTRY_PATH_ENV",
    "SCHEMA_VERSION",
    "SUPPORTED_FORMAT_ID",
    "BlindPoolConfig",
    "BlindPoolEntry",
    "BlindPoolRegistry",
    "BlindPoolValidationError",
    "get_active_blind_pool_entries",
    "get_blind_pool_entry_by_id",
    "load_blind_pool_config",
    "load_blind_pool_registry",
    "validate_blind_pool_config",
    "validate_blind_pool_registry",
)
