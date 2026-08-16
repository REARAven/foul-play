"""External raw-source, canonical-artifact, and shuffled-bag contracts.

This package validates external state and exposes an explicitly selected,
dependency-injected canonical challenge lifecycle. It does not provision teams
or activate the sealed pool implicitly.

The existing raw-source registry contract and its fingerprint semantics remain
unchanged.
"""

from .bag import BlindPoolBagStore
from .canonical_artifacts import load_canonical_team_artifact
from .canonical_models import (
    CANONICAL_ARTIFACT_SCHEMA_VERSION,
    CANONICAL_FORMAT_ID,
    CANONICAL_METADATA_SCHEMA_VERSION,
    CANONICAL_PROVISIONER_PROFILE_VERSION,
    CANONICAL_REGISTRY_FINGERPRINT_PROFILE_VERSION,
    CANONICAL_REGISTRY_SCHEMA_VERSION,
    CANONICAL_SIDECAR_SCHEMA_VERSION,
    CanonicalArtifactError,
    CanonicalRuntimeRegistry,
    CanonicalTeamArtifact,
)
from .canonical_registry import (
    compute_canonical_registry_fingerprint,
    load_canonical_runtime_registry,
)
from .canonical_runtime import (
    CanonicalBlindRuntime,
    CanonicalBlindRuntimeSession,
)
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
    BlindExactChallengeProtocol,
    BlindExactRoomCorrelator,
    BlindPoolLifecycleCoordinator,
    BlindRoomCorrelator,
    normalize_showdown_identity,
    parse_challenge_end,
    parse_challenge_room_binding,
    parse_exact_challenge_offer,
    parse_incoming_challenge,
    parse_private_challenge_event,
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
    BlindChallengeEndEvent,
    BlindChallengeRoomBinding,
    BlindChallengeToken,
    BlindPoolBagState,
    BlindPoolBattleRoom,
    BlindPoolChallenge,
    BlindPoolConfig,
    BlindPoolEntry,
    BlindPoolRegistry,
    BlindPoolReservation,
    BlindPoolStateConfig,
)
from .ownership import (
    BlindPoolDeploymentOwnerGuard,
    acquire_blind_pool_deployment_owner,
)
from .reconciliation import (
    RECONCILIATION_CASE_DOMAIN,
    BlindReconciliationDisposition,
    derive_reconciliation_case,
    validate_reconciliation_case,
)
from .selection import (
    BlindPoolSelectionSnapshot,
    create_canonical_selection_snapshot,
    create_raw_selection_snapshot,
)
from .state import STATE_SCHEMA_VERSION, validate_blind_pool_bag_state
from .startup import (
    CANONICAL_REGISTRY_PATH_ENV,
    BlindCanonicalActivationError,
    BlindCanonicalErrorCategory,
    BlindCanonicalStartupConfig,
    PreparedBlindCanonicalDeployment,
    classify_blind_canonical_runtime_error,
    load_blind_canonical_startup_config,
    prepare_blind_canonical_deployment,
)

__all__ = (
    "PRIVATE_ROOT_ENV",
    "REGISTRY_PATH_ENV",
    "STATE_PATH_ENV",
    "CANONICAL_REGISTRY_PATH_ENV",
    "BLIND_LADDER_FORMAT",
    "BLIND_LADDER_MODE",
    "CANONICAL_ARTIFACT_SCHEMA_VERSION",
    "CANONICAL_FORMAT_ID",
    "CANONICAL_METADATA_SCHEMA_VERSION",
    "CANONICAL_PROVISIONER_PROFILE_VERSION",
    "CANONICAL_REGISTRY_FINGERPRINT_PROFILE_VERSION",
    "CANONICAL_REGISTRY_SCHEMA_VERSION",
    "CANONICAL_SIDECAR_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "STATE_SCHEMA_VERSION",
    "SUPPORTED_FORMAT_ID",
    "BlindChallengeEndEvent",
    "BlindChallengeRoomBinding",
    "BlindChallengeToken",
    "BlindExactChallengeProtocol",
    "BlindExactRoomCorrelator",
    "BlindPoolBagState",
    "BlindPoolDeploymentOwnerGuard",
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
    "BlindReconciliationDisposition",
    "BlindRoomCorrelator",
    "CanonicalArtifactError",
    "CanonicalBlindRuntime",
    "CanonicalBlindRuntimeSession",
    "CanonicalRuntimeRegistry",
    "CanonicalTeamArtifact",
    "BlindCanonicalActivationError",
    "BlindCanonicalErrorCategory",
    "BlindCanonicalStartupConfig",
    "PreparedBlindCanonicalDeployment",
    "RECONCILIATION_CASE_DOMAIN",
    "BlindPoolSelectionSnapshot",
    "compute_canonical_registry_fingerprint",
    "compute_registry_fingerprint",
    "acquire_blind_pool_deployment_owner",
    "classify_blind_canonical_runtime_error",
    "create_canonical_selection_snapshot",
    "create_raw_selection_snapshot",
    "derive_reconciliation_case",
    "get_active_blind_pool_entries",
    "get_blind_pool_entry_by_id",
    "load_blind_pool_config",
    "load_blind_pool_state_config",
    "load_blind_pool_registry",
    "load_canonical_runtime_registry",
    "load_blind_canonical_startup_config",
    "load_canonical_team_artifact",
    "normalize_showdown_identity",
    "parse_challenge_end",
    "parse_challenge_room_binding",
    "parse_exact_challenge_offer",
    "parse_incoming_challenge",
    "parse_private_challenge_event",
    "prepare_blind_canonical_deployment",
    "validate_blind_pool_config",
    "validate_blind_pool_bag_state",
    "validate_blind_pool_registry",
    "validate_blind_pool_state_config",
    "validate_reconciliation_case",
)
