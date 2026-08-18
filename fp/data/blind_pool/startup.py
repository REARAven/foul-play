"""Explicit, network-free preparation for canonical Blind Ladder startup."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
import os
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn

from .bag import BlindPoolBagStore, ShuffleSource
from .canonical_models import CanonicalArtifactError, CanonicalRuntimeRegistry
from .canonical_registry import load_canonical_runtime_registry
from .config import validate_blind_pool_state_config
from .errors import BlindPoolReconciliationRequired, BlindPoolValidationError
from .leaderboard import PUBLIC_TEAM_KIND_BOT, PUBLIC_TEAM_KIND_PLAYER
from .models import BlindPoolConfig, BlindPoolStateConfig
from .ownership import (
    BlindPoolDeploymentOwnerGuard,
    acquire_blind_pool_deployment_owner,
)
from .rating import RATING_STATE_PATH_ENV
from .rating_state import BlindRatingStateStore, validate_rating_state_config
from .result_ledger import (
    RESULT_LEDGER_PATH_ENV,
    BlindResultLedgerStore,
    validate_result_ledger_config,
)
from .team_public_registry import (
    TEAM_PUBLIC_REGISTRY_PATH_ENV,
    BlindTeamPublicRegistryStore,
    validate_team_public_registry_config,
)
from .team_rating_state import (
    TEAM_RATING_STATE_PATH_ENV,
    BlindTeamRatingStateStore,
    validate_team_rating_state_config,
)
from .team_result_ledger import (
    TEAM_RESULT_LEDGER_PATH_ENV,
    BlindTeamResultLedgerState,
    BlindTeamResultLedgerStore,
    validate_team_result_ledger_config,
)
from .selection import BlindPoolSelectionSnapshot, create_canonical_selection_snapshot
from .state import ACCEPT_SENT_PHASE, RESERVATION_PHASE


logger = logging.getLogger(__name__)

CANONICAL_REGISTRY_PATH_ENV = "TUGS_BLIND_CANONICAL_REGISTRY"
TEAM_LADDER_ENABLED_ENV = "TUGS_BLIND_TEAM_LADDER_ENABLED"


class BlindCanonicalErrorCategory(str, Enum):
    CONFIGURATION = "configuration"
    DEPLOYMENT_INTEGRITY = "deployment_integrity"
    DEPLOYMENT_OWNERSHIP = "deployment_ownership"
    RECOVERY_REQUIRED = "recovery_required"
    RECONCILIATION_MISMATCH = "reconciliation_mismatch"
    RECONCILIATION_NOT_APPLICABLE = "reconciliation_not_applicable"
    TRANSIENT_NETWORK = "transient_network"


class BlindCanonicalActivationError(RuntimeError):
    """Stable operator-facing activation error without private context."""

    def __init__(
        self,
        category: BlindCanonicalErrorCategory,
        code: str,
        message: str,
    ) -> None:
        self.category = category
        self.code = code
        self.operator_message = message
        super().__init__("{}: {}".format(code, message))

    def __repr__(self) -> str:
        return "BlindCanonicalActivationError(category={!r}, code={!r})".format(
            self.category.value,
            self.code,
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError("BlindCanonicalActivationError serialization is disabled")


def _error(
    category: BlindCanonicalErrorCategory,
    code: str,
    message: str,
) -> BlindCanonicalActivationError:
    return BlindCanonicalActivationError(category, code, message)


@dataclass(frozen=True, slots=True, repr=False)
class BlindCanonicalStartupConfig:
    """Explicit private paths whose ordinary representation is fully redacted."""

    private_root: Path = field(repr=False)
    canonical_registry_path: Path = field(repr=False)
    state_path: Path = field(repr=False)
    result_ledger_path: Path | None = field(default=None, repr=False)
    rating_state_path: Path | None = field(default=None, repr=False)
    team_result_ledger_path: Path | None = field(default=None, repr=False)
    team_rating_state_path: Path | None = field(default=None, repr=False)
    team_public_registry_path: Path | None = field(default=None, repr=False)
    team_ladder_enabled: bool = False

    def __post_init__(self) -> None:
        invalid = False
        try:
            private_root = Path(self.private_root)
            registry_path = Path(self.canonical_registry_path)
            state_path = Path(self.state_path)
            result_ledger_path = (
                None
                if self.result_ledger_path is None
                else Path(self.result_ledger_path)
            )
            rating_state_path = (
                None if self.rating_state_path is None else Path(self.rating_state_path)
            )
            team_result_ledger_path = (
                None
                if self.team_result_ledger_path is None
                else Path(self.team_result_ledger_path)
            )
            team_rating_state_path = (
                None
                if self.team_rating_state_path is None
                else Path(self.team_rating_state_path)
            )
            team_public_registry_path = (
                None
                if self.team_public_registry_path is None
                else Path(self.team_public_registry_path)
            )
        except (TypeError, ValueError):
            invalid = True
            private_root = registry_path = state_path = result_ledger_path = None
            rating_state_path = None
            team_result_ledger_path = team_rating_state_path = None
            team_public_registry_path = None
        if (
            invalid
            or not all(
                path is not None and path.is_absolute()
                for path in (private_root, registry_path, state_path)
            )
            or (result_ledger_path is not None and not result_ledger_path.is_absolute())
            or (rating_state_path is not None and not rating_state_path.is_absolute())
            or (
                team_result_ledger_path is not None
                and not team_result_ledger_path.is_absolute()
            )
            or (
                team_rating_state_path is not None
                and not team_rating_state_path.is_absolute()
            )
            or (
                team_public_registry_path is not None
                and not team_public_registry_path.is_absolute()
            )
            or type(self.team_ladder_enabled) is not bool
        ):
            raise _error(
                BlindCanonicalErrorCategory.CONFIGURATION,
                "blind_canonical_path_invalid",
                "Blind canonical startup paths must be explicit absolute paths",
            ) from None
        object.__setattr__(self, "private_root", private_root)
        object.__setattr__(self, "canonical_registry_path", registry_path)
        object.__setattr__(self, "state_path", state_path)
        object.__setattr__(self, "result_ledger_path", result_ledger_path)
        object.__setattr__(self, "rating_state_path", rating_state_path)
        object.__setattr__(self, "team_result_ledger_path", team_result_ledger_path)
        object.__setattr__(self, "team_rating_state_path", team_rating_state_path)
        object.__setattr__(self, "team_public_registry_path", team_public_registry_path)

    def __repr__(self) -> str:
        return "BlindCanonicalStartupConfig(configured=True)"

    def __str__(self) -> str:
        return repr(self)

    def __reduce__(self) -> NoReturn:
        raise TypeError("BlindCanonicalStartupConfig serialization is disabled")


@dataclass(frozen=True, slots=True, repr=False)
class PreparedBlindCanonicalDeployment:
    """Immutable deployment identity plus one process-lifetime owner guard."""

    _config: BlindCanonicalStartupConfig = field(repr=False)
    _state_config: BlindPoolStateConfig = field(repr=False)
    _registry: CanonicalRuntimeRegistry = field(repr=False)
    _selection: BlindPoolSelectionSnapshot = field(repr=False)
    _store: BlindPoolBagStore = field(repr=False)
    _owner: BlindPoolDeploymentOwnerGuard = field(repr=False)
    _result_store: BlindResultLedgerStore | BlindTeamResultLedgerStore = field(
        repr=False
    )
    _rating_store: BlindRatingStateStore | BlindTeamRatingStateStore = field(repr=False)
    _team_public_store: BlindTeamPublicRegistryStore | None = field(
        default=None, repr=False
    )
    _team_mode: bool = False

    @property
    def active_count(self) -> int:
        return len(self._selection.active_ids)

    @property
    def owner_held(self) -> bool:
        return self._owner.held

    def create_runtime(
        self,
        transport: object,
        submit_team: Callable[[str], Any],
        initialize_battle: Callable[..., Any],
        **runtime_options: object,
    ):
        """Bind one connection-scoped exact runtime to the prepared deployment."""

        from .canonical_runtime import CanonicalBlindRuntime
        from .lifecycle import BlindExactChallengeProtocol

        return CanonicalBlindRuntime(
            self._state_config,
            self._registry,
            transport,
            submit_team,
            initialize_battle,
            exact_protocol=BlindExactChallengeProtocol.from_transport(transport),
            prepared_store=self._store,
            result_store=self._result_store,
            rating_store=self._rating_store,
            team_public_store=self._team_public_store,
            team_mode=self._team_mode,
            **runtime_options,
        )

    def close(self) -> None:
        self._owner.close()

    def __repr__(self) -> str:
        return "PreparedBlindCanonicalDeployment(active_count={!r}, owned={!r})".format(
            self.active_count,
            self.owner_held,
        )

    def __str__(self) -> str:
        return repr(self)

    def __reduce__(self) -> NoReturn:
        raise TypeError("PreparedBlindCanonicalDeployment serialization is disabled")


def load_blind_canonical_startup_config(
    environ: Mapping[str, str] | None = None,
) -> BlindCanonicalStartupConfig:
    """Read canonical environment only after the explicit mode is selected."""

    from .config import PRIVATE_ROOT_ENV, STATE_PATH_ENV

    values = os.environ if environ is None else environ
    enabled_value = values.get(TEAM_LADDER_ENABLED_ENV)
    if enabled_value not in {None, "", "0", "1"}:
        raise _error(
            BlindCanonicalErrorCategory.CONFIGURATION,
            "blind_team_ladder_gate_invalid",
            "Blind team ladder feature gate is invalid",
        ) from None
    team_mode = enabled_value == "1"
    names = (PRIVATE_ROOT_ENV, CANONICAL_REGISTRY_PATH_ENV, STATE_PATH_ENV) + (
        (
            TEAM_RESULT_LEDGER_PATH_ENV,
            TEAM_RATING_STATE_PATH_ENV,
            TEAM_PUBLIC_REGISTRY_PATH_ENV,
        )
        if team_mode
        else (RESULT_LEDGER_PATH_ENV, RATING_STATE_PATH_ENV)
    )
    configured = tuple(values.get(name) for name in names)
    if any(not isinstance(value, str) or not value for value in configured):
        raise _error(
            BlindCanonicalErrorCategory.CONFIGURATION,
            "blind_canonical_config_incomplete",
            "Blind canonical startup requires all explicit path inputs",
        ) from None
    paths = tuple(Path(value) for value in configured)
    if team_mode:
        return BlindCanonicalStartupConfig(
            paths[0],
            paths[1],
            paths[2],
            team_result_ledger_path=paths[3],
            team_rating_state_path=paths[4],
            team_public_registry_path=paths[5],
            team_ladder_enabled=True,
        )
    return BlindCanonicalStartupConfig(*paths)


def _close_owner_after_failure(owner: BlindPoolDeploymentOwnerGuard) -> None:
    close_failed = False
    try:
        owner.close()
    except BlindPoolValidationError:
        close_failed = True
    if close_failed:
        raise _error(
            BlindCanonicalErrorCategory.DEPLOYMENT_OWNERSHIP,
            "blind_canonical_owner_release_failed",
            "Blind canonical deployment ownership could not be released",
        ) from None


def _state_error(code: str) -> BlindCanonicalActivationError:
    if code == "registry_fingerprint_mismatch":
        return _error(
            BlindCanonicalErrorCategory.RECOVERY_REQUIRED,
            "blind_canonical_state_registry_mismatch",
            "Blind canonical state requires explicit operator recovery",
        )
    return _error(
        BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
        "blind_canonical_state_invalid",
        "Blind canonical state is invalid",
    )


def _load_and_recover_selection_state(
    store: BlindPoolBagStore,
    owner: BlindPoolDeploymentOwnerGuard,
):
    """Validate one selection state and clear only a pre-accept reservation."""

    state_code: str | None = None
    try:
        state = store.initialize_or_load()
    except BlindPoolValidationError as error:
        state_code = error.code
        state = None
    if state_code is not None:
        _close_owner_after_failure(owner)
        raise _state_error(state_code) from None
    assert state is not None

    reservation = state.reservation
    if reservation is not None and reservation.phase == ACCEPT_SENT_PHASE:
        _close_owner_after_failure(owner)
        raise _error(
            BlindCanonicalErrorCategory.RECOVERY_REQUIRED,
            "blind_canonical_recovery_required",
            "Blind canonical acceptance state requires explicit recovery",
        ) from None
    if reservation is not None and reservation.phase == RESERVATION_PHASE:
        release_failed = False
        try:
            store.release_reservation(reservation.reservation_id)
        except BlindPoolValidationError:
            release_failed = True
        if release_failed:
            try:
                recovered = store.snapshot()
            except BlindPoolValidationError:
                recovered = None
            if recovered is not None:
                if recovered.reservation is None:
                    release_failed = False
                elif recovered.reservation.phase == ACCEPT_SENT_PHASE:
                    _close_owner_after_failure(owner)
                    raise _error(
                        BlindCanonicalErrorCategory.RECOVERY_REQUIRED,
                        "blind_canonical_recovery_required",
                        "Blind canonical acceptance state requires explicit recovery",
                    ) from None
        if release_failed:
            _close_owner_after_failure(owner)
            raise _error(
                BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
                "blind_canonical_startup_recovery_failed",
                "Blind canonical startup recovery could not be proven",
            ) from None
    return state


def prepare_blind_canonical_deployment(
    config: BlindCanonicalStartupConfig,
    *,
    repository_root: str | Path | None = None,
    random_source: ShuffleSource | None = None,
    reservation_id_factory: Callable[[], str] | None = None,
    lock_timeout_seconds: float = 5.0,
    owner_timeout_seconds: float = 0.25,
) -> PreparedBlindCanonicalDeployment:
    """Validate and recover one canonical deployment without any network action."""

    if not isinstance(config, BlindCanonicalStartupConfig):
        raise _error(
            BlindCanonicalErrorCategory.CONFIGURATION,
            "blind_canonical_config_invalid",
            "Blind canonical startup configuration is invalid",
        ) from None

    registry_failed = False
    try:
        registry = load_canonical_runtime_registry(
            config.private_root,
            config.canonical_registry_path,
            repository_root=repository_root,
        )
    except CanonicalArtifactError:
        registry_failed = True
        registry = None
    if registry_failed:
        raise _error(
            BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
            "blind_canonical_registry_invalid",
            "Blind canonical deployment validation failed",
        ) from None
    assert registry is not None

    selection_failed = False
    try:
        selection = create_canonical_selection_snapshot(registry)
        state_config = validate_blind_pool_state_config(
            BlindPoolStateConfig(
                BlindPoolConfig(config.private_root, config.canonical_registry_path),
                config.state_path,
            ),
            repository_root=repository_root,
        )
        store = BlindPoolBagStore.from_selection_snapshot(
            state_config,
            selection,
            random_source=random_source,
            reservation_id_factory=reservation_id_factory,
            lock_timeout_seconds=lock_timeout_seconds,
            team_mode=config.team_ladder_enabled,
        )
    except BlindPoolValidationError:
        selection_failed = True
        selection = state_config = store = None
    if selection_failed:
        raise _error(
            BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
            "blind_canonical_deployment_invalid",
            "Blind canonical deployment validation failed",
        ) from None
    assert selection is not None and state_config is not None and store is not None

    result_store = None
    rating_store = None
    team_public_store = None
    persistence_failed = False
    try:
        if config.team_ladder_enabled:
            if (
                config.team_result_ledger_path is None
                or config.team_rating_state_path is None
                or config.team_public_registry_path is None
            ):
                raise BlindPoolValidationError(
                    "team_persistence_config_incomplete",
                    "Team ladder persistence configuration is incomplete",
                )
            public_config = validate_team_public_registry_config(
                config.team_public_registry_path,
                private_root=config.private_root,
                canonical_registry_path=config.canonical_registry_path,
                selection_state_path=config.state_path,
                result_ledger_path=config.team_result_ledger_path,
                rating_state_path=config.team_rating_state_path,
                repository_root=repository_root,
            )
            team_public_store = BlindTeamPublicRegistryStore(
                public_config,
                lock_timeout_seconds=lock_timeout_seconds,
            )
            result_config = validate_team_result_ledger_config(
                config.team_result_ledger_path,
                private_root=config.private_root,
                registry_path=config.canonical_registry_path,
                selection_state_path=config.state_path,
                rating_state_path=config.team_rating_state_path,
                public_registry_path=config.team_public_registry_path,
                repository_root=repository_root,
            )
            result_store = BlindTeamResultLedgerStore(
                result_config,
                lock_timeout_seconds=lock_timeout_seconds,
            )
            rating_config = validate_team_rating_state_config(
                config.team_rating_state_path,
                private_root=config.private_root,
                canonical_registry_path=config.canonical_registry_path,
                selection_state_path=config.state_path,
                result_ledger_path=config.team_result_ledger_path,
                public_registry_path=config.team_public_registry_path,
                repository_root=repository_root,
            )
            rating_store = BlindTeamRatingStateStore(
                rating_config,
                lock_timeout_seconds=lock_timeout_seconds,
            )
        else:
            if config.result_ledger_path is None or config.rating_state_path is None:
                raise BlindPoolValidationError(
                    "account_persistence_config_incomplete",
                    "Account ladder persistence configuration is incomplete",
                )
            result_config = validate_result_ledger_config(
                config.result_ledger_path,
                private_root=config.private_root,
                registry_path=config.canonical_registry_path,
                selection_state_path=config.state_path,
                repository_root=repository_root,
            )
            result_store = BlindResultLedgerStore(
                result_config,
                lock_timeout_seconds=lock_timeout_seconds,
            )
            rating_config = validate_rating_state_config(
                config.rating_state_path,
                private_root=config.private_root,
                registry_path=config.canonical_registry_path,
                selection_state_path=config.state_path,
                result_ledger_path=config.result_ledger_path,
                repository_root=repository_root,
            )
            rating_store = BlindRatingStateStore(
                rating_config,
                lock_timeout_seconds=lock_timeout_seconds,
            )
    except BlindPoolValidationError:
        persistence_failed = True
    if persistence_failed:
        raise _error(
            BlindCanonicalErrorCategory.CONFIGURATION,
            "blind_ladder_persistence_config_invalid",
            "Blind canonical persistence configuration is invalid",
        ) from None
    assert result_store is not None and rating_store is not None

    ownership_code: str | None = None
    try:
        owner = acquire_blind_pool_deployment_owner(
            state_config,
            timeout_seconds=owner_timeout_seconds,
            repository_root=repository_root,
        )
    except BlindPoolValidationError as error:
        ownership_code = error.code
        owner = None
    if ownership_code is not None:
        raise _error(
            BlindCanonicalErrorCategory.DEPLOYMENT_OWNERSHIP,
            "blind_canonical_ownership_unavailable",
            "Blind canonical deployment is already active or unavailable",
        ) from None
    assert owner is not None

    state = None
    if config.team_ladder_enabled:
        state = _load_and_recover_selection_state(store, owner)
        assert team_public_store is not None
        public_failed = False
        try:
            public_state = team_public_store.load()
            for team_id in selection.active_ids:
                identity = public_state.identity(team_id)
                if identity is None or identity.kind != PUBLIC_TEAM_KIND_BOT:
                    raise BlindPoolValidationError(
                        "team_public_registry_bot_missing",
                        "Active bot identity is not registered",
                    )
        except BlindPoolValidationError:
            public_failed = True
        if public_failed:
            _close_owner_after_failure(owner)
            raise _error(
                BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
                "blind_team_public_registry_invalid",
                "Blind canonical team public registry is invalid",
            ) from None

    result_code: str | None = None
    try:
        result_state = result_store.require_ready()
    except BlindPoolValidationError as error:
        result_code = error.code
    if result_code is not None:
        _close_owner_after_failure(owner)
        if result_code in {
            "result_recovery_required",
            "team_result_recovery_required",
        }:
            raise _error(
                BlindCanonicalErrorCategory.RECOVERY_REQUIRED,
                "blind_result_recovery_required",
                "Blind canonical result persistence requires explicit recovery",
            ) from None
        raise _error(
            BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
            "blind_result_ledger_invalid",
            "Blind canonical result persistence is invalid",
        ) from None

    if config.team_ladder_enabled:
        assert isinstance(result_state, BlindTeamResultLedgerState)
        public_history_failed = any(
            public_state.identity(record.player_team_id) is None
            or public_state.identity(record.player_team_id).kind
            != PUBLIC_TEAM_KIND_PLAYER
            or public_state.identity(record.bot_team_id) is None
            or public_state.identity(record.bot_team_id).kind != PUBLIC_TEAM_KIND_BOT
            for record in result_state.completed_results
        )
        if public_history_failed:
            _close_owner_after_failure(owner)
            raise _error(
                BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
                "blind_team_public_registry_invalid",
                "Blind canonical team public registry is invalid",
            ) from None

    rating_code: str | None = None
    try:
        rating_store.sync(result_state)
    except BlindPoolValidationError as error:
        rating_code = error.code
    if rating_code is not None:
        _close_owner_after_failure(owner)
        if rating_code in {"rating_not_initialized", "team_rating_not_initialized"}:
            raise _error(
                BlindCanonicalErrorCategory.RECOVERY_REQUIRED,
                "blind_rating_initialization_required",
                "Blind canonical rating state requires explicit initialization",
            ) from None
        raise _error(
            BlindCanonicalErrorCategory.RECOVERY_REQUIRED,
            "blind_rating_recovery_required",
            "Blind canonical rating state requires explicit recovery",
        ) from None

    if not config.team_ladder_enabled:
        state = _load_and_recover_selection_state(store, owner)
    assert state is not None

    logger.info("Canonical Blind Ladder deployment validated")
    logger.info(
        "Canonical Blind Ladder active pool count: {}".format(len(selection.active_ids))
    )
    logger.info("Canonical Blind Ladder state ready")
    return PreparedBlindCanonicalDeployment(
        config,
        state_config,
        registry,
        selection,
        store,
        owner,
        result_store,
        rating_store,
        team_public_store,
        config.team_ladder_enabled,
    )


def classify_blind_canonical_runtime_error(
    error: Exception,
) -> BlindCanonicalActivationError:
    """Translate one already-sanitized runtime failure into an operator category."""

    if isinstance(error, BlindPoolReconciliationRequired):
        return _error(
            BlindCanonicalErrorCategory.RECOVERY_REQUIRED,
            "blind_canonical_recovery_required",
            "Blind canonical acceptance state requires explicit recovery",
        )
    code = getattr(error, "code", "")
    if code in {
        "canonical_rating_sync_failed",
        "canonical_rating_update_missing",
    }:
        return _error(
            BlindCanonicalErrorCategory.RECOVERY_REQUIRED,
            "blind_rating_recovery_required",
            "Blind canonical rating persistence requires recovery",
        )
    if code in {
        "battle_initialization_failed",
        "result_intent_failed",
        "result_selection_commit_failed",
        "canonical_result_recovery_required",
        "canonical_result_terminal_missing",
    }:
        return _error(
            BlindCanonicalErrorCategory.RECOVERY_REQUIRED,
            "blind_result_recovery_required",
            "Blind canonical result persistence requires explicit recovery",
        )
    if code in {
        "exact_protocol_enable_failed",
        "exact_challenge_receive_failed",
        "team_submission_failed",
    }:
        return _error(
            BlindCanonicalErrorCategory.TRANSIENT_NETWORK,
            "blind_canonical_transport_failed",
            "Blind canonical network operation failed",
        )
    return _error(
        BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
        "blind_canonical_runtime_failed",
        "Blind canonical runtime stopped safely",
    )
