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
from .models import BlindPoolConfig, BlindPoolStateConfig
from .ownership import (
    BlindPoolDeploymentOwnerGuard,
    acquire_blind_pool_deployment_owner,
)
from .selection import BlindPoolSelectionSnapshot, create_canonical_selection_snapshot
from .state import ACCEPT_SENT_PHASE, RESERVATION_PHASE


logger = logging.getLogger(__name__)

CANONICAL_REGISTRY_PATH_ENV = "TUGS_BLIND_CANONICAL_REGISTRY"


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

    def __post_init__(self) -> None:
        invalid = False
        try:
            private_root = Path(self.private_root)
            registry_path = Path(self.canonical_registry_path)
            state_path = Path(self.state_path)
        except (TypeError, ValueError):
            invalid = True
            private_root = registry_path = state_path = None
        if invalid or not all(
            path is not None and path.is_absolute()
            for path in (private_root, registry_path, state_path)
        ):
            raise _error(
                BlindCanonicalErrorCategory.CONFIGURATION,
                "blind_canonical_path_invalid",
                "Blind canonical startup paths must be explicit absolute paths",
            ) from None
        object.__setattr__(self, "private_root", private_root)
        object.__setattr__(self, "canonical_registry_path", registry_path)
        object.__setattr__(self, "state_path", state_path)

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
    names = (PRIVATE_ROOT_ENV, CANONICAL_REGISTRY_PATH_ENV, STATE_PATH_ENV)
    configured = tuple(values.get(name) for name in names)
    if any(not isinstance(value, str) or not value for value in configured):
        raise _error(
            BlindCanonicalErrorCategory.CONFIGURATION,
            "blind_canonical_config_incomplete",
            "Blind canonical startup requires all explicit path inputs",
        ) from None
    return BlindCanonicalStartupConfig(*(Path(value) for value in configured))


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
            snapshot_failed = False
            try:
                recovered = store.snapshot()
            except BlindPoolValidationError:
                snapshot_failed = True
                recovered = None
            if not snapshot_failed and recovered is not None:
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
