"""Dormant canonical Blind Ladder challenge-runtime integration."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, NoReturn, Protocol

from .bag import BlindPoolBagStore, ShuffleSource
from .canonical_artifacts import load_canonical_team_artifact
from .canonical_models import (
    CanonicalArtifactError,
    CanonicalRuntimeRegistry,
    CanonicalTeamArtifact,
)
from .errors import (
    BlindPoolLifecycleError,
    BlindPoolReconciliationRequired,
    BlindPoolValidationError,
)
from .lifecycle import (
    BLIND_LADDER_FORMAT,
    BLIND_LADDER_MODE,
    BlindChallengeTransport,
    BlindPoolLifecycleCoordinator,
)
from .models import BlindPoolBattleRoom, BlindPoolStateConfig
from .selection import create_canonical_selection_snapshot
from .state import RESERVATION_PHASE


_IDLE = "idle"
_READY = "ready"
_PREPARING = "preparing"
_PREPARED = "prepared"
_INITIALIZING = "initializing"
_INITIALIZED = "initialized"
_FAILED = "failed"


class CanonicalTeamSubmitter(Protocol):
    def __call__(self, packed_team: str) -> Awaitable[None]: ...


class CanonicalBattleInitializer(Protocol):
    def __call__(
        self,
        room: BlindPoolBattleRoom,
        team_projection: list[dict[str, object]],
    ) -> Awaitable[Any]: ...


def _runtime_error(code: str, message: str) -> BlindPoolLifecycleError:
    return BlindPoolLifecycleError(code, message)


class CanonicalBlindRuntimeSession:
    """Private state for exactly one explicitly bounded lifecycle attempt."""

    __slots__ = (
        "_artifact",
        "_attempt_generation",
        "_initialize_battle",
        "_phase",
        "_prepared_team_id",
        "_registry",
        "_submit_team",
    )

    def __init__(
        self,
        canonical_registry: CanonicalRuntimeRegistry,
        submit_team: CanonicalTeamSubmitter,
        initialize_battle: CanonicalBattleInitializer,
    ) -> None:
        if not isinstance(canonical_registry, CanonicalRuntimeRegistry):
            raise _runtime_error(
                "canonical_runtime_registry_invalid",
                "Canonical Blind Ladder runtime registry is invalid",
            ) from None
        if not callable(submit_team) or not callable(initialize_battle):
            raise _runtime_error(
                "canonical_runtime_callback_invalid",
                "Canonical Blind Ladder runtime callback is invalid",
            ) from None
        self._registry = canonical_registry
        self._submit_team = submit_team
        self._initialize_battle = initialize_battle
        self._artifact: CanonicalTeamArtifact | None = None
        self._prepared_team_id: str | None = None
        self._phase = _IDLE
        self._attempt_generation = 0

    def __repr__(self) -> str:
        return "CanonicalBlindRuntimeSession(phase={!r}, prepared={!r})".format(
            self._phase,
            self._artifact is not None,
        )

    def __str__(self) -> str:
        return repr(self)

    def __reduce__(self) -> NoReturn:
        raise TypeError("CanonicalBlindRuntimeSession serialization is disabled")

    @property
    def prepared(self) -> bool:
        return self._phase == _PREPARED and self._artifact is not None

    @property
    def phase(self) -> str:
        return self._phase

    def begin_attempt(self) -> None:
        if self._phase != _IDLE or self._artifact is not None:
            raise _runtime_error(
                "canonical_runtime_attempt_active",
                "Canonical Blind Ladder runtime attempt is already active",
            ) from None
        self._prepared_team_id = None
        self._attempt_generation += 1
        self._phase = _READY

    def end_attempt(self) -> None:
        self._artifact = None
        self._prepared_team_id = None
        self._attempt_generation += 1
        self._phase = _IDLE

    async def prepare_team(self, team_id: str) -> None:
        """Verify one selected artifact and submit its exact packed wire value."""

        if self._phase != _READY:
            raise _runtime_error(
                "canonical_runtime_prepare_invalid",
                "Canonical Blind Ladder team preparation is not permitted",
            ) from None
        if team_id not in self._registry.active_ids:
            self._phase = _FAILED
            raise _runtime_error(
                "canonical_runtime_selection_invalid",
                "Canonical Blind Ladder selected team is not active",
            ) from None
        self._phase = _PREPARING
        attempt_generation = self._attempt_generation
        verification_failed = False
        verification_internal_failed = False
        try:
            artifact = load_canonical_team_artifact(self._registry, team_id)
        except CanonicalArtifactError:
            verification_failed = True
            artifact = None
        except Exception:
            verification_internal_failed = True
            artifact = None
        if self._attempt_generation != attempt_generation:
            raise _runtime_error(
                "canonical_runtime_attempt_stale",
                "Canonical Blind Ladder runtime attempt is stale",
            ) from None
        if verification_failed:
            self._phase = _FAILED
            raise _runtime_error(
                "canonical_runtime_artifact_rejected",
                "Canonical Blind Ladder artifact verification failed",
            ) from None
        if verification_internal_failed:
            self._phase = _FAILED
            raise _runtime_error(
                "canonical_runtime_internal_failure",
                "Canonical Blind Ladder runtime failed internally",
            ) from None

        assert artifact is not None
        if self._phase != _PREPARING or self._attempt_generation != attempt_generation:
            raise _runtime_error(
                "canonical_runtime_attempt_stale",
                "Canonical Blind Ladder runtime attempt is stale",
            ) from None
        packed_access_failed = False
        try:
            packed = artifact.packed_for_submission()
        except Exception:
            packed_access_failed = True
            packed = None
        if packed_access_failed:
            self._phase = _FAILED
            raise _runtime_error(
                "canonical_runtime_internal_failure",
                "Canonical Blind Ladder runtime failed internally",
            ) from None
        assert packed is not None
        submission_failed = False
        cancellation: asyncio.CancelledError | None = None
        # This callback is the existing update_team-style /utm boundary. A send
        # failure may be ambiguous about the upload itself, but /utm alone cannot
        # accept a challenge, create a room, or consume the reservation. It is
        # therefore still safely pre-accept under Phase 4's release model.
        try:
            await self._submit_team(packed)
        except asyncio.CancelledError as error:
            cancellation = error
        except Exception:
            submission_failed = True
        if self._phase != _PREPARING or self._attempt_generation != attempt_generation:
            raise _runtime_error(
                "canonical_runtime_attempt_stale",
                "Canonical Blind Ladder runtime attempt is stale",
            ) from None
        if cancellation is not None:
            self._phase = _FAILED
            raise asyncio.CancelledError from None
        if submission_failed:
            self._phase = _FAILED
            raise _runtime_error(
                "canonical_runtime_submission_failed",
                "Canonical Blind Ladder team submission failed",
            ) from None

        self._artifact = artifact
        self._prepared_team_id = artifact.team_id
        self._phase = _PREPARED

    async def initialize_battle(self, room: BlindPoolBattleRoom) -> Any:
        """Transfer only one fresh minimal projection to the battle initializer."""

        artifact = self._artifact
        if (
            self._phase != _PREPARED
            or artifact is None
            or artifact.team_id != self._prepared_team_id
        ):
            raise _runtime_error(
                "canonical_runtime_initializer_invalid",
                "Canonical Blind Ladder battle initialization is not permitted",
            ) from None
        attempt_generation = self._attempt_generation
        self._phase = _INITIALIZING
        projection_failed = False
        try:
            projection = artifact.new_battle_team_projection()
        except Exception:
            projection_failed = True
            projection = None
        self._artifact = None
        if projection_failed:
            self._phase = _FAILED
            raise _runtime_error(
                "canonical_runtime_internal_failure",
                "Canonical Blind Ladder runtime failed internally",
            ) from None
        assert projection is not None

        initialization_failed = False
        cancellation: asyncio.CancelledError | None = None
        try:
            result = await self._initialize_battle(room, projection)
        except asyncio.CancelledError as error:
            cancellation = error
            result = None
        except Exception:
            initialization_failed = True
            result = None
        if self._attempt_generation != attempt_generation:
            raise _runtime_error(
                "canonical_runtime_attempt_stale",
                "Canonical Blind Ladder runtime attempt is stale",
            ) from None
        if cancellation is not None:
            self._phase = _FAILED
            raise asyncio.CancelledError from None
        if initialization_failed:
            self._phase = _FAILED
            raise _runtime_error(
                "canonical_runtime_initialization_failed",
                "Canonical Blind Ladder battle initialization failed",
            ) from None

        self._phase = _INITIALIZED
        return result


class CanonicalBlindRuntime:
    """Explicit dormant assembly of canonical selection and Phase 4 lifecycle."""

    def __init__(
        self,
        state_config: BlindPoolStateConfig,
        canonical_registry: CanonicalRuntimeRegistry,
        transport: BlindChallengeTransport,
        submit_team: CanonicalTeamSubmitter,
        initialize_battle: CanonicalBattleInitializer,
        *,
        random_source: ShuffleSource | None = None,
        reservation_id_factory: Callable[[], str] | None = None,
        lock_timeout_seconds: float = 5.0,
        format_id: str = BLIND_LADDER_FORMAT,
        mode: str = BLIND_LADDER_MODE,
        room_timeout_seconds: float = 30.0,
        monotonic: Callable[[], float] | None = None,
        max_room_candidates: int = 8,
    ) -> None:
        selection = create_canonical_selection_snapshot(canonical_registry)
        self._store = BlindPoolBagStore.from_selection_snapshot(
            state_config,
            selection,
            random_source=random_source,
            reservation_id_factory=reservation_id_factory,
            lock_timeout_seconds=lock_timeout_seconds,
        )
        self._session = CanonicalBlindRuntimeSession(
            canonical_registry,
            submit_team,
            initialize_battle,
        )
        lifecycle_kwargs: dict[str, object] = {
            "format_id": format_id,
            "mode": mode,
            "initialize_battle": self._session.initialize_battle,
            "room_timeout_seconds": room_timeout_seconds,
            "max_room_candidates": max_room_candidates,
        }
        if monotonic is not None:
            lifecycle_kwargs["monotonic"] = monotonic
        self._coordinator = BlindPoolLifecycleCoordinator(
            self._store,
            transport,
            self._prepare_reserved_team,
            **lifecycle_kwargs,
        )
        self._running = False

    def __repr__(self) -> str:
        return "CanonicalBlindRuntime(active={!r})".format(self._running)

    def __str__(self) -> str:
        return repr(self)

    def __reduce__(self) -> NoReturn:
        raise TypeError("CanonicalBlindRuntime serialization is disabled")

    @property
    def retains_selection(self) -> bool:
        """Expose only whether private selected material is currently retained."""

        return self._session.prepared

    async def startup(self) -> None:
        await self._coordinator.startup()

    async def _prepare_reserved_team(self, team_id: str) -> None:
        reservation_check_failed = False
        try:
            state = self._store.snapshot()
        except BlindPoolValidationError:
            reservation_check_failed = True
            state = None
        reservation = None if state is None else state.reservation
        if (
            reservation_check_failed
            or reservation is None
            or reservation.phase != RESERVATION_PHASE
            or reservation.team_id != team_id
        ):
            raise _runtime_error(
                "canonical_runtime_reservation_invalid",
                "Canonical Blind Ladder selection is not the current reservation",
            ) from None
        await self._session.prepare_team(team_id)

    async def run_once(self) -> Any:
        if self._running:
            raise _runtime_error(
                "canonical_runtime_already_active",
                "Canonical Blind Ladder runtime is already active",
            ) from None
        self._running = True
        try:
            self._session.begin_attempt()
        except BaseException:
            self._running = False
            raise
        lifecycle_failure: tuple[type[BlindPoolLifecycleError], str, str] | None = None
        cancelled = False
        result: Any = None
        try:
            result = await self._coordinator.run_once()
        except asyncio.CancelledError:
            cancelled = True
        except BlindPoolLifecycleError as error:
            messages = {
                "team_preparation_failed",
                "team_preparation_cleanup_failed",
                "battle_initialization_failed",
            }
            if error.code in messages:
                message = {
                    "team_preparation_failed": "Blind Ladder team preparation failed",
                    "team_preparation_cleanup_failed": (
                        "Blind Ladder team preparation cleanup failed"
                    ),
                    "battle_initialization_failed": (
                        "Blind Ladder battle initialization failed"
                    ),
                }[error.code]
            else:
                rendered = str(error)
                prefix = error.code + ": "
                message = (
                    rendered[len(prefix) :]
                    if rendered.startswith(prefix)
                    else "Blind Ladder lifecycle failed"
                )
            error_type = (
                BlindPoolReconciliationRequired
                if isinstance(error, BlindPoolReconciliationRequired)
                else BlindPoolLifecycleError
            )
            lifecycle_failure = (error_type, error.code, message)
        finally:
            self._session.end_attempt()
            self._running = False
        if cancelled:
            raise asyncio.CancelledError from None
        if lifecycle_failure is not None:
            error_type, code, message = lifecycle_failure
            raise error_type(code, message) from None
        return result

    def reconcile_as_room_created(self, reservation_id: str):
        return self._coordinator.reconcile_as_room_created(reservation_id)

    def reconcile_as_no_room(self, reservation_id: str):
        return self._coordinator.reconcile_as_no_room(reservation_id)

    def close_attempt(self) -> None:
        """Clear retained selected material for an abandoned outer attempt."""

        if self._running:
            raise _runtime_error(
                "canonical_runtime_attempt_active",
                "Canonical Blind Ladder runtime attempt is still active",
            ) from None
        self._session.end_attempt()
