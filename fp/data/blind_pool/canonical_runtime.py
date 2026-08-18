"""Dormant canonical Blind Ladder challenge-runtime integration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
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
    BlindExactChallengeProtocol,
    BlindPoolLifecycleCoordinator,
    normalize_showdown_identity,
)
from .models import (
    BlindPoolBattleRoom,
    BlindPoolChallenge,
    BlindPoolReservation,
    BlindPoolStateConfig,
)
from .rating import BlindRatingUpdate
from .rating_state import BlindRatingStateStore
from .result_ledger import (
    OUTCOME_PLAYER_LOSS,
    OUTCOME_PLAYER_WIN,
    OUTCOME_TIE,
    BlindResultLedgerStore,
)
from .team_public_registry import BlindTeamPublicRegistryStore
from .team_rating import (
    TEAM_OUTCOME_A_LOSS,
    TEAM_OUTCOME_A_WIN,
    TEAM_OUTCOME_TIE,
    BlindTeamRating,
    BlindTeamRatingUpdate,
)
from .team_rating_state import BlindTeamRatingStateStore
from .team_result_ledger import (
    BlindTeamCompletedBattleResult,
    BlindTeamResultLedgerStore,
)
from .selection import create_canonical_selection_snapshot
from .state import RESERVATION_PHASE


_IDLE = "idle"
_READY = "ready"
_PREPARING = "preparing"
_PREPARED = "prepared"
_INITIALIZING = "initializing"
_INITIALIZED = "initialized"
_FAILED = "failed"


@dataclass(frozen=True, slots=True)
class BlindTeamPublicRatingUpdate:
    """Post-terminal public-only view of one persisted team rating update."""

    player_name: str
    player_rating_before: int
    player_rating_after: int
    player_delta: int
    player_wins: int
    player_losses: int
    player_ties: int
    player_win_rate: str
    player_peak: int
    player_streak: str
    bot_name: str
    bot_rating_after: int
    bot_wins: int
    bot_losses: int
    bot_ties: int
    bot_win_rate: str


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
    """Explicit assembly of canonical selection and Phase 4 lifecycle."""

    def __init__(
        self,
        state_config: BlindPoolStateConfig,
        canonical_registry: CanonicalRuntimeRegistry,
        transport: BlindChallengeTransport,
        submit_team: CanonicalTeamSubmitter,
        initialize_battle: CanonicalBattleInitializer,
        *,
        exact_protocol: BlindExactChallengeProtocol,
        prepared_store: BlindPoolBagStore | None = None,
        random_source: ShuffleSource | None = None,
        reservation_id_factory: Callable[[], str] | None = None,
        lock_timeout_seconds: float = 5.0,
        format_id: str = BLIND_LADDER_FORMAT,
        mode: str = BLIND_LADDER_MODE,
        room_timeout_seconds: float = 30.0,
        monotonic: Callable[[], float] | None = None,
        max_room_candidates: int = 8,
        result_store: BlindResultLedgerStore | BlindTeamResultLedgerStore | None = None,
        rating_store: BlindRatingStateStore | BlindTeamRatingStateStore | None = None,
        team_public_store: BlindTeamPublicRegistryStore | None = None,
        team_mode: bool = False,
    ) -> None:
        if not isinstance(exact_protocol, BlindExactChallengeProtocol):
            raise _runtime_error(
                "canonical_exact_protocol_required",
                "Canonical Blind Ladder runtime requires exact challenge protocol",
            ) from None
        selection = create_canonical_selection_snapshot(canonical_registry)
        if prepared_store is None:
            self._store = BlindPoolBagStore.from_selection_snapshot(
                state_config,
                selection,
                random_source=random_source,
                reservation_id_factory=reservation_id_factory,
                lock_timeout_seconds=lock_timeout_seconds,
                team_mode=team_mode,
            )
        elif (
            not isinstance(prepared_store, BlindPoolBagStore)
            or prepared_store._config != state_config
            or prepared_store._fingerprint != selection.registry_fingerprint
            or prepared_store._active_ids != selection.active_ids
        ):
            raise _runtime_error(
                "canonical_prepared_store_invalid",
                "Canonical Blind Ladder prepared store is invalid",
            ) from None
        else:
            self._store = prepared_store
        if type(team_mode) is not bool:
            raise _runtime_error(
                "canonical_team_mode_invalid",
                "Canonical Blind Ladder team mode is invalid",
            ) from None
        if team_mode:
            stores_valid = (
                isinstance(result_store, BlindTeamResultLedgerStore)
                and isinstance(rating_store, BlindTeamRatingStateStore)
                and isinstance(team_public_store, BlindTeamPublicRegistryStore)
                and self._store._team_mode
            )
        else:
            stores_valid = (
                (
                    result_store is None
                    or isinstance(result_store, BlindResultLedgerStore)
                )
                and (
                    rating_store is None
                    or isinstance(rating_store, BlindRatingStateStore)
                )
                and team_public_store is None
                and not self._store._team_mode
            )
        if not stores_valid:
            raise _runtime_error(
                "canonical_persistence_store_invalid",
                "Canonical Blind Ladder persistence stores are invalid",
            ) from None
        if (result_store is None) != (rating_store is None):
            raise _runtime_error(
                "canonical_rating_store_invalid",
                "Canonical Blind Ladder result and rating stores must be paired",
            ) from None
        bot_id = normalize_showdown_identity(getattr(transport, "username", ""))
        if result_store is not None and not bot_id:
            raise _runtime_error(
                "canonical_result_identity_invalid",
                "Canonical Blind Ladder result identity is invalid",
            ) from None
        self._result_store = result_store
        self._rating_store = rating_store
        self._team_public_store = team_public_store
        self._team_mode = team_mode
        self._registry_fingerprint = canonical_registry.registry_fingerprint
        self._bot_id = bot_id
        self._active_result_battle_id: str | None = None
        self._result_finalized = False
        self._rating_synchronized = False
        self._last_rating_update: BlindRatingUpdate | None = None
        self._last_team_rating_update: BlindTeamPublicRatingUpdate | None = None
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
        if result_store is not None:
            lifecycle_kwargs["create_result_intent"] = self._create_result_intent
            lifecycle_kwargs["mark_result_selection_committed"] = (
                self._mark_result_selection_committed
            )
        if team_mode:
            lifecycle_kwargs["register_player_identity"] = (
                self._register_player_identity
            )
            lifecycle_kwargs["team_mode"] = True
        self._coordinator = BlindPoolLifecycleCoordinator(
            self._store,
            transport,
            self._prepare_reserved_team,
            exact_protocol=exact_protocol,
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

    @property
    def last_rating_update(self) -> BlindRatingUpdate | None:
        """Return the latest public-safe persisted player rating update."""

        return self._last_rating_update

    @property
    def last_team_rating_update(self) -> BlindTeamPublicRatingUpdate | None:
        """Return public team details only after terminal persistence."""

        return self._last_team_rating_update

    def _register_player_identity(self, identity) -> None:
        store = self._team_public_store
        if not self._team_mode or store is None:
            raise _runtime_error(
                "canonical_team_public_store_invalid",
                "Canonical Blind Ladder public registry is unavailable",
            ) from None
        store.register_player(identity)

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
            or reservation.challenge_token is None
        ):
            raise _runtime_error(
                "canonical_runtime_reservation_invalid",
                "Canonical Blind Ladder selection is not the current reservation",
            ) from None
        await self._session.prepare_team(team_id)

    def _create_result_intent(
        self,
        reservation: BlindPoolReservation,
        challenge: BlindPoolChallenge,
        room: BlindPoolBattleRoom,
    ) -> str:
        store = self._result_store
        if store is None or self._active_result_battle_id is not None:
            raise _runtime_error(
                "canonical_result_intent_invalid",
                "Canonical Blind Ladder result intent is invalid",
            ) from None
        player_id = normalize_showdown_identity(room.opponent_id)
        challenge_id = normalize_showdown_identity(challenge.challenger_id)
        if (
            not player_id
            or player_id != challenge_id
            or room.format_id != BLIND_LADDER_FORMAT
        ):
            raise _runtime_error(
                "canonical_result_identity_invalid",
                "Canonical Blind Ladder result identity is invalid",
            ) from None
        if self._team_mode:
            challenge_identity = challenge.player_team_identity
            if (
                not isinstance(store, BlindTeamResultLedgerStore)
                or challenge_identity is None
                or reservation.player_team_id != challenge_identity.team_id
                or reservation.player_team_display_name
                != challenge_identity.display_name
                or reservation.challenge_token is None
                or challenge.challenge_token is None
                or not reservation.challenge_token.matches(challenge.challenge_token)
            ):
                raise _runtime_error(
                    "canonical_team_identity_mismatch",
                    "Canonical Blind Ladder team identity does not match",
                ) from None
            pending = store.create_pending_intent(
                player_account_id=player_id,
                bot_account_id=self._bot_id,
                player_team_id=reservation.player_team_id,
                bot_team_id=reservation.team_id,
                reservation_id=reservation.reservation_id,
                room_id=room.room_id,
                format_id=room.format_id,
                registry_fingerprint=self._registry_fingerprint,
            )
        else:
            assert isinstance(store, BlindResultLedgerStore)
            pending = store.create_pending_intent(
                player_id=player_id,
                bot_id=self._bot_id,
                team_id=reservation.team_id,
                reservation_id=reservation.reservation_id,
                room_id=room.room_id,
                format_id=room.format_id,
                registry_fingerprint=self._registry_fingerprint,
            )
        self._active_result_battle_id = pending.battle_id
        self._result_finalized = False
        self._rating_synchronized = False
        self._last_rating_update = None
        self._last_team_rating_update = None
        return pending.battle_id

    def _mark_result_selection_committed(self, battle_id: str) -> None:
        store = self._result_store
        if store is None or battle_id != self._active_result_battle_id:
            raise _runtime_error(
                "canonical_result_intent_invalid",
                "Canonical Blind Ladder result intent is invalid",
            ) from None
        store.mark_selection_committed(battle_id)

    def record_terminal_result(
        self,
        winner: str | None,
        *,
        tied: bool,
    ) -> None:
        """Persist one authoritative terminal event before battle completion."""

        store = self._result_store
        battle_id = self._active_result_battle_id
        if store is None or battle_id is None or type(tied) is not bool:
            raise _runtime_error(
                "canonical_result_terminal_invalid",
                "Canonical Blind Ladder terminal result is invalid",
            ) from None
        if tied:
            if winner is not None:
                raise _runtime_error(
                    "canonical_result_terminal_invalid",
                    "Canonical Blind Ladder terminal result is invalid",
                ) from None
            outcome = TEAM_OUTCOME_TIE if self._team_mode else OUTCOME_TIE
        else:
            winner_id = normalize_showdown_identity(winner or "")
            pending = store.load().pending_result
            if pending is None or pending.battle_id != battle_id:
                completed = next(
                    (
                        record
                        for record in store.load().completed_results
                        if record.battle_id == battle_id
                    ),
                    None,
                )
                if completed is None:
                    raise _runtime_error(
                        "canonical_result_terminal_invalid",
                        "Canonical Blind Ladder terminal result is invalid",
                    ) from None
                player_id = (
                    completed.player_account_id
                    if self._team_mode
                    else completed.player_id
                )
            else:
                player_id = (
                    pending.player_account_id if self._team_mode else pending.player_id
                )
            if winner_id == player_id:
                outcome = TEAM_OUTCOME_A_WIN if self._team_mode else OUTCOME_PLAYER_WIN
            elif winner_id == self._bot_id:
                outcome = (
                    TEAM_OUTCOME_A_LOSS if self._team_mode else OUTCOME_PLAYER_LOSS
                )
            else:
                raise _runtime_error(
                    "canonical_result_winner_unexpected",
                    "Canonical Blind Ladder terminal winner is invalid",
                ) from None
        completed = store.finalize_terminal(battle_id, outcome)
        self._result_finalized = True
        rating_store = self._rating_store
        if rating_store is None:
            raise _runtime_error(
                "canonical_rating_store_invalid",
                "Canonical Blind Ladder rating store is unavailable",
            ) from None
        synchronized = rating_store.sync(store.require_ready())
        if self._team_mode:
            if not isinstance(
                completed, BlindTeamCompletedBattleResult
            ) or not isinstance(rating_store, BlindTeamRatingStateStore):
                raise _runtime_error(
                    "canonical_team_rating_update_missing",
                    "Canonical Blind Ladder team rating update is unavailable",
                ) from None
            if synchronized.last_update is not None:
                if not isinstance(synchronized.last_update, BlindTeamRatingUpdate):
                    raise _runtime_error(
                        "canonical_team_rating_update_missing",
                        "Canonical Blind Ladder team rating update is unavailable",
                    ) from None
                self._last_team_rating_update = self._public_team_update(
                    completed,
                    synchronized.state,
                    synchronized.last_update,
                )
            elif self._last_team_rating_update is None:
                raise _runtime_error(
                    "canonical_team_rating_update_missing",
                    "Canonical Blind Ladder team rating update is unavailable",
                ) from None
        elif synchronized.last_update is not None:
            self._last_rating_update = synchronized.last_update
        self._rating_synchronized = True

    def _public_team_update(
        self,
        completed,
        rating_state,
        update: BlindTeamRatingUpdate,
    ) -> BlindTeamPublicRatingUpdate:
        public_store = self._team_public_store
        if public_store is None:
            raise _runtime_error(
                "canonical_team_public_store_invalid",
                "Canonical Blind Ladder public registry is unavailable",
            ) from None
        public = public_store.load()
        player_identity = public.identity(completed.player_team_id)
        bot_identity = public.identity(completed.bot_team_id)
        player_rating: BlindTeamRating | None = rating_state.team(
            completed.player_team_id
        )
        bot_rating: BlindTeamRating | None = rating_state.team(completed.bot_team_id)
        if (
            player_identity is None
            or bot_identity is None
            or player_rating is None
            or bot_rating is None
        ):
            raise _runtime_error(
                "canonical_team_public_update_invalid",
                "Canonical Blind Ladder public rating update is unavailable",
            ) from None
        return BlindTeamPublicRatingUpdate(
            player_identity.display_name,
            update.team_a_rating_before,
            update.team_a_rating_after,
            update.team_a_delta,
            player_rating.wins,
            player_rating.losses,
            player_rating.ties,
            player_rating.win_rate_display,
            player_rating.peak_rating,
            player_rating.streak_label,
            bot_identity.display_name,
            bot_rating.rating,
            bot_rating.wins,
            bot_rating.losses,
            bot_rating.ties,
            bot_rating.win_rate_display,
        )

    async def run_once(self) -> Any:
        if self._running:
            raise _runtime_error(
                "canonical_runtime_already_active",
                "Canonical Blind Ladder runtime is already active",
            ) from None
        self._running = True
        if self._result_store is not None and self._active_result_battle_id is not None:
            self._running = False
            raise _runtime_error(
                "canonical_result_recovery_required",
                "Canonical Blind Ladder result recovery is required",
            ) from None
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
                "team_artifact_verification_failed",
                "team_submission_failed",
                "battle_initialization_failed",
            }
            if error.code in messages:
                message = {
                    "team_preparation_failed": "Blind Ladder team preparation failed",
                    "team_preparation_cleanup_failed": (
                        "Blind Ladder team preparation cleanup failed"
                    ),
                    "team_artifact_verification_failed": (
                        "Blind Ladder artifact verification failed"
                    ),
                    "team_submission_failed": "Blind Ladder team submission failed",
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
            if self._result_finalized and not self._rating_synchronized:
                raise _runtime_error(
                    "canonical_rating_sync_failed",
                    "Blind Ladder rating persistence requires recovery",
                ) from None
            error_type, code, message = lifecycle_failure
            raise error_type(code, message) from None
        if self._result_store is not None:
            if (
                self._active_result_battle_id is None
                or not self._result_finalized
                or not self._rating_synchronized
            ):
                if self._result_finalized:
                    raise _runtime_error(
                        "canonical_rating_sync_failed",
                        "Blind Ladder rating persistence requires recovery",
                    ) from None
                raise _runtime_error(
                    "canonical_result_terminal_missing",
                    "Canonical Blind Ladder terminal result was not persisted",
                ) from None
            self._active_result_battle_id = None
            self._result_finalized = False
            self._rating_synchronized = False
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
