"""Persistent shuffled-bag transactions, disconnected from battle runtime."""

from __future__ import annotations

import logging
import random
import re
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Callable, Protocol

from .canonical_models import CanonicalRuntimeRegistry
from .config import validate_blind_pool_state_config
from .errors import BlindPoolValidationError
from .locking import BlindPoolStateLock
from .models import (
    BlindChallengeToken,
    BlindPoolBagState,
    BlindPoolRegistry,
    BlindPoolReservation,
    BlindPoolStateConfig,
)
from .selection import (
    BlindPoolSelectionSnapshot,
    create_canonical_selection_snapshot,
    create_raw_selection_snapshot,
)
from .state import (
    ACCEPT_SENT_PHASE,
    RESERVATION_PHASE,
    STATE_SCHEMA_VERSION,
    load_blind_pool_bag_state,
    validate_blind_pool_bag_state,
    write_blind_pool_bag_state_atomic,
)


logger = logging.getLogger(__name__)
_RESERVATION_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class ShuffleSource(Protocol):
    def shuffle(self, values: list[str]) -> None: ...

    def randrange(self, start: int, stop: int | None = None) -> int: ...


class BlindPoolBagStore:
    """A persistent bag bound to one immutable validated selection snapshot.

    Raw-registry construction remains backward compatible. Callers that
    deliberately reload pool data must construct a new store;
    existing state is never reset automatically when registry semantics change.
    """

    def __init__(
        self,
        config: BlindPoolStateConfig,
        registry: BlindPoolRegistry,
        *,
        random_source: ShuffleSource | None = None,
        reservation_id_factory: Callable[[], str] | None = None,
        lock_timeout_seconds: float = 5.0,
    ) -> None:
        validated_config = validate_blind_pool_state_config(config)
        if not isinstance(registry, BlindPoolRegistry):
            raise BlindPoolValidationError(
                "registry_type_invalid",
                "Blind Ladder registry has an invalid type",
            ) from None
        selection = create_raw_selection_snapshot(
            registry,
            registry_path=validated_config.pool_config.registry_path,
        )
        self._initialize(
            validated_config,
            selection,
            random_source=random_source,
            reservation_id_factory=reservation_id_factory,
            lock_timeout_seconds=lock_timeout_seconds,
        )

    @classmethod
    def from_selection_snapshot(
        cls,
        config: BlindPoolStateConfig,
        selection: BlindPoolSelectionSnapshot,
        *,
        random_source: ShuffleSource | None = None,
        reservation_id_factory: Callable[[], str] | None = None,
        lock_timeout_seconds: float = 5.0,
    ) -> BlindPoolBagStore:
        """Construct explicitly from an already validated selection snapshot."""

        if not isinstance(selection, BlindPoolSelectionSnapshot):
            raise BlindPoolValidationError(
                "selection_type_invalid",
                "Blind Ladder selection snapshot has an invalid type",
            ) from None
        store = cls.__new__(cls)
        store._initialize(
            config,
            selection,
            random_source=random_source,
            reservation_id_factory=reservation_id_factory,
            lock_timeout_seconds=lock_timeout_seconds,
        )
        return store

    @classmethod
    def from_canonical_registry(
        cls,
        config: BlindPoolStateConfig,
        registry: CanonicalRuntimeRegistry,
        *,
        random_source: ShuffleSource | None = None,
        reservation_id_factory: Callable[[], str] | None = None,
        lock_timeout_seconds: float = 5.0,
    ) -> BlindPoolBagStore:
        """Construct explicitly from a committed canonical runtime registry."""

        return cls.from_selection_snapshot(
            config,
            create_canonical_selection_snapshot(registry),
            random_source=random_source,
            reservation_id_factory=reservation_id_factory,
            lock_timeout_seconds=lock_timeout_seconds,
        )

    def _initialize(
        self,
        config: BlindPoolStateConfig,
        selection: BlindPoolSelectionSnapshot,
        *,
        random_source: ShuffleSource | None,
        reservation_id_factory: Callable[[], str] | None,
        lock_timeout_seconds: float,
    ) -> None:
        self._config = validate_blind_pool_state_config(config)
        self._lock_path = self._resolve_lock_path(self._config)
        self._validate_artifact_collisions(
            self._config,
            self._lock_path,
            selection,
        )
        if len(selection.active_ids) < 2:
            raise BlindPoolValidationError(
                "insufficient_active_entries",
                "Blind Ladder shuffled bag requires at least two active entries",
            ) from None
        self._selection = selection
        self._fingerprint = selection.registry_fingerprint
        self._active_ids = selection.active_ids
        self._random = random.SystemRandom() if random_source is None else random_source
        self._reservation_id_factory = reservation_id_factory or (
            lambda: uuid.uuid4().hex
        )
        self._lock_timeout_seconds = lock_timeout_seconds

    def __repr__(self) -> str:
        return "BlindPoolBagStore(active_count={!r})".format(len(self._active_ids))

    @staticmethod
    def _resolve_lock_path(config: BlindPoolStateConfig) -> Path:
        try:
            return config.lock_path.resolve(strict=False)
        except (OSError, RuntimeError):
            raise BlindPoolValidationError(
                "state_lock_path_invalid",
                "Blind Ladder state lock path is invalid",
            ) from None

    @staticmethod
    def _validate_artifact_collisions(
        config: BlindPoolStateConfig,
        lock_path: Path,
        selection: BlindPoolSelectionSnapshot,
    ) -> None:
        selection._validate_collision_paths(
            configured_registry_path=config.pool_config.registry_path,
            state_path=config.state_path,
            lock_path=lock_path,
        )

    def _lock(self) -> BlindPoolStateLock:
        config = validate_blind_pool_state_config(self._config)
        if config.state_path != self._config.state_path:
            raise BlindPoolValidationError(
                "state_path_changed",
                "Blind Ladder state path changed after configuration",
            ) from None
        lock_path = self._resolve_lock_path(config)
        if lock_path != self._lock_path:
            raise BlindPoolValidationError(
                "state_lock_path_changed",
                "Blind Ladder state lock path changed after configuration",
            ) from None
        self._validate_artifact_collisions(config, lock_path, self._selection)
        return BlindPoolStateLock(
            lock_path,
            timeout_seconds=self._lock_timeout_seconds,
        )

    def _new_cycle_order(self, last_consumed_id: str | None = None) -> tuple[str, ...]:
        order = list(self._active_ids)
        self._random.shuffle(order)
        if last_consumed_id is not None and order[0] == last_consumed_id:
            replacement_position = self._random.randrange(1, len(order))
            order[0], order[replacement_position] = (
                order[replacement_position],
                order[0],
            )
        return tuple(order)

    def _initial_state(self) -> BlindPoolBagState:
        state = BlindPoolBagState(
            schema_version=STATE_SCHEMA_VERSION,
            registry_fingerprint=self._fingerprint,
            cycle_number=1,
            cycle_order=self._new_cycle_order(),
            next_index=0,
            last_consumed_id=None,
            reservation=None,
        )
        return validate_blind_pool_bag_state(
            {
                "schema_version": state.schema_version,
                "registry_fingerprint": state.registry_fingerprint,
                "cycle_number": state.cycle_number,
                "cycle_order": list(state.cycle_order),
                "next_index": state.next_index,
                "last_consumed_id": state.last_consumed_id,
                "reservation": None,
            },
            self._selection,
        )

    def initialize_or_load(self) -> BlindPoolBagState:
        """Create cycle one once, or return the latest strictly validated state."""

        with self._lock():
            if self._config.state_path.exists():
                return load_blind_pool_bag_state(self._config, self._selection)
            state = self._initial_state()
            write_blind_pool_bag_state_atomic(self._config, state, self._selection)
            logger.info(
                "Initialized Blind Ladder bag active={} cycle={} "
                "next_position={}".format(
                    len(self._active_ids), state.cycle_number, state.next_index
                )
            )
            return state

    def snapshot(self) -> BlindPoolBagState:
        """Return the latest state, including any unresolved reservation."""

        with self._lock():
            return load_blind_pool_bag_state(self._config, self._selection)

    def reserve_next(
        self,
        challenge_token: BlindChallengeToken | None = None,
    ) -> BlindPoolReservation:
        """Persist a reservation for the next unconsumed cycle position."""

        if challenge_token is not None and not isinstance(
            challenge_token, BlindChallengeToken
        ):
            raise BlindPoolValidationError(
                "challenge_token_invalid",
                "Blind Ladder challenge token is malformed",
            ) from None

        with self._lock():
            state = load_blind_pool_bag_state(self._config, self._selection)
            if state.reservation is not None:
                raise BlindPoolValidationError(
                    "unresolved_reservation_exists",
                    "Blind Ladder bag already has an unresolved reservation",
                ) from None
            if state.next_index == len(state.cycle_order):
                state = BlindPoolBagState(
                    schema_version=STATE_SCHEMA_VERSION,
                    registry_fingerprint=self._fingerprint,
                    cycle_number=state.cycle_number + 1,
                    cycle_order=self._new_cycle_order(state.last_consumed_id),
                    next_index=0,
                    last_consumed_id=state.last_consumed_id,
                    reservation=None,
                )
            try:
                reservation_id = self._reservation_id_factory()
            except Exception:
                raise BlindPoolValidationError(
                    "reservation_id_generation_failed",
                    "Blind Ladder reservation ID generation failed",
                ) from None
            if (
                not isinstance(reservation_id, str)
                or _RESERVATION_ID_PATTERN.fullmatch(reservation_id) is None
            ):
                raise BlindPoolValidationError(
                    "reservation_id_generation_failed",
                    "Blind Ladder reservation ID generation failed",
                ) from None
            reservation = BlindPoolReservation(
                reservation_id=reservation_id,
                team_id=state.cycle_order[state.next_index],
                cycle_number=state.cycle_number,
                position=state.next_index,
                phase=RESERVATION_PHASE,
                challenge_token=challenge_token,
            )
            updated = replace(state, reservation=reservation)
            write_blind_pool_bag_state_atomic(self._config, updated, self._selection)
            logger.info(
                "Reserved Blind Ladder team_id={} cycle={} position={}".format(
                    reservation.team_id,
                    reservation.cycle_number,
                    reservation.position,
                )
            )
            return reservation

    def _active_reservation(
        self,
        state: BlindPoolBagState,
        reservation_id: str,
        *,
        required_phase: str | None = None,
    ) -> BlindPoolReservation:
        if (
            not isinstance(reservation_id, str)
            or _RESERVATION_ID_PATTERN.fullmatch(reservation_id) is None
        ):
            raise BlindPoolValidationError(
                "reservation_id_invalid",
                "Blind Ladder reservation ID is malformed",
            ) from None
        if state.reservation is None:
            raise BlindPoolValidationError(
                "reservation_not_found",
                "Blind Ladder bag has no active reservation",
            ) from None
        if state.reservation.reservation_id != reservation_id:
            raise BlindPoolValidationError(
                "reservation_identity_mismatch",
                "Blind Ladder reservation identity does not match",
            ) from None
        if required_phase is not None and state.reservation.phase != required_phase:
            raise BlindPoolValidationError(
                "reservation_phase_transition_invalid",
                "Blind Ladder reservation phase does not permit this transition",
            ) from None
        return state.reservation

    def mark_accept_sent(
        self,
        reservation_id: str,
        challenge_token: BlindChallengeToken | None = None,
    ) -> BlindPoolBagState:
        """Persist that acceptance may now have been transmitted.

        This write-ahead marker is deliberately durable before a caller attempts
        the websocket send. It does not claim that transmission succeeded.
        """

        with self._lock():
            state = load_blind_pool_bag_state(self._config, self._selection)
            reservation = self._active_reservation(
                state,
                reservation_id,
                required_phase=RESERVATION_PHASE,
            )
            if challenge_token is not None and (
                reservation.challenge_token is None
                or not reservation.challenge_token.matches(challenge_token)
            ):
                raise BlindPoolValidationError(
                    "reservation_challenge_token_mismatch",
                    "Blind Ladder reservation challenge identity does not match",
                ) from None
            updated = replace(
                state,
                reservation=replace(reservation, phase=ACCEPT_SENT_PHASE),
            )
            write_blind_pool_bag_state_atomic(self._config, updated, self._selection)
            logger.info(
                "Marked Blind Ladder acceptance pending team_id={} cycle={} "
                "position={}".format(
                    reservation.team_id,
                    reservation.cycle_number,
                    reservation.position,
                )
            )
            return updated

    def _commit_reservation_in_phase(
        self,
        reservation_id: str,
        required_phase: str,
    ) -> BlindPoolBagState:
        with self._lock():
            state = load_blind_pool_bag_state(self._config, self._selection)
            reservation = self._active_reservation(
                state,
                reservation_id,
                required_phase=required_phase,
            )
            updated = replace(
                state,
                next_index=state.next_index + 1,
                last_consumed_id=reservation.team_id,
                reservation=None,
            )
            write_blind_pool_bag_state_atomic(self._config, updated, self._selection)
            logger.info(
                "Committed Blind Ladder team_id={} cycle={} position={}".format(
                    reservation.team_id,
                    reservation.cycle_number,
                    reservation.position,
                )
            )
            return updated

    def commit_reservation(self, reservation_id: str) -> BlindPoolBagState:
        """Consume exactly the reservation identified by the caller."""

        return self._commit_reservation_in_phase(
            reservation_id,
            RESERVATION_PHASE,
        )

    def commit_room_created(self, reservation_id: str) -> BlindPoolBagState:
        """Consume one write-ahead reservation after exact room correlation."""

        return self._commit_reservation_in_phase(
            reservation_id,
            ACCEPT_SENT_PHASE,
        )

    def reconcile_accept_sent_as_room_created(
        self,
        reservation_id: str,
    ) -> BlindPoolBagState:
        """Explicitly consume an ambiguously accepted reservation."""

        return self._commit_reservation_in_phase(
            reservation_id,
            ACCEPT_SENT_PHASE,
        )

    def release_reservation(self, reservation_id: str) -> BlindPoolBagState:
        """Clear exactly one reservation without consuming its position."""

        with self._lock():
            state = load_blind_pool_bag_state(self._config, self._selection)
            reservation = self._active_reservation(
                state,
                reservation_id,
                required_phase=RESERVATION_PHASE,
            )
            updated = replace(state, reservation=None)
            write_blind_pool_bag_state_atomic(self._config, updated, self._selection)
            logger.info(
                "Released Blind Ladder team_id={} cycle={} position={}".format(
                    reservation.team_id,
                    reservation.cycle_number,
                    reservation.position,
                )
            )
            return updated

    def reconcile_accept_sent_as_no_room(
        self,
        reservation_id: str,
    ) -> BlindPoolBagState:
        """Explicitly release an ambiguously accepted reservation without use."""

        with self._lock():
            state = load_blind_pool_bag_state(self._config, self._selection)
            reservation = self._active_reservation(
                state,
                reservation_id,
                required_phase=ACCEPT_SENT_PHASE,
            )
            updated = replace(state, reservation=None)
            write_blind_pool_bag_state_atomic(self._config, updated, self._selection)
            logger.info(
                "Reconciled Blind Ladder no-room outcome team_id={} cycle={} "
                "position={}".format(
                    reservation.team_id,
                    reservation.cycle_number,
                    reservation.position,
                )
            )
            return updated
