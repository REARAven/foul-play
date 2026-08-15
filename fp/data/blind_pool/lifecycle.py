"""Synthetic-safe Blind Ladder challenge and room lifecycle coordination.

This module deliberately has no team-file reader and is not wired into the
production entry point. A later phase can supply canonical team preparation
and battle initialization through the narrow protocols defined here.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from .bag import BlindPoolBagStore
from .errors import (
    BlindPoolLifecycleError,
    BlindPoolReconciliationRequired,
    BlindPoolValidationError,
)
from .models import (
    BlindPoolBagState,
    BlindPoolBattleRoom,
    BlindPoolChallenge,
    BlindPoolReservation,
)
from .state import ACCEPT_SENT_PHASE, RESERVATION_PHASE


logger = logging.getLogger(__name__)

BLIND_LADDER_FORMAT = "gen9tugs"
BLIND_LADDER_MODE = "accept_challenge"
_CHALLENGE_SOURCE = "pm"
_ROOM_ID_PATTERN = re.compile(
    r"^battle-(?P<format_id>[a-z0-9]+)-(?P<number>[1-9][0-9]*)$"
)
_MAX_ROOM_MESSAGES = 16
_MAX_ROOM_EVENTS = 256
_MAX_ROOM_CHARACTERS = 262_144
_GLOBAL_EVENT_PREFIXES = (
    "|challstr|",
    "|formats|",
    "|nametaken|",
    "|pm|",
    "|queryresponse|",
    "|updateuser|",
)


def normalize_showdown_identity(value: str) -> str:
    """Apply the client's established alphanumeric Showdown ID normalization."""

    if not isinstance(value, str):
        return ""
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def parse_incoming_challenge(
    message: str,
    *,
    bot_username: str,
    required_format: str,
) -> BlindPoolChallenge | None:
    """Parse the two exact PM challenge encodings used by this deployment.

    The server grammar has no durable challenge identifier. Deduplication must
    therefore remain process-local and use challenger plus exact format. The
    legacy encoding is retained for the existing client fixture; the current
    server includes the exact format in both its command and builder fields.
    """

    if not isinstance(message, str):
        return None
    bot_id = normalize_showdown_identity(bot_username)
    if not bot_id or not isinstance(required_format, str):
        return None
    for line in message.splitlines():
        fields = line.split("|")
        legacy_grammar = (
            len(fields) == 9
            and fields[4] == "/challenge"
            and fields[5] == required_format
            and fields[6:] == ["", "", ""]
        )
        current_server_grammar = (
            len(fields) == 9
            and fields[4] == "/challenge " + required_format
            and fields[5] == required_format
            and fields[6:] == ["", "", ""]
        )
        if fields[:2] != ["", "pm"] or not (legacy_grammar or current_server_grammar):
            continue
        challenger_name = fields[2].strip()
        challenger_id = normalize_showdown_identity(challenger_name)
        recipient_id = normalize_showdown_identity(fields[3])
        if not challenger_id or recipient_id != bot_id:
            continue
        return BlindPoolChallenge(
            challenger_id=challenger_id,
            challenger_name=challenger_name,
            format_id=required_format,
            source=_CHALLENGE_SOURCE,
        )
    return None


class BlindChallengeTransport(Protocol):
    username: str

    async def receive_message(self) -> str: ...

    async def send_challenge_acceptance(
        self,
        challenge: BlindPoolChallenge,
    ) -> None: ...

    def install_blind_room_handoff(
        self,
        room: BlindPoolBattleRoom,
        event_lines: tuple[str, ...],
    ) -> None: ...


class BlindTeamPreparer(Protocol):
    def __call__(self, team_id: str) -> Awaitable[None]: ...


class BlindBattleInitializer(Protocol):
    def __call__(self, room: BlindPoolBattleRoom) -> Awaitable[Any]: ...


@dataclass
class _RoomCandidate:
    room_id: str
    format_id: str
    initialized: bool
    players: dict[str, tuple[str, str]]
    event_lines: list[str]
    message_count: int = 0
    character_count: int = 0
    rejected: bool = False


def _room_frames(message: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Split one websocket payload into structurally valid room sections."""

    if not isinstance(message, str):
        return ()
    frames: list[tuple[str, tuple[str, ...]]] = []
    room_id: str | None = None
    events: list[str] = []

    def finish() -> None:
        if room_id is not None and _ROOM_ID_PATTERN.fullmatch(room_id) is not None:
            frames.append((room_id, tuple(events)))

    for line in message.splitlines():
        if line.startswith(">"):
            finish()
            header = line[1:]
            events = []
            if "|" in header:
                header, first_event = header.split("|", 1)
                events.append("|" + first_event)
            room_id = header
        elif room_id is not None:
            events.append(line)
    finish()
    return tuple(frames)


def room_ids_in_message(message: str) -> tuple[str, ...]:
    """Return only structurally valid room IDs, without retaining payloads."""

    return tuple(room_id for room_id, _events in _room_frames(message))


class BlindRoomCorrelator:
    """Bounded exact correlation state for multiplexed websocket room traffic."""

    def __init__(
        self,
        *,
        challenge: BlindPoolChallenge,
        bot_username: str,
        excluded_room_ids: tuple[str, ...] = (),
        max_candidates: int = 8,
    ) -> None:
        if (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int)
            or max_candidates < 1
        ):
            raise BlindPoolLifecycleError(
                "room_correlation_configuration_invalid",
                "Blind Ladder room correlation capacity must be positive",
            ) from None
        self._challenge = challenge
        self._bot_id = normalize_showdown_identity(bot_username)
        self._excluded = frozenset(excluded_room_ids)
        self._max_candidates = max_candidates
        self._candidates: OrderedDict[str, _RoomCandidate] = OrderedDict()
        self._failed_closed = False

    def __repr__(self) -> str:
        return "BlindRoomCorrelator(candidate_count={!r}, failed_closed={!r})".format(
            len(self._candidates),
            self._failed_closed,
        )

    @property
    def candidate_count(self) -> int:
        return len(self._candidates)

    def event_lines_for(self, room_id: str) -> tuple[str, ...]:
        candidate = self._candidates.get(room_id)
        if candidate is None or candidate.rejected:
            raise BlindPoolLifecycleError(
                "room_handoff_unavailable",
                "Blind Ladder room handoff is unavailable",
            ) from None
        return tuple(candidate.event_lines)

    def feed(self, message: str) -> BlindPoolBattleRoom | None:
        if self._failed_closed:
            return None
        for room_id, events in _room_frames(message):
            if room_id in self._excluded:
                continue
            room_match = _ROOM_ID_PATTERN.fullmatch(room_id)
            assert room_match is not None
            format_id = room_match.group("format_id")
            if format_id != self._challenge.format_id:
                continue

            candidate = self._candidates.get(room_id)
            if candidate is None:
                if len(self._candidates) >= self._max_candidates:
                    self._failed_closed = True
                    return None
                candidate = _RoomCandidate(room_id, format_id, False, {}, [])
                self._candidates[room_id] = candidate
            else:
                self._candidates.move_to_end(room_id)
            if candidate.rejected:
                continue

            candidate.message_count += 1
            candidate.character_count += sum(len(event) for event in events)
            if (
                candidate.message_count > _MAX_ROOM_MESSAGES
                or len(candidate.event_lines) + len(events) > _MAX_ROOM_EVENTS
                or candidate.character_count > _MAX_ROOM_CHARACTERS
            ):
                candidate.rejected = True
                self._failed_closed = True
                candidate.event_lines.clear()
                candidate.players.clear()
                continue
            room_events = tuple(
                event
                for event in events
                if not event.startswith(_GLOBAL_EVENT_PREFIXES)
            )
            candidate.event_lines.extend(room_events)

            for event in room_events:
                fields = event.split("|")
                if len(fields) >= 3 and fields[0] == "" and fields[1] == "init":
                    if fields[2] == "battle":
                        candidate.initialized = True
                elif (
                    len(fields) >= 4
                    and fields[0] == ""
                    and fields[1] == "player"
                    and fields[2] in ("p1", "p2")
                ):
                    slot = fields[2]
                    display_name = fields[3].strip()
                    player_id = normalize_showdown_identity(display_name)
                    previous = candidate.players.get(slot)
                    if not player_id:
                        if previous is not None:
                            candidate.rejected = True
                        continue
                    if previous is not None and previous[0] != player_id:
                        candidate.rejected = True
                        continue
                    candidate.players[slot] = (player_id, display_name)

            if candidate.rejected:
                candidate.event_lines.clear()
                candidate.players.clear()
                continue
            if not candidate.initialized or set(candidate.players) != {"p1", "p2"}:
                continue
            bot_slots = [
                slot
                for slot, (player_id, _name) in candidate.players.items()
                if player_id == self._bot_id
            ]
            opponent_slots = [
                slot
                for slot, (player_id, _name) in candidate.players.items()
                if player_id == self._challenge.challenger_id
            ]
            if len(bot_slots) != 1 or len(opponent_slots) != 1:
                continue
            if bot_slots[0] == opponent_slots[0]:
                continue
            opponent_id, opponent_name = candidate.players[opponent_slots[0]]
            room = BlindPoolBattleRoom(
                room_id=room_id,
                format_id=format_id,
                opponent_id=opponent_id,
                opponent_name=opponent_name,
                bot_slot=bot_slots[0],
                opponent_slot=opponent_slots[0],
            )
            return room
        return None


class BlindPoolLifecycleCoordinator:
    """One-at-a-time synthetic-safe challenge lifecycle coordinator."""

    def __init__(
        self,
        store: BlindPoolBagStore,
        transport: BlindChallengeTransport,
        prepare_team: BlindTeamPreparer,
        *,
        format_id: str = BLIND_LADDER_FORMAT,
        mode: str = BLIND_LADDER_MODE,
        initialize_battle: BlindBattleInitializer | None = None,
        room_timeout_seconds: float = 30.0,
        monotonic: Callable[[], float] = time.monotonic,
        max_room_candidates: int = 8,
    ) -> None:
        if format_id != BLIND_LADDER_FORMAT or mode != BLIND_LADDER_MODE:
            raise BlindPoolLifecycleError(
                "lifecycle_ineligible",
                "Blind Ladder lifecycle requires exact gen9tugs acceptance mode",
            ) from None
        if (
            isinstance(room_timeout_seconds, bool)
            or not isinstance(room_timeout_seconds, (int, float))
            or not math.isfinite(room_timeout_seconds)
            or room_timeout_seconds <= 0
        ):
            raise BlindPoolLifecycleError(
                "room_timeout_invalid",
                "Blind Ladder room timeout must be finite and positive",
            ) from None
        if (
            isinstance(max_room_candidates, bool)
            or not isinstance(max_room_candidates, int)
            or max_room_candidates < 1
        ):
            raise BlindPoolLifecycleError(
                "room_correlation_configuration_invalid",
                "Blind Ladder room correlation capacity must be positive",
            ) from None
        bot_id = normalize_showdown_identity(getattr(transport, "username", ""))
        if not bot_id:
            raise BlindPoolLifecycleError(
                "transport_identity_invalid",
                "Blind Ladder transport identity is invalid",
            ) from None
        if not callable(getattr(transport, "install_blind_room_handoff", None)):
            raise BlindPoolLifecycleError(
                "transport_handoff_invalid",
                "Blind Ladder transport does not support bounded room handoff",
            ) from None
        self._store = store
        self._execution_guard = vars(store).setdefault(
            "_blind_lifecycle_execution_guard",
            asyncio.Lock(),
        )
        self._owns_execution_guard = False
        self._transport = transport
        self._prepare_team = prepare_team
        self._format_id = format_id
        self._mode = mode
        self._initialize_battle = initialize_battle
        self._room_timeout = float(room_timeout_seconds)
        self._monotonic = monotonic
        self._max_room_candidates = max_room_candidates
        self._active_challenge_identity: tuple[str, str] | None = None
        self._active_challenge: BlindPoolChallenge | None = None
        self._startup_checked = False
        self._reconciliation_pending = False
        self._run_in_progress = False
        self._task_executing = False
        self._execution_task: asyncio.Task[Any] | None = None
        self._resolved_room_ids: deque[str] = deque(maxlen=32)
        self._preaccept_room_ids: deque[str] = deque(maxlen=32)

    def __repr__(self) -> str:
        return (
            "BlindPoolLifecycleCoordinator(format_id={!r}, mode={!r}, active={!r})"
        ).format(
            self._format_id,
            self._mode,
            self._active_challenge_identity is not None,
        )

    async def startup(self) -> None:
        """Recover only unambiguous pre-accept state; quarantine accept_sent."""

        current_task = asyncio.current_task()
        if self._execution_guard.locked() and (
            not self._owns_execution_guard or current_task is not self._execution_task
        ):
            raise BlindPoolLifecycleError(
                "lifecycle_already_active",
                "Blind Ladder lifecycle is already active",
            ) from None
        state = self._store.initialize_or_load()
        reservation = state.reservation
        if reservation is None:
            self._startup_checked = True
            return
        if reservation.phase == RESERVATION_PHASE:
            before = state
            try:
                self._store.release_reservation(reservation.reservation_id)
            except (BlindPoolValidationError, asyncio.CancelledError) as error:
                after = self._authoritative_snapshot()
                if not self._state_proves_release(before, after, reservation):
                    if self._reservation_is(
                        after.reservation, reservation, ACCEPT_SENT_PHASE
                    ):
                        self._quarantine()
                        raise BlindPoolReconciliationRequired(
                            "reconciliation_required",
                            "Blind Ladder acceptance state requires explicit reconciliation",
                        ) from None
                    if isinstance(error, asyncio.CancelledError):
                        raise
                    raise BlindPoolLifecycleError(
                        "startup_release_failed",
                        "Blind Ladder startup could not prove reservation release",
                    ) from None
                if isinstance(error, asyncio.CancelledError):
                    raise
            logger.info(
                "Released pre-accept Blind Ladder reservation during startup "
                "team_id={} cycle={} position={}".format(
                    reservation.team_id,
                    reservation.cycle_number,
                    reservation.position,
                )
            )
            self._active_challenge_identity = None
            self._active_challenge = None
            self._startup_checked = True
            return
        self._quarantine()
        raise BlindPoolReconciliationRequired(
            "reconciliation_required",
            "Blind Ladder acceptance state requires explicit reconciliation",
        ) from None

    def reconcile_as_room_created(self, reservation_id: str) -> BlindPoolBagState:
        if self._task_executing:
            raise BlindPoolLifecycleError(
                "lifecycle_already_active",
                "Blind Ladder lifecycle is already active",
            ) from None
        if self._execution_guard.locked() and not self._owns_execution_guard:
            raise BlindPoolLifecycleError(
                "lifecycle_already_active",
                "Blind Ladder lifecycle is already active",
            ) from None
        before = self._store.snapshot()
        reservation = before.reservation
        cancellation: asyncio.CancelledError | None = None
        try:
            state = self._store.reconcile_accept_sent_as_room_created(reservation_id)
        except (BlindPoolValidationError, asyncio.CancelledError) as error:
            after = self._authoritative_snapshot()
            if reservation is not None and self._state_proves_commit(
                after, reservation
            ):
                state = after
                if isinstance(error, asyncio.CancelledError):
                    cancellation = error
            elif after == before:
                if isinstance(error, asyncio.CancelledError):
                    raise
                if isinstance(error, BlindPoolValidationError):
                    raise error
                raise BlindPoolLifecycleError(
                    "reconciliation_failed",
                    "Blind Ladder reconciliation failed before changing state",
                ) from None
            else:
                self._quarantine()
                if isinstance(error, asyncio.CancelledError):
                    raise
                raise BlindPoolLifecycleError(
                    "reconciliation_outcome_ambiguous",
                    "Blind Ladder reconciliation outcome cannot be proven",
                ) from None
        self._active_challenge_identity = None
        self._active_challenge = None
        self._startup_checked = True
        self._reconciliation_pending = False
        self._run_in_progress = False
        self._release_execution_guard()
        logger.info("Explicit Blind Ladder room-created reconciliation completed")
        if cancellation is not None:
            raise cancellation
        return state

    def reconcile_as_no_room(self, reservation_id: str) -> BlindPoolBagState:
        if self._task_executing:
            raise BlindPoolLifecycleError(
                "lifecycle_already_active",
                "Blind Ladder lifecycle is already active",
            ) from None
        if self._execution_guard.locked() and not self._owns_execution_guard:
            raise BlindPoolLifecycleError(
                "lifecycle_already_active",
                "Blind Ladder lifecycle is already active",
            ) from None
        before = self._store.snapshot()
        reservation = before.reservation
        cancellation: asyncio.CancelledError | None = None
        try:
            state = self._store.reconcile_accept_sent_as_no_room(reservation_id)
        except (BlindPoolValidationError, asyncio.CancelledError) as error:
            after = self._authoritative_snapshot()
            if reservation is not None and self._state_proves_release(
                before, after, reservation
            ):
                state = after
                if isinstance(error, asyncio.CancelledError):
                    cancellation = error
            elif after == before:
                if isinstance(error, asyncio.CancelledError):
                    raise
                if isinstance(error, BlindPoolValidationError):
                    raise error
                raise BlindPoolLifecycleError(
                    "reconciliation_failed",
                    "Blind Ladder reconciliation failed before changing state",
                ) from None
            else:
                self._quarantine()
                if isinstance(error, asyncio.CancelledError):
                    raise
                raise BlindPoolLifecycleError(
                    "reconciliation_outcome_ambiguous",
                    "Blind Ladder reconciliation outcome cannot be proven",
                ) from None
        self._active_challenge_identity = None
        self._active_challenge = None
        self._startup_checked = True
        self._reconciliation_pending = False
        self._run_in_progress = False
        self._release_execution_guard()
        logger.info("Explicit Blind Ladder no-room reconciliation completed")
        if cancellation is not None:
            raise cancellation
        return state

    async def _wait_for_challenge(self) -> BlindPoolChallenge:
        while True:
            message = await self._transport.receive_message()
            for room_id in room_ids_in_message(message):
                if room_id not in self._preaccept_room_ids:
                    self._preaccept_room_ids.append(room_id)
            challenge = parse_incoming_challenge(
                message,
                bot_username=self._transport.username,
                required_format=self._format_id,
            )
            if challenge is None:
                continue
            if challenge.deduplication_identity == self._active_challenge_identity:
                logger.info("Ignored duplicate active Blind Ladder challenge")
                continue
            if self._active_challenge_identity is not None:
                logger.info("Ignored Blind Ladder challenge while lifecycle is active")
                continue
            self._active_challenge_identity = challenge.deduplication_identity
            self._active_challenge = challenge
            logger.info("Received matching Blind Ladder challenge")
            return challenge

    def _clear_active_challenge(self) -> None:
        self._active_challenge_identity = None
        self._active_challenge = None

    def _quarantine(self) -> None:
        self._reconciliation_pending = True

    def _release_execution_guard(self) -> None:
        if self._owns_execution_guard:
            self._execution_guard.release()
            self._owns_execution_guard = False
        self._execution_task = None

    def _authoritative_snapshot(self) -> BlindPoolBagState:
        try:
            return self._store.snapshot()
        except BlindPoolValidationError:
            self._quarantine()
            raise BlindPoolLifecycleError(
                "state_outcome_unavailable",
                "Blind Ladder durable state outcome is unavailable",
            ) from None

    @staticmethod
    def _reservation_is(
        actual: BlindPoolReservation | None,
        expected: BlindPoolReservation,
        phase: str,
    ) -> bool:
        return (
            actual is not None
            and actual.reservation_id == expected.reservation_id
            and actual.team_id == expected.team_id
            and actual.cycle_number == expected.cycle_number
            and actual.position == expected.position
            and actual.phase == phase
        )

    @classmethod
    def _state_proves_commit(
        cls,
        state: BlindPoolBagState,
        reservation: BlindPoolReservation,
    ) -> bool:
        return (
            state.reservation is None
            and state.cycle_number == reservation.cycle_number
            and state.next_index == reservation.position + 1
            and state.cycle_order[reservation.position] == reservation.team_id
            and state.last_consumed_id == reservation.team_id
        )

    @classmethod
    def _state_proves_release(
        cls,
        before: BlindPoolBagState,
        after: BlindPoolBagState,
        reservation: BlindPoolReservation,
    ) -> bool:
        return (
            after.reservation is None
            and after.schema_version == before.schema_version
            and after.registry_fingerprint == before.registry_fingerprint
            and after.cycle_number == before.cycle_number
            and after.cycle_order == before.cycle_order
            and after.next_index == before.next_index == reservation.position
            and after.last_consumed_id == before.last_consumed_id
        )

    def _release_pre_accept(self, reservation: BlindPoolReservation) -> None:
        before = self._authoritative_snapshot()
        if before.reservation is None:
            self._clear_active_challenge()
            return
        if not self._reservation_is(before.reservation, reservation, RESERVATION_PHASE):
            if self._reservation_is(before.reservation, reservation, ACCEPT_SENT_PHASE):
                self._quarantine()
                raise BlindPoolReconciliationRequired(
                    "reconciliation_required",
                    "Blind Ladder acceptance state requires explicit reconciliation",
                ) from None
            raise BlindPoolLifecycleError(
                "pre_accept_release_failed",
                "Blind Ladder pre-accept reservation could not be released safely",
            ) from None
        try:
            self._store.release_reservation(reservation.reservation_id)
        except (BlindPoolValidationError, asyncio.CancelledError) as error:
            after = self._authoritative_snapshot()
            if self._state_proves_release(before, after, reservation):
                self._clear_active_challenge()
                if isinstance(error, asyncio.CancelledError):
                    raise
                return
            if self._reservation_is(after.reservation, reservation, ACCEPT_SENT_PHASE):
                self._quarantine()
                raise BlindPoolReconciliationRequired(
                    "reconciliation_required",
                    "Blind Ladder acceptance state requires explicit reconciliation",
                ) from None
            if self._reservation_is(after.reservation, reservation, RESERVATION_PHASE):
                self._clear_active_challenge()
                self._startup_checked = False
            if isinstance(error, asyncio.CancelledError):
                raise
            raise BlindPoolLifecycleError(
                "pre_accept_release_failed",
                "Blind Ladder pre-accept reservation could not be released safely",
            ) from None
        self._clear_active_challenge()

    def _remember_resolved_room(self, room_id: str) -> None:
        if room_id in self._resolved_room_ids:
            self._resolved_room_ids.remove(room_id)
        self._resolved_room_ids.append(room_id)

    async def _wait_for_matching_room(
        self,
        challenge: BlindPoolChallenge,
    ) -> tuple[BlindPoolBattleRoom, tuple[str, ...]]:
        correlator = BlindRoomCorrelator(
            challenge=challenge,
            bot_username=self._transport.username,
            excluded_room_ids=tuple(
                dict.fromkeys((*self._resolved_room_ids, *self._preaccept_room_ids))
            ),
            max_candidates=self._max_room_candidates,
        )
        deadline = self._monotonic() + self._room_timeout
        # The protocol has neither an authoritative correlated cancellation
        # event nor a durable challenge ID. Rooms observed before acceptance
        # are excluded, but an older same-opponent room already queued behind
        # the challenge frame cannot be distinguished from a new room. Such a
        # deployment requires a stronger transport observation boundary in a
        # later activation phase. Silence and errors always remain ambiguous.
        while True:
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                raise BlindPoolReconciliationRequired(
                    "matching_room_timed_out",
                    "Blind Ladder room wait timed out; reconciliation is required",
                ) from None
            try:
                message = await asyncio.wait_for(
                    self._transport.receive_message(),
                    timeout=remaining,
                )
            except asyncio.CancelledError:
                raise
            except TimeoutError:
                raise BlindPoolReconciliationRequired(
                    "matching_room_timed_out",
                    "Blind Ladder room wait timed out; reconciliation is required",
                ) from None
            except Exception:
                raise BlindPoolReconciliationRequired(
                    "room_correlation_failed",
                    "Blind Ladder room correlation is ambiguous",
                ) from None

            duplicate = parse_incoming_challenge(
                message,
                bot_username=self._transport.username,
                required_format=self._format_id,
            )
            if duplicate is not None:
                if duplicate.deduplication_identity == challenge.deduplication_identity:
                    logger.info("Ignored duplicate active Blind Ladder challenge")
                else:
                    logger.info(
                        "Ignored Blind Ladder challenge while lifecycle is active"
                    )
            room = correlator.feed(message)
            if room is not None:
                return room, correlator.event_lines_for(room.room_id)

    async def run_once(self) -> Any:
        """Resolve one synthetic lifecycle and initialize only after commit."""

        if self._run_in_progress:
            if self._reconciliation_pending:
                raise BlindPoolReconciliationRequired(
                    "reconciliation_required",
                    "Blind Ladder acceptance state requires explicit reconciliation",
                ) from None
            raise BlindPoolLifecycleError(
                "lifecycle_already_active",
                "Blind Ladder lifecycle is already active",
            ) from None
        self._run_in_progress = True
        if self._execution_guard.locked():
            self._run_in_progress = False
            raise BlindPoolLifecycleError(
                "lifecycle_already_active",
                "Blind Ladder lifecycle is already active",
            ) from None
        try:
            await self._execution_guard.acquire()
        except BaseException:
            self._run_in_progress = False
            raise
        self._owns_execution_guard = True
        self._task_executing = True
        self._execution_task = asyncio.current_task()
        try:
            if self._reconciliation_pending:
                raise BlindPoolReconciliationRequired(
                    "reconciliation_required",
                    "Blind Ladder acceptance state requires explicit reconciliation",
                ) from None
            if not self._startup_checked:
                await self.startup()
            challenge = await self._wait_for_challenge()
            try:
                reservation = self._store.reserve_next()
            except (BlindPoolValidationError, asyncio.CancelledError) as error:
                after = self._authoritative_snapshot()
                if after.reservation is None:
                    self._clear_active_challenge()
                    if isinstance(error, asyncio.CancelledError):
                        raise
                    raise BlindPoolLifecycleError(
                        "reservation_failed",
                        "Blind Ladder reservation failed before selection",
                    ) from None
                elif after.reservation.phase == ACCEPT_SENT_PHASE:
                    self._quarantine()
                    if isinstance(error, asyncio.CancelledError):
                        raise
                    raise BlindPoolReconciliationRequired(
                        "reconciliation_required",
                        "Blind Ladder acceptance state requires explicit reconciliation",
                    ) from None
                else:
                    self._clear_active_challenge()
                    self._startup_checked = False
                    if isinstance(error, asyncio.CancelledError):
                        raise
                    raise BlindPoolLifecycleError(
                        "reservation_outcome_ambiguous",
                        "Blind Ladder reservation outcome requires startup recovery",
                    ) from None

            try:
                await self._prepare_team(reservation.team_id)
            except asyncio.CancelledError:
                try:
                    self._release_pre_accept(reservation)
                except BlindPoolReconciliationRequired:
                    raise
                except BlindPoolLifecycleError:
                    logger.error(
                        "Blind Ladder cancellation left reserved state for startup recovery"
                    )
                raise
            except Exception:
                try:
                    self._release_pre_accept(reservation)
                except BlindPoolReconciliationRequired:
                    raise
                except BlindPoolLifecycleError:
                    raise BlindPoolLifecycleError(
                        "team_preparation_cleanup_failed",
                        "Blind Ladder team preparation failed and cleanup could not be proven",
                    ) from None
                raise BlindPoolLifecycleError(
                    "team_preparation_failed",
                    "Blind Ladder team preparation failed",
                ) from None

            task = asyncio.current_task()
            if task is not None and task.cancelling():
                self._release_pre_accept(reservation)
                await asyncio.sleep(0)
                raise asyncio.CancelledError

            try:
                self._store.mark_accept_sent(reservation.reservation_id)
            except (BlindPoolValidationError, asyncio.CancelledError) as error:
                after = self._authoritative_snapshot()
                if self._reservation_is(
                    after.reservation,
                    reservation,
                    ACCEPT_SENT_PHASE,
                ):
                    self._quarantine()
                    if isinstance(error, asyncio.CancelledError):
                        raise
                    raise BlindPoolReconciliationRequired(
                        "acceptance_transition_ambiguous",
                        "Blind Ladder acceptance transition requires reconciliation",
                    ) from None
                if self._reservation_is(
                    after.reservation,
                    reservation,
                    RESERVATION_PHASE,
                ):
                    try:
                        self._release_pre_accept(reservation)
                    except BlindPoolReconciliationRequired:
                        raise
                    except BlindPoolLifecycleError:
                        raise BlindPoolLifecycleError(
                            "acceptance_transition_cleanup_failed",
                            "Blind Ladder acceptance transition failed and cleanup could not be proven",
                        ) from None
                    if isinstance(error, asyncio.CancelledError):
                        raise
                    raise BlindPoolLifecycleError(
                        "acceptance_transition_failed",
                        "Blind Ladder acceptance transition failed",
                    ) from None
                if isinstance(error, asyncio.CancelledError):
                    raise
                raise BlindPoolLifecycleError(
                    "acceptance_transition_outcome_ambiguous",
                    "Blind Ladder acceptance transition outcome cannot be proven",
                ) from None
            self._quarantine()

            task = asyncio.current_task()
            if task is not None and task.cancelling():
                await asyncio.sleep(0)
                raise asyncio.CancelledError

            try:
                await self._transport.send_challenge_acceptance(challenge)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.error(
                    "Blind Ladder acceptance is ambiguous; reconciliation required"
                )
                raise BlindPoolReconciliationRequired(
                    "acceptance_transmission_ambiguous",
                    "Blind Ladder acceptance transmission is ambiguous",
                ) from None

            room, event_lines = await self._wait_for_matching_room(challenge)
            try:
                self._store.commit_room_created(reservation.reservation_id)
            except (BlindPoolValidationError, asyncio.CancelledError) as error:
                after = self._authoritative_snapshot()
                if self._state_proves_commit(after, reservation):
                    if isinstance(error, asyncio.CancelledError):
                        self._remember_resolved_room(room.room_id)
                        self._clear_active_challenge()
                        self._reconciliation_pending = False
                        raise
                else:
                    self._quarantine()
                    if isinstance(error, asyncio.CancelledError):
                        raise
                    raise BlindPoolLifecycleError(
                        "room_created_commit_failed",
                        "Blind Ladder room-created commit could not be proven",
                    ) from None

            self._remember_resolved_room(room.room_id)
            self._clear_active_challenge()
            self._reconciliation_pending = False
            try:
                self._transport.install_blind_room_handoff(room, event_lines)
            except Exception:
                raise BlindPoolLifecycleError(
                    "room_handoff_failed",
                    "Battle initialization handoff failed after Blind Ladder commit",
                ) from None
            logger.info(
                "Correlated Blind Ladder room={} format={} and committed reservation".format(
                    room.room_id,
                    room.format_id,
                )
            )
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                await asyncio.sleep(0)
                raise asyncio.CancelledError
            if self._initialize_battle is None:
                return room
            try:
                return await self._initialize_battle(room)
            except asyncio.CancelledError:
                raise
            except Exception:
                raise BlindPoolLifecycleError(
                    "battle_initialization_failed",
                    "Battle initialization failed after Blind Ladder room commit",
                ) from None
        finally:
            self._task_executing = False
            if (
                not self._reconciliation_pending
                and self._active_challenge_identity is None
            ):
                self._run_in_progress = False
                self._release_execution_guard()
