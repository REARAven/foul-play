from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import tempfile
import traceback
from types import SimpleNamespace
from unittest import mock
import unittest

from fp.data.blind_pool import (
    PRIVATE_ROOT_ENV,
    REGISTRY_PATH_ENV,
    STATE_PATH_ENV,
    BlindPoolBagStore,
    BlindPoolLifecycleCoordinator,
    BlindPoolLifecycleError,
    BlindPoolReconciliationRequired,
    BlindPoolValidationError,
    BlindRoomCorrelator,
    load_blind_pool_config,
    load_blind_pool_registry,
    load_blind_pool_state_config,
    parse_incoming_challenge,
)
from fp.config import FoulPlayConfig
from fp.modes.base import get_battle_tag_and_opponent, get_first_request_json
from fp.websocket_client import PSWebsocketClient, _redact_received_message


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_SENTINEL = "PHASE4-SYNTHETIC-PRIVATE-SENTINEL"
CHALLENGE = "|pm|Synthetic Opponent|Blind Bot|/challenge|gen9tugs|||"
CURRENT_SERVER_CHALLENGE = (
    "|pm|Synthetic Opponent|Blind Bot|/challenge gen9tugs|gen9tugs|||"
)
DUPLICATE_CHALLENGE = CHALLENGE
WRONG_FORMAT_CHALLENGE = (
    "|pm|Synthetic Opponent|Blind Bot|/challenge|gen9nationaldex|||"
)


def room_message(room_id: str = "battle-gen9tugs-401") -> str:
    return "\n".join(
        (
            ">" + room_id,
            "|init|battle",
            "|title|" + PRIVATE_SENTINEL,
            "|player|p1|Synthetic Opponent|1|",
            "|player|p2|Blind Bot|1|",
        )
    )


class IdentityRandom:
    def shuffle(self, values: list[str]) -> None:
        return None

    def randrange(self, start: int, stop: int | None = None) -> int:
        return start


class FakeTransport:
    username = "Blind Bot"

    def __init__(
        self,
        messages=(),
        *,
        send_error: BaseException | None = None,
        block_when_empty: bool = False,
        receive_observer=None,
        send_observer=None,
    ) -> None:
        self.messages = list(messages)
        self.send_error = send_error
        self.block_when_empty = block_when_empty
        self.receive_observer = receive_observer
        self.send_observer = send_observer
        self.sent = []
        self.handoffs = []
        self.receive_calls = 0
        self._never = asyncio.Event()

    async def receive_message(self) -> str:
        self.receive_calls += 1
        replay = vars(self).get("_blind_room_replay")
        if replay:
            return replay.popleft()
        if self.messages:
            message = self.messages.pop(0)
            if self.receive_observer is not None:
                self.receive_observer(message)
            return message
        if self.block_when_empty:
            await self._never.wait()
        raise ConnectionError(PRIVATE_SENTINEL)

    async def send_challenge_acceptance(self, challenge) -> None:
        self.sent.append(challenge)
        if self.send_observer is not None:
            self.send_observer(challenge)
        if self.send_error is not None:
            raise self.send_error

    def install_blind_room_handoff(self, room, event_lines) -> None:
        self.handoffs.append((room, event_lines))
        PSWebsocketClient.install_blind_room_handoff(self, room, event_lines)


class BlindPoolLifecycleFixture(unittest.IsolatedAsyncioTestCase):
    active_ids = ("BL-001-v1", "BL-002-v1", "BL-003-v1")

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.base = Path(self.temporary_directory.name)
        self.private_root = self.base / "synthetic-private"
        self.team_directory = self.private_root / "teams"
        self.state_directory = self.private_root / "state"
        self.team_directory.mkdir(parents=True)
        self.state_directory.mkdir()
        self.registry_path = self.private_root / "registry.json"
        self.state_path = self.state_directory / "bag-state.json"
        entries = []
        for team_id in self.active_ids:
            payload = (PRIVATE_SENTINEL + "-" + team_id).encode()
            path = self.team_directory / (team_id + ".team")
            path.write_bytes(payload)
            entries.append(
                {
                    "team_id": team_id,
                    "active": True,
                    "team_file": "teams/" + path.name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        self.registry_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "registry_version": "4.0",
                    "format_id": "gen9tugs",
                    "entries": entries,
                }
            ),
            encoding="utf-8",
        )
        environment = {
            PRIVATE_ROOT_ENV: str(self.private_root),
            REGISTRY_PATH_ENV: str(self.registry_path),
            STATE_PATH_ENV: str(self.state_path),
        }
        config = load_blind_pool_config(environment, repository_root=ROOT)
        state_config = load_blind_pool_state_config(
            environment,
            repository_root=ROOT,
        )
        registry = load_blind_pool_registry(config)
        identifiers = iter(format(index, "032x") for index in range(1, 100))
        self.store = BlindPoolBagStore(
            state_config,
            registry,
            random_source=IdentityRandom(),
            reservation_id_factory=lambda: next(identifiers),
            lock_timeout_seconds=1,
        )

    def coordinator(
        self,
        transport: FakeTransport,
        *,
        prepare=None,
        initialize=None,
        timeout: float = 0.1,
        mode: str = "accept_challenge",
        format_id: str = "gen9tugs",
        max_room_candidates: int = 8,
    ) -> BlindPoolLifecycleCoordinator:
        async def successful_prepare(_team_id: str) -> None:
            return None

        return BlindPoolLifecycleCoordinator(
            self.store,
            transport,
            prepare or successful_prepare,
            initialize_battle=initialize,
            room_timeout_seconds=timeout,
            mode=mode,
            format_id=format_id,
            max_room_candidates=max_room_candidates,
        )

    def assert_error(self, code: str, operation):
        with self.assertRaises(BlindPoolValidationError) as captured:
            operation()
        self.assertEqual(code, captured.exception.code)
        return captured.exception

    def reserve_accept_sent(self):
        self.store.initialize_or_load()
        reservation = self.store.reserve_next()
        self.store.mark_accept_sent(reservation.reservation_id)
        return reservation

    async def assert_self_cancellation(self, awaitable) -> None:
        try:
            await awaitable
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is not None:
                while task.cancelling():
                    task.uncancel()
        else:
            self.fail("expected coordinator cancellation")


class TestBlindPoolChallengeParsing(BlindPoolLifecycleFixture):
    def test_valid_exact_challenge_returns_minimal_immutable_model(self):
        challenge = parse_incoming_challenge(
            CHALLENGE,
            bot_username="Blind Bot",
            required_format="gen9tugs",
        )
        self.assertEqual("syntheticopponent", challenge.challenger_id)
        self.assertEqual("Synthetic Opponent", challenge.challenger_name)
        self.assertEqual("gen9tugs", challenge.format_id)
        self.assertEqual("pm", challenge.source)
        self.assertFalse(hasattr(challenge, "raw_message"))
        self.assertNotIn("Synthetic Opponent", repr(challenge))
        with self.assertRaises(Exception):
            challenge.format_id = "changed"

    def test_current_server_challenge_encoding_is_accepted_exactly(self):
        challenge = parse_incoming_challenge(
            CURRENT_SERVER_CHALLENGE,
            bot_username="blindbot",
            required_format="gen9tugs",
        )
        self.assertEqual("syntheticopponent", challenge.challenger_id)
        altered = (
            CURRENT_SERVER_CHALLENGE + "|",
            CURRENT_SERVER_CHALLENGE.replace("|||", "||x"),
            CURRENT_SERVER_CHALLENGE.replace(
                "/challenge gen9tugs", "/challenge  gen9tugs"
            ),
            CURRENT_SERVER_CHALLENGE.replace(
                "gen9tugs|gen9tugs", "gen9tugs|gen9nationaldex"
            ),
        )
        for message in altered:
            with self.subTest(message_length=len(message)):
                self.assertIsNone(
                    parse_incoming_challenge(
                        message,
                        bot_username="blindbot",
                        required_format="gen9tugs",
                    )
                )

    def test_multiline_payload_can_contain_multiple_exact_challenges(self):
        message = "\n".join(("|updateuser|safe|1|0", CHALLENGE, CHALLENGE))
        challenge = parse_incoming_challenge(
            message,
            bot_username="Blind Bot",
            required_format="gen9tugs",
        )
        self.assertEqual("syntheticopponent", challenge.challenger_id)

    def test_challenger_and_ranked_recipient_are_normalized(self):
        challenge = parse_incoming_challenge(
            "|pm|Opp-on_ent|‽Blind Bot|/challenge|gen9tugs|||",
            bot_username="blindbot",
            required_format="gen9tugs",
        )
        self.assertEqual("opponent", challenge.challenger_id)

    def test_wrong_format_malformed_and_wrong_recipient_are_ignored(self):
        messages = (
            WRONG_FORMAT_CHALLENGE,
            "|pm|Opponent|Blind Bot|/challenge-extra|gen9tugs|||",
            "|pm|Opponent|Other Bot|/challenge|gen9tugs|||",
            "|pm|Opponent|Blind Bot|/challenge|gen9tugs||",
            PRIVATE_SENTINEL,
        )
        for message in messages:
            with self.subTest(message_length=len(message)):
                self.assertIsNone(
                    parse_incoming_challenge(
                        message,
                        bot_username="Blind Bot",
                        required_format="gen9tugs",
                    )
                )

    async def test_legacy_combined_accept_wrapper_remains_functional(self):
        client = PSWebsocketClient()
        client.username = "Blind Bot"
        client.receive_message = mock.AsyncMock(
            side_effect=(WRONG_FORMAT_CHALLENGE, CHALLENGE)
        )
        client.join_room = mock.AsyncMock()
        client.send_message = mock.AsyncMock()
        await client.accept_challenge("gen9tugs", "synthetic-lobby")
        client.join_room.assert_awaited_once_with("synthetic-lobby")
        client.send_message.assert_awaited_once_with(
            "",
            ["/accept Synthetic Opponent"],
        )

    def test_lifecycle_is_eligible_only_for_exact_format_and_mode(self):
        transport = FakeTransport()
        for format_id, mode in (
            ("gen9nationaldex", "accept_challenge"),
            ("gen9tugs", "challenge_user"),
            ("gen9tugs", "search_ladder"),
        ):
            with self.subTest(format_id=format_id, mode=mode):
                with self.assertRaises(BlindPoolLifecycleError) as captured:
                    self.coordinator(
                        transport,
                        format_id=format_id,
                        mode=mode,
                    )
                self.assertEqual("lifecycle_ineligible", captured.exception.code)

    def test_invalid_room_capacity_is_rejected_before_lifecycle_start(self):
        for value in (0, -1, True, 1.5, "8"):
            with self.subTest(value=value):
                with self.assertRaises(BlindPoolLifecycleError) as captured:
                    self.coordinator(
                        FakeTransport((CHALLENGE, room_message())),
                        max_room_candidates=value,
                    )
                self.assertEqual(
                    "room_correlation_configuration_invalid",
                    captured.exception.code,
                )
                self.assertFalse(self.state_path.exists())


class TestBlindPoolRoomCorrelation(BlindPoolLifecycleFixture):
    def correlator(self, **kwargs) -> BlindRoomCorrelator:
        challenge = parse_incoming_challenge(
            CHALLENGE,
            bot_username="Blind Bot",
            required_format="gen9tugs",
        )
        return BlindRoomCorrelator(
            challenge=challenge,
            bot_username="Blind Bot",
            **kwargs,
        )

    def test_exact_complete_room_returns_safe_metadata(self):
        room = self.correlator().feed(room_message())
        self.assertEqual("battle-gen9tugs-401", room.room_id)
        self.assertEqual("gen9tugs", room.format_id)
        self.assertEqual("syntheticopponent", room.opponent_id)
        self.assertEqual("p2", room.bot_slot)
        self.assertEqual("p1", room.opponent_slot)
        self.assertNotIn(PRIVATE_SENTINEL, repr(room))
        self.assertFalse(hasattr(room, "raw_message"))

    def test_both_player_orders_and_optional_fields_correlate(self):
        messages = (
            ">battle-gen9tugs-420\n|init|battle\n"
            "|player|p1|Blind Bot|12|1234\n"
            "|player|p2|Synthetic Opponent|7|1450",
            ">battle-gen9tugs-421\n|init|battle\n"
            "|player|p1|Synthetic Opponent|7|1450\n"
            "|player|p2|Blind Bot|12|1234",
        )
        for index, message in enumerate(messages):
            with self.subTest(index=index):
                room = self.correlator().feed(message)
                self.assertIsNotNone(room)
                self.assertNotEqual(room.bot_slot, room.opponent_slot)

    def test_contradictory_or_empty_player_reassignment_fails_closed(self):
        for contradictory in (
            "|player|p1|Different Opponent|1|",
            "|player|p1||1|",
        ):
            with self.subTest(contradictory=bool(contradictory)):
                correlator = self.correlator()
                self.assertIsNone(
                    correlator.feed(
                        ">battle-gen9tugs-430\n|init|battle\n"
                        "|player|p1|Synthetic Opponent|1|"
                    )
                )
                self.assertIsNone(
                    correlator.feed(
                        ">battle-gen9tugs-430\n{}\n|player|p2|Blind Bot|1|".format(
                            contradictory
                        )
                    )
                )

    def test_multiple_room_sections_in_one_payload_are_routed_exactly(self):
        payload = "\n".join(
            (
                "|updateuser|safe|1|0",
                ">battle-gen9nationaldex-1",
                "|init|battle",
                "|player|p1|Synthetic Opponent|1|",
                "|player|p2|Blind Bot|1|",
                ">battle-gen9tugs-431",
                "|init|battle",
                "|player|p1|Synthetic Opponent|1|",
                "|player|p2|Blind Bot|1|",
            )
        )
        room = self.correlator().feed(payload)
        self.assertEqual("battle-gen9tugs-431", room.room_id)

    def test_incomplete_multi_message_room_does_not_commit_early(self):
        correlator = self.correlator()
        room_id = "battle-gen9tugs-402"
        self.assertIsNone(correlator.feed(">" + room_id + "\n|init|battle"))
        self.assertIsNone(
            correlator.feed(">" + room_id + "\n|player|p1|Synthetic Opponent|1|")
        )
        room = correlator.feed(">" + room_id + "\n|player|p2|Blind Bot|1|")
        self.assertEqual(room_id, room.room_id)

    def test_inexact_or_uninitialized_rooms_never_correlate(self):
        messages = (
            "this contains battle but has no room envelope",
            ">not-a-battle-room\n|init|battle",
            room_message("battle-gen9nationaldex-1"),
            ">battle-gen9tugs-0\n|init|battle",
            ">battle-gen9tugs-2\n|player|p1|Synthetic Opponent|1|\n"
            "|player|p2|Blind Bot|1|",
            ">battle-gen9tugs-3\n|init|battle\n|player|p1|Wrong Opponent|1|\n"
            "|player|p2|Blind Bot|1|",
            ">battle-gen9tugs-4\n|init|chat\n|player|p1|Synthetic Opponent|1|\n"
            "|player|p2|Blind Bot|1|",
        )
        correlator = self.correlator()
        for message in messages:
            with self.subTest(message_length=len(message)):
                self.assertIsNone(correlator.feed(message))

    def test_stale_excluded_room_is_ignored(self):
        correlator = self.correlator(
            excluded_room_ids=("battle-gen9tugs-401",)
        )
        self.assertIsNone(correlator.feed(room_message()))

    def test_candidate_state_is_bounded(self):
        correlator = self.correlator(max_candidates=2)
        for number in range(1, 8):
            correlator.feed(">battle-gen9tugs-{}\n|init|battle".format(number))
        self.assertEqual(2, correlator.candidate_count)
        self.assertIsNone(correlator.feed(room_message("battle-gen9tugs-99")))
        self.assertNotIn(PRIVATE_SENTINEL, repr(correlator))

    def test_repeated_identical_player_and_init_events_remain_coherent(self):
        correlator = self.correlator()
        message = "\n".join(
            (
                ">battle-gen9tugs-432",
                "|init|battle",
                "|init|battle",
                "|player|p1|Synthetic Opponent|1|",
                "|player|p1|Synthetic Opponent|2|",
                "|player|p2|Blind Bot|1|",
                "|player|p2|Blind Bot|2|",
            )
        )
        room = correlator.feed(message)
        self.assertEqual("battle-gen9tugs-432", room.room_id)


class TestBlindPoolLifecycleOrdering(BlindPoolLifecycleFixture):
    async def test_bot_as_p1_handoff_preserves_opponent_slot_for_legacy_reader(self):
        initial = "\n".join(
            (
                ">battle-gen9tugs-451",
                "|init|battle",
                "|player|p1|Blind Bot|1|",
                "|player|p2|Synthetic Opponent|1|",
            )
        )
        transport = FakeTransport((CHALLENGE, initial))

        async def initialize(room):
            staged = tuple(transport._blind_room_replay)
            self.assertIn("|player|p2|Synthetic Opponent", staged[1])
            self.assertEqual("p2", staged[1].split("|")[2])
            with mock.patch.object(
                FoulPlayConfig,
                "username",
                "Blind Bot",
                create=True,
            ):
                battle_tag, opponent = await get_battle_tag_and_opponent(transport)
            self.assertEqual("battle-gen9tugs-451", battle_tag)
            self.assertEqual("Synthetic Opponent", opponent)
            return room

        room = await self.coordinator(
            transport,
            initialize=initialize,
        ).run_once()
        self.assertEqual("p1", room.bot_slot)
        self.assertEqual("p2", room.opponent_slot)

    async def test_duplicate_complete_room_in_one_payload_commits_once(self):
        duplicate_payload = room_message() + "\n" + room_message()
        transport = FakeTransport((CHALLENGE, duplicate_payload))
        with mock.patch.object(
            self.store,
            "commit_room_created",
            wraps=self.store.commit_room_created,
        ) as commit:
            await self.coordinator(transport).run_once()
        self.assertEqual(1, commit.call_count)
        self.assertEqual(1, self.store.snapshot().next_index)

    async def test_duplicate_challenge_and_global_event_do_not_hide_same_frame_room(self):
        combined = "\n".join(
            (
                CHALLENGE,
                room_message(),
                "|updateuser|unrelated|1|0",
            )
        )
        transport = FakeTransport((CHALLENGE, combined))
        room = await self.coordinator(transport).run_once()
        self.assertEqual("battle-gen9tugs-401", room.room_id)
        self.assertEqual(1, self.store.snapshot().next_index)
        self.assertNotIn(
            "|updateuser|",
            "\n".join(transport.handoffs[0][1]),
        )
    async def test_correlated_initialization_and_request_are_handed_to_normal_readers(self):
        request = json.dumps({"rqid": 17, "private": PRIVATE_SENTINEL})
        initial = "\n".join(
            (
                ">battle-gen9tugs-450",
                "|init|battle",
                "|title|Synthetic Opponent vs. Blind Bot",
                "|player|p1|Synthetic Opponent|1|",
                "|player|p2|Blind Bot|1|",
                "|gen|9",
                "|tier|gen9tugs",
                "|teampreview",
                "|request|" + request,
            )
        )
        transport = FakeTransport((CHALLENGE, initial))

        async def initialize(room):
            with mock.patch.object(
                FoulPlayConfig,
                "username",
                "Blind Bot",
                create=True,
            ):
                battle_tag, opponent = await get_battle_tag_and_opponent(transport)
            self.assertEqual(room.room_id, battle_tag)
            self.assertEqual(room.opponent_name, opponent)
            participant = await transport.receive_message()
            self.assertIn("|player|p1|Synthetic Opponent", participant)
            setup = await transport.receive_message()
            self.assertIn("|player|p2|Blind Bot", setup)
            self.assertIn("|gen|9", setup)
            self.assertIn("|teampreview", setup)
            self.assertNotIn("|request|", setup)
            battle = SimpleNamespace(
                user=SimpleNamespace(
                    initialize_first_turn_user_from_json=mock.Mock()
                ),
                request_json=None,
                rqid=None,
            )
            await get_first_request_json(transport, battle)
            self.assertEqual(17, battle.rqid)
            self.assertEqual(PRIVATE_SENTINEL, battle.request_json["private"])
            return "initialized"

        output = io.StringIO()
        handler = logging.StreamHandler(output)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        lifecycle_logger = logging.getLogger("fp.data.blind_pool.lifecycle")
        websocket_logger = logging.getLogger("fp.websocket_client")
        lifecycle_logger.addHandler(handler)
        websocket_logger.addHandler(handler)
        try:
            result = await self.coordinator(
                transport,
                initialize=initialize,
            ).run_once()
        finally:
            lifecycle_logger.removeHandler(handler)
            websocket_logger.removeHandler(handler)
        self.assertEqual("initialized", result)
        self.assertEqual(1, self.store.snapshot().next_index)
        self.assertNotIn(PRIVATE_SENTINEL, output.getvalue())

    async def test_true_concurrent_run_attempt_is_rejected_before_second_read(self):
        preparation_started = asyncio.Event()
        never = asyncio.Event()

        async def blocked_prepare(_team_id):
            preparation_started.set()
            await never.wait()

        transport = FakeTransport((CHALLENGE,), block_when_empty=True)
        coordinator = self.coordinator(transport, prepare=blocked_prepare)
        first = asyncio.create_task(coordinator.run_once())
        await asyncio.wait_for(preparation_started.wait(), timeout=1)
        reads = transport.receive_calls
        with self.assertRaises(BlindPoolLifecycleError) as captured:
            await coordinator.run_once()
        self.assertEqual("lifecycle_already_active", captured.exception.code)
        self.assertEqual(reads, transport.receive_calls)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first

    async def test_room_observed_before_challenge_is_excluded_as_stale(self):
        stale = room_message("battle-gen9tugs-440")
        fresh = room_message("battle-gen9tugs-441")
        transport = FakeTransport((stale, CHALLENGE, stale, fresh))
        room = await self.coordinator(transport).run_once()
        self.assertEqual("battle-gen9tugs-441", room.room_id)

    async def test_unrelated_global_messages_do_not_trigger_acceptance(self):
        transport = FakeTransport(
            (
                "|updateuser|Someone Else|1|0",
                WRONG_FORMAT_CHALLENGE,
                CHALLENGE,
                room_message(),
            )
        )
        await self.coordinator(transport).run_once()
        self.assertEqual(1, len(transport.sent))
        self.assertEqual(1, self.store.snapshot().next_index)

    async def test_successful_path_has_exact_write_ahead_and_commit_order(self):
        events = []

        async def prepare(team_id: str) -> None:
            state = self.store.snapshot()
            self.assertEqual("reserved", state.reservation.phase)
            self.assertEqual(team_id, state.reservation.team_id)
            events.append("prepare")

        def receive_observer(message: str) -> None:
            if message.startswith(">"):
                events.append("room")

        def send_observer(_challenge) -> None:
            self.assertEqual("accept_sent", self.store.snapshot().reservation.phase)
            events.append("send")

        async def initialize(room) -> str:
            state = self.store.snapshot()
            self.assertIsNone(state.reservation)
            self.assertEqual(1, state.next_index)
            self.assertEqual("battle-gen9tugs-401", room.room_id)
            events.append("initialize")
            return "initialized"

        transport = FakeTransport(
            (CHALLENGE, room_message()),
            receive_observer=receive_observer,
            send_observer=send_observer,
        )
        original_reserve = self.store.reserve_next
        original_mark = self.store.mark_accept_sent
        original_commit = self.store.commit_room_created

        def reserve():
            result = original_reserve()
            events.append("reserve")
            return result

        def mark(identifier):
            result = original_mark(identifier)
            events.append("mark")
            return result

        def commit(identifier):
            result = original_commit(identifier)
            events.append("commit")
            return result

        with mock.patch.object(self.store, "reserve_next", side_effect=reserve), mock.patch.object(
            self.store,
            "mark_accept_sent",
            side_effect=mark,
        ), mock.patch.object(
            self.store,
            "commit_room_created",
            side_effect=commit,
        ):
            result = await self.coordinator(
                transport,
                prepare=prepare,
                initialize=initialize,
            ).run_once()
        self.assertEqual("initialized", result)
        self.assertEqual(
            ["reserve", "prepare", "mark", "send", "room", "commit", "initialize"],
            events,
        )

    async def test_duplicate_challenge_reserves_uploads_and_accepts_once(self):
        prepare_calls = []

        async def prepare(team_id: str) -> None:
            prepare_calls.append(team_id)

        transport = FakeTransport(
            (CHALLENGE, DUPLICATE_CHALLENGE, DUPLICATE_CHALLENGE, room_message())
        )
        coordinator = self.coordinator(transport, prepare=prepare)
        with mock.patch.object(
            self.store,
            "reserve_next",
            wraps=self.store.reserve_next,
        ) as reserve:
            await coordinator.run_once()
        self.assertEqual(1, reserve.call_count)
        self.assertEqual(1, len(prepare_calls))
        self.assertEqual(1, len(transport.sent))

    async def test_deduplication_clears_after_resolution_and_stale_room_is_ignored(self):
        transport = FakeTransport(
            (
                CHALLENGE,
                room_message("battle-gen9tugs-410"),
                CHALLENGE,
                room_message("battle-gen9tugs-410"),
                room_message("battle-gen9tugs-411"),
            )
        )
        coordinator = self.coordinator(transport)
        first = await coordinator.run_once()
        second = await coordinator.run_once()
        self.assertEqual("battle-gen9tugs-410", first.room_id)
        self.assertEqual("battle-gen9tugs-411", second.room_id)
        self.assertEqual(2, self.store.snapshot().next_index)
        self.assertEqual(2, len(transport.sent))


class TestBlindPoolLifecycleFailures(BlindPoolLifecycleFixture):
    async def test_cancellation_before_reservation_leaves_no_reservation(self):
        transport = FakeTransport((), block_when_empty=True)
        task = asyncio.create_task(self.coordinator(transport).run_once())
        for _ in range(100):
            await asyncio.sleep(0)
            if transport.receive_calls:
                break
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIsNone(self.store.snapshot().reservation)

    async def test_cancellation_requested_after_preparation_releases_reserved(self):
        async def cancel_after_prepare(_team_id):
            asyncio.current_task().cancel()

        await self.assert_self_cancellation(
            self.coordinator(
                FakeTransport((CHALLENGE,)),
                prepare=cancel_after_prepare,
            ).run_once()
        )
        self.assertIsNone(self.store.snapshot().reservation)

    async def test_cancellation_after_durable_mark_quarantines_accept_sent(self):
        transport = FakeTransport((CHALLENGE,))
        original = self.store.mark_accept_sent

        def mark_then_cancel(identifier):
            state = original(identifier)
            asyncio.current_task().cancel()
            return state

        with mock.patch.object(
            self.store,
            "mark_accept_sent",
            side_effect=mark_then_cancel,
        ):
            await self.assert_self_cancellation(
                self.coordinator(transport).run_once()
            )
        self.assertEqual("accept_sent", self.store.snapshot().reservation.phase)
        self.assertLessEqual(len(transport.sent), 1)

    async def test_cancellation_during_accept_transmission_quarantines(self):
        started = asyncio.Event()
        never = asyncio.Event()
        transport = FakeTransport((CHALLENGE,))

        async def blocked_send(challenge):
            transport.sent.append(challenge)
            started.set()
            await never.wait()

        transport.send_challenge_acceptance = blocked_send
        task = asyncio.create_task(self.coordinator(transport).run_once())
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual("accept_sent", self.store.snapshot().reservation.phase)
        self.assertEqual(1, len(transport.sent))

    async def test_cancellation_after_room_correlation_commits_before_propagating(self):
        transport = FakeTransport((CHALLENGE, room_message()))
        original = BlindRoomCorrelator.feed

        def correlate_then_cancel(correlator, message):
            room = original(correlator, message)
            if room is not None:
                asyncio.current_task().cancel()
            return room

        with mock.patch.object(
            BlindRoomCorrelator,
            "feed",
            new=correlate_then_cancel,
        ):
            await self.assert_self_cancellation(
                self.coordinator(transport).run_once()
            )
        state = self.store.snapshot()
        self.assertIsNone(state.reservation)
        self.assertEqual(1, state.next_index)

    async def test_cancellation_raised_after_commit_is_not_swallowed(self):
        transport = FakeTransport((CHALLENGE, room_message()))
        original = self.store.commit_room_created

        def commit_then_cancel(identifier):
            original(identifier)
            raise asyncio.CancelledError

        with mock.patch.object(
            self.store,
            "commit_room_created",
            side_effect=commit_then_cancel,
        ), self.assertRaises(asyncio.CancelledError):
            await self.coordinator(transport).run_once()
        state = self.store.snapshot()
        self.assertIsNone(state.reservation)
        self.assertEqual(1, state.next_index)

    async def test_cancellation_during_battle_initializer_keeps_consumed(self):
        started = asyncio.Event()
        never = asyncio.Event()

        async def blocked_initialize(_room):
            started.set()
            await never.wait()

        transport = FakeTransport((CHALLENGE, room_message()))
        task = asyncio.create_task(
            self.coordinator(transport, initialize=blocked_initialize).run_once()
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        state = self.store.snapshot()
        self.assertIsNone(state.reservation)
        self.assertEqual(1, state.next_index)

    async def test_reserve_exception_after_replace_keeps_one_identified_reservation(self):
        prepared = []

        async def prepare(team_id):
            prepared.append(team_id)

        transport = FakeTransport((CHALLENGE,))
        coordinator = self.coordinator(transport, prepare=prepare)
        await coordinator.startup()
        with mock.patch(
            "fp.data.blind_pool.state._fsync_directory",
            side_effect=OSError(PRIVATE_SENTINEL),
        ):
            with self.assertRaises(BlindPoolLifecycleError) as captured:
                await coordinator.run_once()
        self.assertEqual("reservation_outcome_ambiguous", captured.exception.code)
        self.assertEqual("reserved", self.store.snapshot().reservation.phase)
        self.assertEqual([], prepared)
        self.assertEqual([], transport.sent)

    async def test_mark_exception_after_replace_quarantines_without_accepting(self):
        transport = FakeTransport((CHALLENGE,))
        original = self.store.mark_accept_sent

        def fail_after_replace(identifier):
            with mock.patch(
                "fp.data.blind_pool.state._fsync_directory",
                side_effect=OSError(PRIVATE_SENTINEL),
            ):
                return original(identifier)

        with mock.patch.object(
            self.store,
            "mark_accept_sent",
            side_effect=fail_after_replace,
        ), self.assertRaises(BlindPoolReconciliationRequired) as captured:
            await self.coordinator(transport).run_once()
        self.assertEqual("acceptance_transition_ambiguous", captured.exception.code)
        self.assertEqual("accept_sent", self.store.snapshot().reservation.phase)
        self.assertEqual([], transport.sent)

    async def test_commit_exception_after_replace_is_classified_as_consumed(self):
        initialized = []

        async def initialize(room):
            initialized.append(room.room_id)
            return "ready"

        transport = FakeTransport((CHALLENGE, room_message()))
        original = self.store.commit_room_created

        def fail_after_replace(identifier):
            with mock.patch(
                "fp.data.blind_pool.state._fsync_directory",
                side_effect=OSError(PRIVATE_SENTINEL),
            ):
                return original(identifier)

        with mock.patch.object(
            self.store,
            "commit_room_created",
            side_effect=fail_after_replace,
        ):
            result = await self.coordinator(
                transport,
                initialize=initialize,
            ).run_once()
        self.assertEqual("ready", result)
        self.assertEqual(["battle-gen9tugs-401"], initialized)
        state = self.store.snapshot()
        self.assertIsNone(state.reservation)
        self.assertEqual(1, state.next_index)

    async def test_release_exception_after_replace_preserves_original_failure(self):
        async def fail(_team_id):
            raise ValueError(PRIVATE_SENTINEL)

        transport = FakeTransport((CHALLENGE,))
        original = self.store.release_reservation

        def fail_after_replace(identifier):
            with mock.patch(
                "fp.data.blind_pool.state._fsync_directory",
                side_effect=OSError(PRIVATE_SENTINEL),
            ):
                return original(identifier)

        with mock.patch.object(
            self.store,
            "release_reservation",
            side_effect=fail_after_replace,
        ), self.assertRaises(BlindPoolLifecycleError) as captured:
            await self.coordinator(transport, prepare=fail).run_once()
        self.assertEqual("team_preparation_failed", captured.exception.code)
        self.assertIsNone(self.store.snapshot().reservation)

    async def test_cancellation_during_preparation_releases_reserved_state(self):
        started = asyncio.Event()
        never = asyncio.Event()

        async def prepare(_team_id: str) -> None:
            started.set()
            await never.wait()

        transport = FakeTransport((CHALLENGE,))
        task = asyncio.create_task(
            self.coordinator(transport, prepare=prepare).run_once()
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        state = self.store.snapshot()
        self.assertIsNone(state.reservation)
        self.assertEqual(0, state.next_index)

    async def test_preparation_and_provider_failures_release_same_position(self):
        for failure in (ValueError(PRIVATE_SENTINEL), OSError(PRIVATE_SENTINEL)):
            with self.subTest(failure_type=type(failure).__name__):
                if self.state_path.exists():
                    self.state_path.unlink()

                async def fail(_team_id: str, error=failure) -> None:
                    raise error

                transport = FakeTransport((CHALLENGE,))
                with self.assertRaises(BlindPoolLifecycleError) as captured:
                    await self.coordinator(transport, prepare=fail).run_once()
                self.assertEqual("team_preparation_failed", captured.exception.code)
                state = self.store.snapshot()
                self.assertIsNone(state.reservation)
                self.assertEqual(0, state.next_index)
                next_reservation = self.store.reserve_next()
                self.assertEqual(self.active_ids[0], next_reservation.team_id)
                self.store.release_reservation(next_reservation.reservation_id)

    async def test_failed_accept_transition_does_not_send_and_releases(self):
        transport = FakeTransport((CHALLENGE,))
        coordinator = self.coordinator(transport)
        with mock.patch.object(
            self.store,
            "mark_accept_sent",
            side_effect=BlindPoolValidationError("synthetic", "safe"),
        ):
            with self.assertRaises(BlindPoolLifecycleError) as captured:
                await coordinator.run_once()
        self.assertEqual("acceptance_transition_failed", captured.exception.code)
        self.assertEqual([], transport.sent)
        self.assertIsNone(self.store.snapshot().reservation)

    async def test_pre_accept_release_failure_is_surfaced(self):
        async def fail(_team_id: str) -> None:
            raise ValueError(PRIVATE_SENTINEL)

        transport = FakeTransport((CHALLENGE,))
        error = BlindPoolValidationError("synthetic_release_failed", "safe")
        with mock.patch.object(self.store, "release_reservation", side_effect=error):
            with self.assertRaises(BlindPoolLifecycleError) as captured:
                await self.coordinator(transport, prepare=fail).run_once()
        self.assertEqual("team_preparation_cleanup_failed", captured.exception.code)
        self.assertEqual("reserved", self.store.snapshot().reservation.phase)

    async def test_accept_transition_and_cleanup_failure_are_both_reported(self):
        transport = FakeTransport((CHALLENGE,))
        transition_error = BlindPoolValidationError("synthetic_mark_failed", "safe")
        release_error = BlindPoolValidationError("synthetic_release_failed", "safe")
        with mock.patch.object(
            self.store,
            "mark_accept_sent",
            side_effect=transition_error,
        ), mock.patch.object(
            self.store,
            "release_reservation",
            side_effect=release_error,
        ), self.assertRaises(BlindPoolLifecycleError) as captured:
            await self.coordinator(transport).run_once()
        self.assertEqual(
            "acceptance_transition_cleanup_failed",
            captured.exception.code,
        )
        self.assertEqual("reserved", self.store.snapshot().reservation.phase)
        self.assertEqual([], transport.sent)

    async def test_cancellation_is_preserved_when_reserved_cleanup_fails(self):
        started = asyncio.Event()
        never = asyncio.Event()

        async def blocked_prepare(_team_id):
            started.set()
            await never.wait()

        transport = FakeTransport((CHALLENGE,))
        coordinator = self.coordinator(transport, prepare=blocked_prepare)
        task = asyncio.create_task(coordinator.run_once())
        await asyncio.wait_for(started.wait(), timeout=1)
        release_error = BlindPoolValidationError("synthetic_release_failed", "safe")
        with mock.patch.object(
            self.store,
            "release_reservation",
            side_effect=release_error,
        ):
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual("reserved", self.store.snapshot().reservation.phase)

    async def test_send_exception_leaves_accept_sent_and_never_releases(self):
        transport = FakeTransport(
            (CHALLENGE,),
            send_error=OSError(PRIVATE_SENTINEL),
        )
        with mock.patch.object(
            self.store,
            "release_reservation",
            wraps=self.store.release_reservation,
        ) as release:
            with self.assertRaises(BlindPoolReconciliationRequired) as captured:
                await self.coordinator(transport).run_once()
        self.assertEqual("acceptance_transmission_ambiguous", captured.exception.code)
        self.assertEqual("accept_sent", self.store.snapshot().reservation.phase)
        release.assert_not_called()

    async def test_same_coordinator_reentry_fails_closed_without_listening(self):
        transport = FakeTransport(
            (CHALLENGE,),
            send_error=OSError(PRIVATE_SENTINEL),
        )
        coordinator = self.coordinator(transport)
        with self.assertRaises(BlindPoolReconciliationRequired):
            await coordinator.run_once()
        calls_after_ambiguity = transport.receive_calls
        with self.assertRaises(BlindPoolReconciliationRequired) as captured:
            await coordinator.run_once()
        self.assertEqual("reconciliation_required", captured.exception.code)
        self.assertEqual(calls_after_ambiguity, transport.receive_calls)

    async def test_disconnect_and_unrelated_or_malformed_traffic_leave_accept_sent(self):
        messages = (
            CHALLENGE,
            PRIVATE_SENTINEL,
            room_message("battle-gen9nationaldex-7"),
            ">battle-gen9tugs-8\n|init|battle\n|player|p1|Wrong Opponent|1|",
        )
        transport = FakeTransport(messages)
        with self.assertRaises(BlindPoolReconciliationRequired) as captured:
            await self.coordinator(transport).run_once()
        self.assertEqual("room_correlation_failed", captured.exception.code)
        self.assertEqual("accept_sent", self.store.snapshot().reservation.phase)

    async def test_room_timeout_leaves_accept_sent(self):
        transport = FakeTransport((CHALLENGE,), block_when_empty=True)
        with self.assertRaises(BlindPoolReconciliationRequired) as captured:
            await self.coordinator(transport, timeout=0.01).run_once()
        self.assertEqual("matching_room_timed_out", captured.exception.code)
        self.assertEqual("accept_sent", self.store.snapshot().reservation.phase)

    async def test_coordinator_cancellation_after_accept_sent_does_not_release(self):
        transport = FakeTransport((CHALLENGE,), block_when_empty=True)
        coordinator = self.coordinator(transport, timeout=2)
        task = asyncio.create_task(coordinator.run_once())
        for _ in range(100):
            await asyncio.sleep(0)
            state = self.store.initialize_or_load()
            if state.reservation is not None and state.reservation.phase == "accept_sent":
                break
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual("accept_sent", self.store.snapshot().reservation.phase)

    async def test_commit_failure_prevents_battle_initialization(self):
        initialized = False

        async def initialize(_room):
            nonlocal initialized
            initialized = True

        transport = FakeTransport((CHALLENGE, room_message()))
        with mock.patch.object(
            self.store,
            "commit_room_created",
            side_effect=BlindPoolValidationError("synthetic", "safe"),
        ):
            with self.assertRaises(BlindPoolLifecycleError) as captured:
                await self.coordinator(
                    transport,
                    initialize=initialize,
                ).run_once()
        self.assertEqual("room_created_commit_failed", captured.exception.code)
        self.assertFalse(initialized)
        self.assertEqual("accept_sent", self.store.snapshot().reservation.phase)

    async def test_post_room_battle_initialization_failure_keeps_team_consumed(self):
        async def fail(_room):
            raise ValueError(PRIVATE_SENTINEL)

        transport = FakeTransport((CHALLENGE, room_message()))
        with self.assertRaises(BlindPoolLifecycleError) as captured:
            await self.coordinator(transport, initialize=fail).run_once()
        self.assertEqual("battle_initialization_failed", captured.exception.code)
        state = self.store.snapshot()
        self.assertIsNone(state.reservation)
        self.assertEqual(1, state.next_index)

    async def test_commit_precedes_battle_construction_request_and_identity_steps(self):
        observed = []

        async def initialize(_room):
            for step in ("battle construction", "request parsing", "identity"):
                state = self.store.snapshot()
                self.assertIsNone(state.reservation)
                self.assertEqual(1, state.next_index)
                observed.append(step)
            return "ready"

        transport = FakeTransport((CHALLENGE, room_message()))
        result = await self.coordinator(
            transport,
            initialize=initialize,
        ).run_once()
        self.assertEqual("ready", result)
        self.assertEqual(
            ["battle construction", "request parsing", "identity"],
            observed,
        )


class TestBlindPoolLifecycleRecovery(BlindPoolLifecycleFixture):
    async def test_startup_release_exception_after_replace_is_proven_safe(self):
        self.store.initialize_or_load()
        reservation = self.store.reserve_next()
        coordinator = self.coordinator(FakeTransport())
        with mock.patch(
            "fp.data.blind_pool.state._fsync_directory",
            side_effect=OSError(PRIVATE_SENTINEL),
        ):
            await coordinator.startup()
        self.assertIsNone(self.store.snapshot().reservation)
        next_reservation = self.store.reserve_next()
        self.assertEqual(reservation.team_id, next_reservation.team_id)

    def test_reconciliation_post_replace_exceptions_are_classified_once(self):
        reservation = self.reserve_accept_sent()
        coordinator = self.coordinator(FakeTransport())
        with mock.patch(
            "fp.data.blind_pool.state._fsync_directory",
            side_effect=OSError(PRIVATE_SENTINEL),
        ):
            committed = coordinator.reconcile_as_room_created(
                reservation.reservation_id
            )
        self.assertEqual(1, committed.next_index)
        self.assertIsNone(committed.reservation)

        second = self.store.reserve_next()
        self.store.mark_accept_sent(second.reservation_id)
        with mock.patch(
            "fp.data.blind_pool.state._fsync_directory",
            side_effect=OSError(PRIVATE_SENTINEL),
        ):
            released = coordinator.reconcile_as_no_room(second.reservation_id)
        self.assertEqual(1, released.next_index)
        self.assertIsNone(released.reservation)

    async def test_two_started_coordinators_cannot_reserve_same_position(self):
        preparation_started = asyncio.Event()
        never = asyncio.Event()

        async def blocked_prepare(_team_id: str) -> None:
            preparation_started.set()
            await never.wait()

        first = self.coordinator(
            FakeTransport((CHALLENGE,)),
            prepare=blocked_prepare,
        )
        second = self.coordinator(FakeTransport((CHALLENGE, room_message())))
        await first.startup()
        await second.startup()
        first_task = asyncio.create_task(first.run_once())
        await asyncio.wait_for(preparation_started.wait(), timeout=1)
        with self.assertRaises(BlindPoolLifecycleError) as startup_error:
            await second.startup()
        self.assertEqual("lifecycle_already_active", startup_error.exception.code)
        self.assertEqual("reserved", self.store.snapshot().reservation.phase)
        with self.assertRaises(BlindPoolLifecycleError) as captured:
            await second.run_once()
        self.assertEqual("lifecycle_already_active", captured.exception.code)
        first_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first_task
        self.assertIsNone(self.store.snapshot().reservation)

    async def test_startup_releases_reserved_and_preserves_next_team(self):
        self.store.initialize_or_load()
        reservation = self.store.reserve_next()
        transport = FakeTransport()
        await self.coordinator(transport).startup()
        state = self.store.snapshot()
        self.assertIsNone(state.reservation)
        self.assertEqual(0, state.next_index)
        next_reservation = self.store.reserve_next()
        self.assertEqual(reservation.team_id, next_reservation.team_id)

    async def test_startup_quarantines_accept_sent_before_listening(self):
        self.reserve_accept_sent()
        transport = FakeTransport((CHALLENGE, room_message()))
        coordinator = self.coordinator(transport)
        with self.assertRaises(BlindPoolReconciliationRequired) as captured:
            await coordinator.run_once()
        self.assertEqual("reconciliation_required", captured.exception.code)
        self.assertEqual(0, transport.receive_calls)
        self.assertEqual("accept_sent", self.store.snapshot().reservation.phase)

    def test_explicit_room_created_reconciliation_consumes_exactly_once(self):
        reservation = self.reserve_accept_sent()
        coordinator = self.coordinator(FakeTransport())
        state = coordinator.reconcile_as_room_created(reservation.reservation_id)
        self.assertEqual(1, state.next_index)
        self.assertIsNone(state.reservation)
        self.assert_error(
            "reservation_not_found",
            lambda: coordinator.reconcile_as_room_created(reservation.reservation_id),
        )

    def test_explicit_no_room_reconciliation_releases_same_team(self):
        reservation = self.reserve_accept_sent()
        coordinator = self.coordinator(FakeTransport())
        state = coordinator.reconcile_as_no_room(reservation.reservation_id)
        self.assertEqual(0, state.next_index)
        self.assertIsNone(state.reservation)
        next_reservation = self.store.reserve_next()
        self.assertEqual(reservation.team_id, next_reservation.team_id)

    def test_wrong_or_stale_reconciliation_identity_is_rejected(self):
        reservation = self.reserve_accept_sent()
        coordinator = self.coordinator(FakeTransport())
        self.assert_error(
            "reservation_identity_mismatch",
            lambda: coordinator.reconcile_as_no_room("f" * 32),
        )
        coordinator.reconcile_as_no_room(reservation.reservation_id)
        self.assert_error(
            "reservation_not_found",
            lambda: coordinator.reconcile_as_no_room(reservation.reservation_id),
        )

    async def test_second_coordinator_is_blocked_by_unresolved_acceptance(self):
        first_transport = FakeTransport((CHALLENGE,), block_when_empty=True)
        first = self.coordinator(first_transport, timeout=2)
        task = asyncio.create_task(first.run_once())
        for _ in range(100):
            await asyncio.sleep(0)
            state = self.store.initialize_or_load()
            if state.reservation is not None and state.reservation.phase == "accept_sent":
                break
        second_transport = FakeTransport((CHALLENGE, room_message()))
        with self.assertRaises(BlindPoolLifecycleError) as captured:
            await self.coordinator(second_transport).run_once()
        self.assertEqual("lifecycle_already_active", captured.exception.code)
        self.assertEqual(0, second_transport.receive_calls)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task


class TestBlindPoolLifecycleStateTransitions(BlindPoolLifecycleFixture):
    def test_mark_accept_sent_changes_only_phase_and_is_not_repeatable(self):
        initial = self.store.initialize_or_load()
        reservation = self.store.reserve_next()
        before = self.store.snapshot()
        marked = self.store.mark_accept_sent(reservation.reservation_id)
        self.assertEqual("accept_sent", marked.reservation.phase)
        self.assertEqual(before.cycle_order, marked.cycle_order)
        self.assertEqual(before.next_index, marked.next_index)
        self.assertEqual(before.last_consumed_id, marked.last_consumed_id)
        self.assertEqual(initial.cycle_number, marked.cycle_number)
        self.assert_error(
            "reservation_phase_transition_invalid",
            lambda: self.store.mark_accept_sent(reservation.reservation_id),
        )

    def test_accept_sent_rejects_ordinary_release_and_phase3_commit(self):
        reservation = self.reserve_accept_sent()
        for operation in (
            self.store.release_reservation,
            self.store.commit_reservation,
        ):
            with self.subTest(operation=operation.__name__):
                self.assert_error(
                    "reservation_phase_transition_invalid",
                    lambda operation=operation: operation(reservation.reservation_id),
                )
        self.assertEqual("accept_sent", self.store.snapshot().reservation.phase)

    def test_room_commit_requires_accept_sent_and_advances_once(self):
        self.store.initialize_or_load()
        reservation = self.store.reserve_next()
        self.assert_error(
            "reservation_phase_transition_invalid",
            lambda: self.store.commit_room_created(reservation.reservation_id),
        )
        self.store.mark_accept_sent(reservation.reservation_id)
        committed = self.store.commit_room_created(reservation.reservation_id)
        self.assertEqual(1, committed.next_index)
        self.assertEqual(reservation.team_id, committed.last_consumed_id)
        self.assert_error(
            "reservation_not_found",
            lambda: self.store.commit_room_created(reservation.reservation_id),
        )

    def test_legacy_reserved_state_remains_readable_and_schema_is_unchanged(self):
        self.store.initialize_or_load()
        reservation = self.store.reserve_next()
        state = self.store.snapshot()
        self.assertEqual(1, state.schema_version)
        self.assertEqual("reserved", state.reservation.phase)
        self.assertEqual(reservation.team_id, state.cycle_order[state.next_index])


class TestBlindPoolLifecyclePrivacy(BlindPoolLifecycleFixture):
    async def test_protocol_models_logs_errors_and_tracebacks_hide_sentinel(self):
        output = io.StringIO()
        handler = logging.StreamHandler(output)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        package_logger = logging.getLogger("fp.data.blind_pool")
        old_level = package_logger.level
        package_logger.setLevel(logging.DEBUG)
        package_logger.addHandler(handler)

        async def fail(_team_id: str) -> None:
            raise ValueError(PRIVATE_SENTINEL)

        transport = FakeTransport((PRIVATE_SENTINEL, CHALLENGE))
        coordinator = self.coordinator(transport, prepare=fail)
        try:
            with self.assertRaises(BlindPoolLifecycleError) as captured:
                await coordinator.run_once()
        finally:
            package_logger.removeHandler(handler)
            package_logger.setLevel(old_level)
        challenge = parse_incoming_challenge(
            CHALLENGE,
            bot_username="Blind Bot",
            required_format="gen9tugs",
        )
        rendered = output.getvalue() + repr(challenge) + repr(coordinator)
        formatted = "".join(
            traceback.format_exception(
                type(captured.exception),
                captured.exception,
                captured.exception.__traceback__,
            )
        )
        self.assertNotIn(PRIVATE_SENTINEL, rendered + formatted)
        self.assertIsNone(captured.exception.__cause__)

    def test_websocket_redaction_hides_challenges_room_init_and_players(self):
        raw = "\n".join(
            (
                "|pm|" + PRIVATE_SENTINEL + "|Blind Bot|/challenge|gen9tugs|||",
                "|pm|" + PRIVATE_SENTINEL + "|Blind Bot|/challenge-broken|bad",
                ">battle-gen9tugs-99|init|battle|title|" + PRIVATE_SENTINEL,
                ">battle-gen9tugs-99|title|" + PRIVATE_SENTINEL,
                ">battle-gen9tugs-99|player|p1|" + PRIVATE_SENTINEL + "|1|",
                ">" + PRIVATE_SENTINEL + "|init|battle|malformed",
                "|player|p1|" + PRIVATE_SENTINEL + "|1|",
            )
        )
        redacted = _redact_received_message(raw)
        self.assertNotIn(PRIVATE_SENTINEL, redacted)

    async def test_acceptance_command_is_not_logged_or_retained(self):
        client = PSWebsocketClient()
        client.username = "Blind Bot"
        client.websocket = mock.AsyncMock()
        challenge = parse_incoming_challenge(
            CHALLENGE,
            bot_username=client.username,
            required_format="gen9tugs",
        )
        with self.assertLogs("fp.websocket_client", level="DEBUG") as captured:
            await client.send_challenge_acceptance(challenge)
        rendered = "\n".join(captured.output)
        self.assertNotIn(challenge.challenger_name, rendered)
        self.assertIsNone(client.last_message)

    def test_reconciliation_logging_never_contains_reservation_identity(self):
        reservation = self.reserve_accept_sent()
        coordinator = self.coordinator(FakeTransport())
        with self.assertLogs("fp.data.blind_pool", level="INFO") as captured:
            coordinator.reconcile_as_no_room(reservation.reservation_id)
        self.assertNotIn(reservation.reservation_id, "\n".join(captured.output))


if __name__ == "__main__":
    unittest.main()
