from __future__ import annotations

import asyncio
import copy
import json
import logging
import pickle
import traceback
import unittest
from unittest import mock

from fp.data.blind_pool import (
    BlindChallengeEndEvent,
    BlindChallengeRoomBinding,
    BlindChallengeToken,
    BlindExactChallengeProtocol,
    BlindPoolLifecycleCoordinator,
    BlindPoolLifecycleError,
    BlindPoolReconciliationRequired,
    BlindPoolValidationError,
    parse_challenge_end,
    parse_challenge_room_binding,
    parse_exact_challenge_offer,
    parse_private_challenge_event,
)
from fp.websocket_client import PSWebsocketClient, _redact_received_message
from tests.test_blind_pool_bag import BlindPoolBagFixture
from tests.test_blind_pool_lifecycle import (
    CHALLENGE,
    FakeTransport,
    BlindPoolLifecycleFixture,
    room_message,
)


TOKEN = "a" * 32
TOKEN_TWO = "b" * 32
UNKNOWN_TOKEN = "c" * 32
TOKEN_SENTINEL = "deadc0de" * 4
OFFER = "|tugschallenge|syntheticopponent|gen9tugs|" + TOKEN


def offer(
    token: str = TOKEN,
    *,
    challenger: str = "syntheticopponent",
    format_id: str = "gen9tugs",
) -> str:
    return "|tugschallenge|{}|{}|{}".format(challenger, format_id, token)


def binding(token: str, room_id: str) -> str:
    return "|tugschallengeroom|{}|{}".format(token, room_id)


class _Websocket:
    def __init__(self, messages=(), *, send_error: BaseException | None = None):
        self.messages = list(messages)
        self.sent = []
        self.send_error = send_error

    async def recv(self):
        return self.messages.pop(0)

    async def send(self, message):
        if self.send_error is not None:
            raise self.send_error
        self.sent.append(message)


class TokenModelAndParserTests(unittest.TestCase):
    def test_token_is_exact_immutable_safe_and_timing_safe(self):
        token = BlindChallengeToken(TOKEN)
        same = BlindChallengeToken(TOKEN)
        different = BlindChallengeToken(TOKEN_TWO)
        self.assertEqual(TOKEN, token.wire_value())
        self.assertTrue(token.matches(same))
        self.assertEqual(token, same)
        self.assertFalse(token.matches(different))
        self.assertNotIn(TOKEN, repr(token) + str(token))
        with self.assertRaises(TypeError):
            hash(token)
        with self.assertRaises(AttributeError):
            token._value = TOKEN_TWO
        with self.assertRaises(TypeError):
            pickle.dumps(token)
        with self.assertRaises(TypeError):
            json.dumps(token)

    def test_token_rejects_every_noncanonical_form_without_echo(self):
        invalid = (
            "",
            "a" * 31,
            "a" * 33,
            "A" * 32,
            "g" * 32,
            " " + TOKEN,
            TOKEN + " ",
            TOKEN[:16] + " " + TOKEN[17:],
            "0x" + TOKEN,
            TOKEN + "suffix",
            b"a" * 32,
            None,
            7,
        )
        for value in invalid:
            with self.subTest(
                value_type=type(value).__name__,
                length=len(value) if hasattr(value, "__len__") else None,
            ):
                with self.assertRaises(BlindPoolValidationError) as caught:
                    BlindChallengeToken(value)
                self.assertEqual("challenge_token_invalid", caught.exception.code)
                if str(value):
                    self.assertNotIn(str(value), str(caught.exception))

    def test_valid_private_events_return_safe_immutable_models(self):
        challenge = parse_exact_challenge_offer(OFFER)
        ended = parse_challenge_end("|tugschallengeend|" + TOKEN)
        bound = parse_challenge_room_binding(binding(TOKEN, "battle-gen9tugs-401"))
        self.assertEqual("syntheticopponent", challenge.challenger_id)
        self.assertEqual("gen9tugs", challenge.format_id)
        self.assertEqual("tugs_exact", challenge.source)
        self.assertTrue(challenge.challenge_token.matches(BlindChallengeToken(TOKEN)))
        self.assertIs(challenge.challenge_token, challenge.deduplication_identity)
        self.assertIsInstance(ended, BlindChallengeEndEvent)
        self.assertIsInstance(bound, BlindChallengeRoomBinding)
        self.assertEqual("battle-gen9tugs-401", bound.room_id)
        rendered = repr(challenge) + repr(ended) + repr(bound)
        self.assertNotIn(TOKEN, rendered)
        for value in (challenge, ended, bound):
            with self.subTest(model=type(value).__name__):
                with self.assertRaises(TypeError):
                    pickle.dumps(value)

    def test_private_event_parsers_reject_exact_grammar_violations_safely(self):
        cases = (
            "|tugschallenge|syntheticopponent|gen9tugs",
            OFFER + "|extra",
            "|tugschallenge|SyntheticOpponent|gen9tugs|" + TOKEN,
            "|tugschallenge|synthetic_opponent|gen9tugs|" + TOKEN,
            "|tugschallenge|syntheticopponent|gen9-tugs|" + TOKEN,
            "|tugschallenge|syntheticopponent|gen9tugs|" + TOKEN.upper(),
            "|tugschallenge|syntheticopponent|gen9tugs| " + TOKEN,
            "|tugschallengeend|" + TOKEN + "|extra",
            "|tugschallengeend|" + TOKEN.upper(),
            "|tugschallengeroom|{}|not-a-room".format(TOKEN),
            "|tugschallengeroom|{}|battle-gen9tugs-0".format(TOKEN),
            binding(TOKEN, "battle-gen9tugs-401") + "|extra",
            binding(TOKEN.upper(), "battle-gen9tugs-401"),
        )
        for message in cases:
            with self.subTest(event=message.split("|", 2)[1]):
                with self.assertRaises(BlindPoolLifecycleError) as caught:
                    parse_private_challenge_event(message)
                rendered = str(caught.exception) + "".join(
                    traceback.format_exception(
                        type(caught.exception),
                        caught.exception,
                        caught.exception.__traceback__,
                    )
                )
                self.assertNotIn(TOKEN, rendered)
                self.assertNotIn(message, rendered)


class StateSchemaV2Tests(BlindPoolBagFixture):
    def test_raw_and_exact_reservations_round_trip_under_schema_v2(self):
        raw_store = self.store(reservation_ids=["1" * 32, "2" * 32])
        raw_store.initialize_or_load()
        raw = raw_store.reserve_next()
        raw_document = self.state_document()
        self.assertEqual(2, raw_document["schema_version"])
        self.assertIsNone(raw_document["reservation"]["challenge_token"])
        self.assertIsNone(raw.challenge_token)
        marked_raw = raw_store.mark_accept_sent(raw.reservation_id)
        self.assertIsNone(marked_raw.reservation.challenge_token)
        raw_store.reconcile_accept_sent_as_no_room(raw.reservation_id)

        exact = BlindChallengeToken(TOKEN)
        reserved = raw_store.reserve_next(exact)
        reloaded = raw_store.snapshot().reservation
        self.assertTrue(reloaded.challenge_token.matches(exact))
        self.assertEqual(TOKEN, self.state_document()["reservation"]["challenge_token"])
        self.assertNotIn(TOKEN, repr(reserved) + repr(reloaded))
        with self.assertRaises(TypeError):
            pickle.dumps(reserved)

    def test_exact_reserve_has_no_intermediate_null_token_reservation(self):
        store = self.store(reservation_ids=["1" * 32])
        observed = []
        from fp.data.blind_pool import bag as bag_module

        original = bag_module.write_blind_pool_bag_state_atomic

        def observe(config, state, registry):
            if state.reservation is not None:
                observed.append(state.reservation.challenge_token)
            return original(config, state, registry)

        with mock.patch.object(
            bag_module,
            "write_blind_pool_bag_state_atomic",
            side_effect=observe,
        ):
            store.initialize_or_load()
            reservation = store.reserve_next(BlindChallengeToken(TOKEN))
        self.assertEqual(1, len(observed))
        self.assertTrue(observed[0].matches(reservation.challenge_token))

    def test_mark_accept_sent_preserves_and_checks_exact_token(self):
        store = self.store(reservation_ids=["1" * 32])
        store.initialize_or_load()
        reservation = store.reserve_next(BlindChallengeToken(TOKEN))
        before = self.state_document()["reservation"]["challenge_token"]
        with self.assertRaises(BlindPoolValidationError) as caught:
            store.mark_accept_sent(
                reservation.reservation_id,
                BlindChallengeToken(TOKEN_TWO),
            )
        self.assertEqual("reservation_challenge_token_mismatch", caught.exception.code)
        marked = store.mark_accept_sent(
            reservation.reservation_id,
            BlindChallengeToken(TOKEN),
        )
        self.assertTrue(
            marked.reservation.challenge_token.matches(BlindChallengeToken(TOKEN))
        )
        self.assertEqual(
            before, self.state_document()["reservation"]["challenge_token"]
        )

    def test_schema_v1_is_rejected_without_mutating_bytes(self):
        store = self.store()
        store.initialize_or_load()
        document = self.state_document()
        document["schema_version"] = 1
        raw = (json.dumps(document, separators=(",", ":")) + "\n").encode()
        self.state_path.write_bytes(raw)
        with self.assertRaises(BlindPoolValidationError) as caught:
            store.snapshot()
        self.assertEqual("state_schema_unsupported", caught.exception.code)
        self.assertEqual(raw, self.state_path.read_bytes())

    def test_future_schema_and_missing_token_field_are_rejected(self):
        store = self.store(reservation_ids=["1" * 32])
        store.initialize_or_load()
        future = self.state_document()
        future["schema_version"] = 3
        self.write_state_document(future)
        with self.assertRaises(BlindPoolValidationError) as caught:
            store.snapshot()
        self.assertEqual("state_schema_unsupported", caught.exception.code)

        self.state_path.unlink()
        store.initialize_or_load()
        store.reserve_next()
        missing = self.state_document()
        del missing["reservation"]["challenge_token"]
        self.write_state_document(missing)
        with self.assertRaises(BlindPoolValidationError) as caught:
            store.snapshot()
        self.assertEqual("state_missing_required_field", caught.exception.code)

    def test_invalid_persisted_tokens_are_rejected_without_echo(self):
        store = self.store(reservation_ids=["1" * 32])
        store.initialize_or_load()
        store.reserve_next(BlindChallengeToken(TOKEN))
        valid = self.state_document()
        for value in (
            TOKEN.upper(),
            TOKEN[:-1],
            "g" * 32,
            " " + TOKEN,
            TOKEN + " ",
            7,
            [],
            {},
        ):
            with self.subTest(value_type=type(value).__name__):
                document = copy.deepcopy(valid)
                document["reservation"]["challenge_token"] = value
                self.write_state_document(document)
                with self.assertRaises(BlindPoolValidationError) as caught:
                    store.snapshot()
                self.assertEqual("state_challenge_token_invalid", caught.exception.code)
                self.assertNotIn(str(value), str(caught.exception))


class WebsocketExactProtocolTests(unittest.TestCase):
    def client(self, messages=(), *, send_error=None):
        client = PSWebsocketClient()
        client.username = "Blind Bot"
        client.websocket = _Websocket(messages, send_error=send_error)
        client.last_message = None
        return client

    def test_capability_and_exact_accept_wire_are_distinct_and_exact(self):
        client = self.client()
        protocol = BlindExactChallengeProtocol.from_transport(client)
        self.assertNotIn(TOKEN, repr(protocol))
        with self.assertRaises(TypeError):
            pickle.dumps(protocol)
        challenge = parse_exact_challenge_offer(OFFER)
        asyncio.run(client.enable_challenge_tokens())
        asyncio.run(client.disable_challenge_tokens())
        with self.assertLogs("fp.websocket_client", logging.DEBUG) as captured:
            asyncio.run(client.send_exact_challenge_acceptance(challenge))
        self.assertEqual(
            [
                "|/tugschallengetokens on",
                "|/tugschallengetokens off",
                "|/accept syntheticopponent, " + TOKEN,
            ],
            client.websocket.sent,
        )
        self.assertIsNone(client.last_message)
        self.assertNotIn(TOKEN, "\n".join(captured.output))

    def test_legacy_accept_wire_remains_tokenless(self):
        client = self.client()
        legacy = parse_exact_challenge_offer(OFFER)
        legacy = type(legacy)(
            challenger_id=legacy.challenger_id,
            challenger_name="Synthetic Opponent",
            format_id=legacy.format_id,
            source="pm",
        )
        asyncio.run(client.send_challenge_acceptance(legacy))
        self.assertEqual(["|/accept Synthetic Opponent"], client.websocket.sent)
        with self.assertRaises(ValueError):
            asyncio.run(client.send_exact_challenge_acceptance(legacy))
        self.assertEqual(["|/accept Synthetic Opponent"], client.websocket.sent)

    def test_exact_accept_requires_token_and_sanitizes_transport_failure(self):
        client = self.client(send_error=OSError(TOKEN_SENTINEL))
        challenge = parse_exact_challenge_offer(offer(TOKEN_SENTINEL))
        with self.assertRaises(ValueError) as caught:
            asyncio.run(client.send_exact_challenge_acceptance(challenge))
        rendered = str(caught.exception) + repr(caught.exception)
        self.assertNotIn(TOKEN_SENTINEL, rendered)
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        self.assertIsNone(client.last_message)

    def test_inbound_private_events_are_redacted_before_parsing(self):
        prefixes = (
            "|tugschallenge|",
            "|tugschallengeend|",
            "|tugschallengeroom|",
        )
        suffixes = (
            TOKEN_SENTINEL,
            "malformed-" + TOKEN_SENTINEL,
            TOKEN_SENTINEL + "|extra",
        )
        for prefix in prefixes:
            for suffix in suffixes:
                with self.subTest(prefix=prefix, malformed="malformed" in suffix):
                    raw = prefix + suffix
                    redacted = _redact_received_message(raw)
                    self.assertNotIn(TOKEN_SENTINEL, redacted)
                    self.assertEqual(prefix + "<redacted>", redacted)
                    client = self.client((raw,))
                    with self.assertLogs(
                        "fp.websocket_client", logging.DEBUG
                    ) as captured:
                        self.assertEqual(raw, asyncio.run(client.receive_message()))
                    self.assertNotIn(TOKEN_SENTINEL, "\n".join(captured.output))


class ExactLifecycleTests(BlindPoolLifecycleFixture):
    def coordinator_exact(
        self,
        transport: FakeTransport,
        *,
        prepare=None,
        initialize=None,
        timeout: float = 0.1,
    ) -> BlindPoolLifecycleCoordinator:
        async def successful_prepare(_team_id):
            return None

        return BlindPoolLifecycleCoordinator(
            self.store,
            transport,
            prepare or successful_prepare,
            initialize_battle=initialize,
            room_timeout_seconds=timeout,
            exact_protocol=BlindExactChallengeProtocol.from_transport(transport),
        )

    async def test_stale_matching_room_and_new_room_have_no_authority_before_binding(
        self,
    ):
        stale = "battle-gen9tugs-501"
        new = "battle-gen9tugs-502"
        observed = []

        def receive_observer(message):
            if message.startswith(">"):
                observed.append(self.store.snapshot().reservation.phase)

        transport = FakeTransport(
            (
                OFFER,
                room_message(stale),
                room_message(new),
                binding(TOKEN, new),
            ),
            receive_observer=receive_observer,
        )
        with mock.patch.object(
            self.store,
            "commit_room_created",
            wraps=self.store.commit_room_created,
        ) as commit:
            room = await self.coordinator_exact(transport).run_once()
        self.assertEqual(new, room.room_id)
        self.assertEqual(["accept_sent", "accept_sent"], observed)
        self.assertEqual(1, commit.call_count)
        self.assertEqual(["enable"], transport.controls)

    async def test_capability_is_enabled_before_first_authoritative_offer_read(self):
        room_id = "battle-gen9tugs-500"

        def observe(message):
            if message == OFFER:
                self.assertEqual(["enable"], transport.controls)

        transport = FakeTransport(
            (OFFER, room_message(room_id), binding(TOKEN, room_id)),
            receive_observer=observe,
        )
        await self.coordinator_exact(transport).run_once()
        self.assertEqual(["enable"], transport.controls)

    async def test_room_first_and_first_request_commit_once_after_binding(self):
        room_id = "battle-gen9tugs-503"
        request = '|request|{"rqid":9}'
        payload = room_message(room_id) + "\n" + request
        transport = FakeTransport((OFFER, payload, binding(TOKEN, room_id)))
        with mock.patch.object(
            self.store,
            "commit_room_created",
            wraps=self.store.commit_room_created,
        ) as commit:
            room = await self.coordinator_exact(transport).run_once()
        self.assertEqual(room_id, room.room_id)
        self.assertEqual(1, commit.call_count)
        handed_off = transport.handoffs[0][1]
        self.assertEqual(1, handed_off.count(request))
        self.assertFalse(any("tugschallenge" in line for line in handed_off))

    async def test_binding_first_waits_for_the_bound_room(self):
        room_id = "battle-gen9tugs-504"
        request = '|request|{"rqid":10}'
        transport = FakeTransport(
            (
                OFFER,
                binding(TOKEN, room_id),
                room_message(room_id) + "\n" + request,
            )
        )
        room = await self.coordinator_exact(transport).run_once()
        self.assertEqual(room_id, room.room_id)
        self.assertEqual(1, transport.handoffs[0][1].count(request))

    async def test_no_binding_never_falls_back_to_matching_room(self):
        transport = FakeTransport(
            (OFFER, room_message("battle-gen9tugs-505")),
            block_when_empty=True,
        )
        with self.assertRaises(BlindPoolReconciliationRequired) as caught:
            await self.coordinator_exact(transport, timeout=0.01).run_once()
        self.assertEqual("exact_room_binding_timed_out", caught.exception.code)
        self.assertEqual("accept_sent", self.store.snapshot().reservation.phase)

    async def test_incomplete_bound_room_times_out_without_commit(self):
        room_id = "battle-gen9tugs-524"
        incomplete = "\n".join(
            (
                ">" + room_id,
                "|init|battle",
                "|player|p1|Synthetic Opponent|1|",
            )
        )
        transport = FakeTransport(
            (OFFER, binding(TOKEN, room_id), incomplete),
            block_when_empty=True,
        )
        with self.assertRaises(BlindPoolReconciliationRequired) as caught:
            await self.coordinator_exact(transport, timeout=0.01).run_once()
        self.assertEqual("exact_room_binding_timed_out", caught.exception.code)
        state = self.store.snapshot()
        self.assertEqual("accept_sent", state.reservation.phase)
        self.assertEqual(0, state.next_index)

    async def test_unknown_then_current_binding_selects_only_current(self):
        room_id = "battle-gen9tugs-506"
        transport = FakeTransport(
            (
                OFFER,
                binding(UNKNOWN_TOKEN, room_id),
                room_message(room_id),
                binding(TOKEN, room_id),
            )
        )
        room = await self.coordinator_exact(transport).run_once()
        self.assertEqual(room_id, room.room_id)

    async def test_duplicate_identical_binding_is_idempotent(self):
        room_id = "battle-gen9tugs-507"
        payload = "\n".join(
            (room_message(room_id), binding(TOKEN, room_id), binding(TOKEN, room_id))
        )
        transport = FakeTransport((OFFER, payload))
        with mock.patch.object(
            self.store,
            "commit_room_created",
            wraps=self.store.commit_room_created,
        ) as commit:
            await self.coordinator_exact(transport).run_once()
        self.assertEqual(1, commit.call_count)

    async def test_conflicting_binding_quarantines_without_commit_or_release(self):
        transport = FakeTransport(
            (
                OFFER,
                "\n".join(
                    (
                        binding(TOKEN, "battle-gen9tugs-508"),
                        binding(TOKEN, "battle-gen9tugs-509"),
                    )
                ),
            )
        )
        with mock.patch.object(
            self.store,
            "commit_room_created",
            wraps=self.store.commit_room_created,
        ) as commit, mock.patch.object(
            self.store,
            "release_reservation",
            wraps=self.store.release_reservation,
        ) as release, self.assertRaises(BlindPoolReconciliationRequired) as caught:
            await self.coordinator_exact(transport).run_once()
        self.assertEqual("exact_room_binding_conflict", caught.exception.code)
        commit.assert_not_called()
        release.assert_not_called()
        self.assertEqual("accept_sent", self.store.snapshot().reservation.phase)

    async def test_wrong_format_bound_room_fails_closed(self):
        transport = FakeTransport((OFFER, binding(TOKEN, "battle-gen9nationaldex-510")))
        with self.assertRaises(BlindPoolReconciliationRequired) as caught:
            await self.coordinator_exact(transport).run_once()
        self.assertEqual("exact_bound_room_invalid", caught.exception.code)

    async def test_wrong_challenger_bound_room_fails_closed(self):
        room_id = "battle-gen9tugs-511"
        wrong = room_message(room_id).replace(
            "Synthetic Opponent", "Different Opponent"
        )
        transport = FakeTransport((OFFER, wrong, binding(TOKEN, room_id)))
        with self.assertRaises(BlindPoolReconciliationRequired) as caught:
            await self.coordinator_exact(transport).run_once()
        self.assertEqual("exact_bound_room_invalid", caught.exception.code)

    async def test_missing_bot_bound_room_fails_closed(self):
        room_id = "battle-gen9tugs-512"
        wrong = room_message(room_id).replace("Blind Bot", "Different Bot")
        transport = FakeTransport((OFFER, wrong, binding(TOKEN, room_id)))
        with self.assertRaises(BlindPoolReconciliationRequired) as caught:
            await self.coordinator_exact(transport).run_once()
        self.assertEqual("exact_bound_room_invalid", caught.exception.code)

    async def test_preaccept_room_binding_is_rejected(self):
        room_id = "battle-gen9tugs-513"
        transport = FakeTransport(
            (room_message(room_id), OFFER, binding(TOKEN, room_id))
        )
        with self.assertRaises(BlindPoolReconciliationRequired) as caught:
            await self.coordinator_exact(transport).run_once()
        self.assertEqual("exact_bound_room_preaccept", caught.exception.code)

    async def test_duplicate_offer_creates_one_attempt(self):
        room_id = "battle-gen9tugs-514"
        transport = FakeTransport(
            ("\n".join((OFFER, OFFER)), room_message(room_id), binding(TOKEN, room_id))
        )
        with mock.patch.object(
            self.store,
            "reserve_next",
            wraps=self.store.reserve_next,
        ) as reserve:
            await self.coordinator_exact(transport).run_once()
        self.assertEqual(1, reserve.call_count)
        self.assertEqual(1, len(transport.sent))

    async def test_malformed_private_events_are_ignored_without_crashing_loop(self):
        room_id = "battle-gen9tugs-519"
        malformed = "|tugschallenge|syntheticopponent|gen9tugs|" + TOKEN.upper()
        transport = FakeTransport(
            (malformed, OFFER, room_message(room_id), binding(TOKEN, room_id))
        )
        with self.assertLogs(
            "fp.data.blind_pool.lifecycle", logging.WARNING
        ) as captured:
            room = await self.coordinator_exact(transport).run_once()
        self.assertEqual(room_id, room.room_id)
        self.assertNotIn(TOKEN.upper(), "\n".join(captured.output))

    async def test_offer_metadata_conflict_fails_before_reservation(self):
        conflict = offer(TOKEN, challenger="differentopponent")
        transport = FakeTransport(("\n".join((OFFER, conflict)),))
        with self.assertRaises(BlindPoolLifecycleError) as caught:
            await self.coordinator_exact(transport).run_once()
        self.assertEqual("exact_challenge_metadata_conflict", caught.exception.code)
        self.assertIsNone(self.store.snapshot().reservation)
        self.assertEqual([], transport.sent)

    async def test_offer_end_before_reservation_tombstones_first_offer(self):
        room_id = "battle-gen9tugs-515"
        payload = "\n".join(
            (
                OFFER,
                "|tugschallengeend|" + TOKEN,
                offer(TOKEN_TWO),
            )
        )
        transport = FakeTransport(
            (payload, room_message(room_id), binding(TOKEN_TWO, room_id))
        )
        await self.coordinator_exact(transport).run_once()
        self.assertEqual(1, len(transport.sent))
        self.assertTrue(
            transport.sent[0].challenge_token.matches(BlindChallengeToken(TOKEN_TWO))
        )

    async def test_wrong_format_offer_cannot_drive_reservation(self):
        room_id = "battle-gen9tugs-520"
        transport = FakeTransport(
            (
                offer(TOKEN, format_id="gen9nationaldex"),
                offer(TOKEN_TWO),
                room_message(room_id),
                binding(TOKEN_TWO, room_id),
            )
        )
        await self.coordinator_exact(transport).run_once()
        self.assertEqual(1, len(transport.sent))
        self.assertTrue(
            transport.sent[0].challenge_token.matches(BlindChallengeToken(TOKEN_TWO))
        )

    async def test_current_end_after_accept_is_quarantined(self):
        transport = FakeTransport((OFFER, "|tugschallengeend|" + TOKEN))
        with mock.patch.object(
            self.store,
            "release_reservation",
            wraps=self.store.release_reservation,
        ) as release, self.assertRaises(BlindPoolReconciliationRequired) as caught:
            await self.coordinator_exact(transport).run_once()
        self.assertEqual("exact_challenge_ended_after_accept", caught.exception.code)
        release.assert_not_called()
        self.assertEqual("accept_sent", self.store.snapshot().reservation.phase)

    async def test_exact_accept_failure_keeps_durable_token_and_sanitizes(self):
        transport = FakeTransport(
            (OFFER,),
            send_error=OSError(TOKEN_SENTINEL),
        )
        with self.assertRaises(BlindPoolReconciliationRequired) as caught:
            await self.coordinator_exact(transport).run_once()
        self.assertEqual("acceptance_transmission_ambiguous", caught.exception.code)
        reservation = self.store.snapshot().reservation
        self.assertEqual("accept_sent", reservation.phase)
        self.assertTrue(reservation.challenge_token.matches(BlindChallengeToken(TOKEN)))
        rendered = str(caught.exception) + repr(caught.exception)
        self.assertNotIn(TOKEN_SENTINEL, rendered)
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)

    async def test_sequential_attempts_do_not_reuse_token_state(self):
        room_one = "battle-gen9tugs-516"
        room_two = "battle-gen9tugs-517"
        transport = FakeTransport(
            (
                OFFER,
                room_message(room_one),
                binding(TOKEN, room_one),
                offer(TOKEN_TWO),
                binding(TOKEN, room_two),
                room_message(room_two),
                binding(TOKEN_TWO, room_two),
            )
        )
        coordinator = self.coordinator_exact(transport)
        first = await coordinator.run_once()
        second = await coordinator.run_once()
        self.assertEqual(room_one, first.room_id)
        self.assertEqual(room_two, second.room_id)
        self.assertEqual(2, self.store.snapshot().next_index)

    async def test_resolved_offer_end_and_binding_cannot_create_a_new_attempt(self):
        first_room = "battle-gen9tugs-521"
        second_room = "battle-gen9tugs-522"
        transport = FakeTransport(
            (
                OFFER,
                room_message(first_room),
                binding(TOKEN, first_room),
                OFFER,
                "|tugschallengeend|" + TOKEN,
                binding(TOKEN, first_room),
                offer(TOKEN_TWO),
                room_message(second_room),
                binding(TOKEN_TWO, second_room),
            )
        )
        coordinator = self.coordinator_exact(transport)
        await coordinator.run_once()
        await coordinator.run_once()
        self.assertEqual(2, len(transport.sent))
        self.assertEqual(2, self.store.snapshot().next_index)

    def test_exact_tracking_bounds_evict_oldest_entries_without_exposure(self):
        coordinator = self.coordinator_exact(FakeTransport())
        tokens = tuple("{:032x}".format(index) for index in range(1, 98))
        for wire_token in tokens:
            challenge = parse_exact_challenge_offer(offer(wire_token))
            self.assertTrue(coordinator._record_exact_offer(challenge))
            coordinator._remember_resolved_token(challenge.challenge_token)

        self.assertEqual(96, len(coordinator._exact_offer_metadata))
        self.assertEqual(64, len(coordinator._resolved_challenge_tokens))
        oldest = BlindChallengeToken(tokens[0])
        newest = BlindChallengeToken(tokens[-1])
        self.assertFalse(
            any(token.matches(oldest) for token, _ in coordinator._exact_offer_metadata)
        )
        self.assertFalse(
            any(
                token.matches(oldest)
                for token in coordinator._resolved_challenge_tokens
            )
        )
        self.assertTrue(
            any(
                token.matches(newest)
                for token in coordinator._resolved_challenge_tokens
            )
        )
        self.assertNotIn(tokens[0], repr(coordinator))
        self.assertNotIn(tokens[-1], repr(coordinator))

    def test_pending_exact_offer_queue_fails_closed_at_fixed_bound(self):
        coordinator = self.coordinator_exact(FakeTransport())
        challenges = tuple(
            parse_exact_challenge_offer(offer("{:032x}".format(index)))
            for index in range(1, 34)
        )
        for challenge in challenges[:32]:
            coordinator._queue_exact_challenge(challenge)
        with self.assertRaises(BlindPoolLifecycleError) as caught:
            coordinator._queue_exact_challenge(challenges[32])
        self.assertEqual("exact_challenge_queue_full", caught.exception.code)
        self.assertEqual(32, len(coordinator._pending_exact_challenges))

    async def test_preaccept_failure_releases_token_and_same_team_is_next(self):
        room_id = "battle-gen9tugs-523"
        attempts = []

        async def prepare(team_id):
            attempts.append(team_id)
            if len(attempts) == 1:
                raise RuntimeError(TOKEN_SENTINEL)

        transport = FakeTransport(
            (
                OFFER,
                offer(TOKEN_TWO),
                room_message(room_id),
                binding(TOKEN_TWO, room_id),
            )
        )
        coordinator = self.coordinator_exact(transport, prepare=prepare)
        with self.assertRaises(BlindPoolLifecycleError):
            await coordinator.run_once()
        self.assertIsNone(self.store.snapshot().reservation)
        await coordinator.run_once()
        self.assertEqual([self.active_ids[0], self.active_ids[0]], attempts)
        self.assertEqual(1, self.store.snapshot().next_index)

    async def test_reserved_restart_releases_exact_token_without_enabling(self):
        self.store.initialize_or_load()
        reservation = self.store.reserve_next(BlindChallengeToken(TOKEN))
        transport = FakeTransport()
        await self.coordinator_exact(transport).startup()
        self.assertIsNone(self.store.snapshot().reservation)
        self.assertEqual([], transport.controls)
        next_reservation = self.store.reserve_next()
        self.assertEqual(reservation.team_id, next_reservation.team_id)

    async def test_accept_sent_restart_quarantines_exact_token_without_enabling(self):
        self.store.initialize_or_load()
        reservation = self.store.reserve_next(BlindChallengeToken(TOKEN))
        self.store.mark_accept_sent(
            reservation.reservation_id,
            BlindChallengeToken(TOKEN),
        )
        transport = FakeTransport()
        with self.assertRaises(BlindPoolReconciliationRequired) as caught:
            await self.coordinator_exact(transport).startup()
        self.assertEqual("reconciliation_required", caught.exception.code)
        current = self.store.snapshot().reservation
        self.assertTrue(current.challenge_token.matches(BlindChallengeToken(TOKEN)))
        self.assertEqual([], transport.controls)

    async def test_pm_only_and_stale_room_cannot_drive_exact_mode(self):
        transport = FakeTransport((CHALLENGE, room_message("battle-gen9tugs-518")))
        coordinator = self.coordinator_exact(transport)
        with self.assertRaises(BlindPoolLifecycleError) as caught:
            await coordinator.run_once()
        self.assertEqual("exact_challenge_receive_failed", caught.exception.code)
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        self.assertIsNone(self.store.snapshot().reservation)
        self.assertEqual([], transport.sent)
        self.assertEqual(["enable"], transport.controls)


if __name__ == "__main__":
    unittest.main()
