from __future__ import annotations

import asyncio
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import logging
import pickle
import unittest
from unittest import mock

from fp.battle.state import Battle
from fp.data.blind_pool.bag import BlindPoolBagStore
from fp.data.blind_pool.canonical_runtime import (
    CanonicalBlindRuntime,
    CanonicalBlindRuntimeSession,
)
from fp.data.blind_pool.errors import (
    BlindPoolLifecycleError,
    BlindPoolReconciliationRequired,
    BlindPoolValidationError,
)
from fp.data.blind_pool.lifecycle import BlindExactChallengeProtocol
from fp.data.blind_pool.models import (
    BlindChallengeToken,
    BlindPoolBattleRoom,
    BlindPoolConfig,
    BlindPoolStateConfig,
)
from tests.test_blind_canonical_registry import (
    SyntheticCanonicalFixture,
    synthetic_sidecar,
)
from tests.test_blind_pool_lifecycle import (
    EXACT_CHALLENGE,
    EXACT_TOKEN,
    FakeTransport,
    IdentityRandom,
    exact_room_message,
)


PRIVATE_PACKED = "PHASE5C2-SYNTHETIC-PACKED-SENTINEL"
PRIVATE_ERROR = "PHASE5C2-SYNTHETIC-ERROR-SENTINEL"
PRIVATE_SIDECAR = "PHASE5C2-SYNTHETIC-SIDECAR-SENTINEL"
SECOND_EXACT_TOKEN = "b" * 32
SECOND_EXACT_CHALLENGE = (
    "|tugschallenge|syntheticopponent|gen9tugs|" + SECOND_EXACT_TOKEN
)


def battle_compatible_sidecar(team_id):
    document = synthetic_sidecar(team_id, move_count=4)
    species = ("pikachu", "rattata", "caterpie", "weedle", "pidgey", "spearow")
    for record, species_id in zip(document["sets"], species):
        record["name"] = PRIVATE_SIDECAR
        record["species"] = PRIVATE_SIDECAR
        record["species_id"] = species_id
        record["nature"] = PRIVATE_SIDECAR
        record["nature_id"] = "jolly"
    document["sets"][0]["evs"] = {
        "hp": 4,
        "atk": 8,
        "def": 12,
        "spa": 16,
        "spd": 20,
        "spe": 24,
    }
    document["sets"][0]["ivs"] = {
        "hp": 31,
        "atk": 30,
        "def": 29,
        "spa": 28,
        "spd": 27,
        "spe": 26,
    }
    return document


class RuntimeFixture(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fixture = SyntheticCanonicalFixture()
        self.addCleanup(self.fixture.close)
        self.first_directory, _ = self.fixture.add_artifact(
            "BL-001-v1",
            packed=PRIVATE_PACKED.encode(),
            sidecar_document=battle_compatible_sidecar("BL-001-v1"),
        )
        self.fixture.add_artifact(
            "BL-002-v1",
            packed=(PRIVATE_PACKED + "-TWO").encode(),
            sidecar_document=battle_compatible_sidecar("BL-002-v1"),
        )
        self.fixture.add_artifact(
            "BL-003-v1",
            active=False,
            packed=(PRIVATE_PACKED + "-INACTIVE").encode(),
            sidecar_document=battle_compatible_sidecar("BL-003-v1"),
        )
        self.registry = self.fixture.load()
        self.state_directory = self.fixture.private_root / "state"
        self.state_directory.mkdir()
        self.state_path = self.state_directory / "bag.json"
        self.state_config = BlindPoolStateConfig(
            BlindPoolConfig(
                self.fixture.private_root.resolve(),
                self.fixture.registry_path.resolve(),
            ),
            self.state_path.resolve(),
        )
        self.identifiers = iter(format(index, "032x") for index in range(1, 100))

    def runtime(
        self,
        transport,
        submit_team,
        initialize_battle,
        *,
        registry=None,
        state_config=None,
    ):
        return CanonicalBlindRuntime(
            state_config or self.state_config,
            registry or self.registry,
            transport,
            submit_team,
            initialize_battle,
            exact_protocol=BlindExactChallengeProtocol.from_transport(transport),
            random_source=IdentityRandom(),
            reservation_id_factory=lambda: next(self.identifiers),
            lock_timeout_seconds=1,
            room_timeout_seconds=0.1,
        )

    def state_document(self):
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def assert_safe_error(self, error, code):
        self.assertEqual(code, error.code)
        rendered = str(error) + repr(error)
        for sentinel in (
            PRIVATE_PACKED,
            PRIVATE_ERROR,
            PRIVATE_SIDECAR,
            str(self.fixture.private_root),
        ):
            self.assertNotIn(sentinel, rendered)
        self.assertIsNone(error.__cause__)
        self.assertIsNone(error.__context__)

    async def successful_initializer(self, room, projection):
        return room, projection


class CanonicalRuntimeSuccessTests(RuntimeFixture):
    async def test_success_submits_exact_packed_then_initializes_fresh_projection(self):
        submitted = []
        initialized = []
        transport = FakeTransport((EXACT_CHALLENGE, exact_room_message()))

        async def submit_team(packed):
            submitted.append(packed)

        async def initialize(room, projection):
            initialized.append((room, projection))
            return room

        runtime = self.runtime(transport, submit_team, initialize)
        rendered = repr(runtime) + str(runtime)
        self.assertNotIn(PRIVATE_PACKED, rendered)
        self.assertNotIn(str(self.fixture.private_root), rendered)
        self.assertNotIn(self.registry.registry_fingerprint, rendered)
        with (
            mock.patch(
                "fp.teams.team_converter.export_to_packed",
                side_effect=AssertionError("legacy packer called"),
            ),
            mock.patch(
                "fp.teams.team_converter.json_to_packed",
                side_effect=AssertionError("legacy packer called"),
            ),
        ):
            result = await runtime.run_once()
        self.assertEqual("battle-gen9tugs-401", result.room_id)
        self.assertEqual([PRIVATE_PACKED], submitted)
        self.assertEqual(1, len(initialized))
        self.assertEqual("pikachu", initialized[0][1][0]["species"])
        self.assertFalse(runtime.retains_selection)
        self.assertIsNone(self.state_document()["reservation"])
        self.assertEqual(1, self.state_document()["next_index"])

    async def test_loaded_registry_snapshot_survives_registry_file_mutation(self):
        submitted = []

        async def submit_team(packed):
            submitted.append(packed)

        runtime = self.runtime(
            FakeTransport((EXACT_CHALLENGE, exact_room_message())),
            submit_team,
            self.successful_initializer,
        )
        self.fixture.registry_path.write_text("{}", encoding="utf-8")
        await runtime.run_once()
        self.assertEqual([PRIVATE_PACKED], submitted)

    async def test_sequential_successes_use_fresh_attempt_state(self):
        submitted = []

        async def submit_team(packed):
            submitted.append(packed)

        runtime = self.runtime(
            FakeTransport(
                (
                    EXACT_CHALLENGE,
                    exact_room_message(),
                    SECOND_EXACT_CHALLENGE,
                    exact_room_message(
                        "battle-gen9tugs-402",
                        SECOND_EXACT_TOKEN,
                    ),
                )
            ),
            submit_team,
            self.successful_initializer,
        )
        await runtime.run_once()
        await runtime.run_once()
        self.assertEqual(
            [PRIVATE_PACKED, PRIVATE_PACKED + "-TWO"],
            submitted,
        )
        self.assertFalse(runtime.retains_selection)
        self.assertEqual(["enable", "enable"], runtime._coordinator._transport.controls)

    async def test_post_battle_enable_replays_offer_consumed_during_battle(self):
        transport = FakeTransport((EXACT_CHALLENGE, exact_room_message()))
        pending_replay = False
        original_enable = transport.enable_challenge_tokens

        async def enable_with_replay():
            nonlocal pending_replay
            await original_enable()
            if pending_replay:
                transport.messages.extend(
                    (
                        SECOND_EXACT_CHALLENGE,
                        exact_room_message(
                            "battle-gen9tugs-402",
                            SECOND_EXACT_TOKEN,
                        ),
                    )
                )
                pending_replay = False

        transport.enable_challenge_tokens = enable_with_replay

        async def initialize(room, _projection):
            nonlocal pending_replay
            if room.room_id == "battle-gen9tugs-401":
                pending_replay = True
            return room

        runtime = self.runtime(transport, mock.AsyncMock(), initialize)
        first = await runtime.run_once()
        second = await runtime.run_once()
        self.assertEqual("battle-gen9tugs-401", first.room_id)
        self.assertEqual("battle-gen9tugs-402", second.room_id)
        self.assertEqual(["enable", "enable"], transport.controls)
        self.assertIsNone(self.state_document()["reservation"])
        self.assertEqual(2, self.state_document()["next_index"])

    async def test_durable_order_uses_real_lifecycle_boundaries(self):
        events = []

        def observe_accept(_challenge):
            reservation = self.state_document()["reservation"]
            self.assertEqual("accept_sent", reservation["phase"])
            self.assertEqual(EXACT_TOKEN, reservation["challenge_token"])
            events.append("accept")

        transport = FakeTransport(
            (EXACT_CHALLENGE, exact_room_message()),
            send_observer=observe_accept,
        )

        async def submit_team(_packed):
            self.assertEqual("reserved", self.state_document()["reservation"]["phase"])
            self.assertEqual(
                EXACT_TOKEN,
                self.state_document()["reservation"]["challenge_token"],
            )
            events.append("submission")

        runtime = self.runtime(transport, submit_team, self.successful_initializer)
        original_mark = runtime._store.mark_accept_sent

        def mark_accept_sent(reservation_id, challenge_token=None):
            result = original_mark(reservation_id, challenge_token)
            events.append("accept_sent")
            return result

        def verification_boundary(name, _team_id):
            if name == "metadata_verified":
                self.assertEqual(
                    "reserved", self.state_document()["reservation"]["phase"]
                )
                events.append("reservation_durable")
            elif name == "sidecar_verified":
                events.append("artifact_verified")

        with (
            mock.patch.object(
                runtime._store,
                "mark_accept_sent",
                side_effect=mark_accept_sent,
            ),
            mock.patch(
                "fp.data.blind_pool.canonical_artifacts._verification_boundary",
                side_effect=verification_boundary,
            ),
        ):
            await runtime.run_once()
        self.assertEqual(
            [
                "reservation_durable",
                "artifact_verified",
                "submission",
                "accept_sent",
                "accept",
            ],
            events,
        )

    async def test_buffered_room_handoff_precedes_initializer_and_is_unchanged(self):
        request_line = '|request|{"rqid":1}'
        payload = exact_room_message() + "\n" + request_line
        transport = FakeTransport((EXACT_CHALLENGE, payload))
        observed = []

        async def submit_team(_packed):
            return None

        async def initialize(room, _projection):
            observed.extend(tuple(transport._blind_room_replay))
            return room

        runtime = self.runtime(transport, submit_team, initialize)
        await runtime.run_once()
        self.assertTrue(any(event.endswith(request_line) for event in observed))
        self.assertTrue(observed[-1].endswith(request_line))

    async def test_real_battler_initializer_keeps_request_data_authoritative(self):
        transport = FakeTransport((EXACT_CHALLENGE, exact_room_message()))

        async def submit_team(_packed):
            return None

        async def initialize(room, projection):
            battle = Battle(room.room_id)
            battle.user.team_dict = projection
            battle.user.initialize_first_turn_user_from_json(
                {
                    "active": [
                        {
                            "moves": [
                                {
                                    "move": "Tackle",
                                    "id": "tackle",
                                    "pp": 35,
                                    "maxpp": 35,
                                    "target": "normal",
                                    "disabled": False,
                                }
                            ]
                        }
                    ],
                    "side": {
                        "id": "p2",
                        "pokemon": [
                            {
                                "ident": "p2: Request Nickname",
                                "details": "Pikachu, L77, M",
                                "condition": "101/202",
                                "active": True,
                                "stats": {
                                    "atk": 111,
                                    "def": 112,
                                    "spa": 113,
                                    "spd": 114,
                                    "spe": 115,
                                },
                                "moves": ["tackle"],
                                "baseAbility": "static",
                                "ability": "static",
                                "item": "lightball",
                                "teraType": "Electric",
                            }
                        ],
                    },
                }
            )
            return battle

        runtime = self.runtime(transport, submit_team, initialize)
        battle = await runtime.run_once()
        active = battle.user.active
        self.assertEqual("Request Nickname", active.nickname)
        self.assertEqual(77, active.level)
        self.assertEqual("static", active.ability)
        self.assertEqual("lightball", active.item)
        self.assertEqual("electric", active.tera_type)
        self.assertEqual(101, active.hp)
        self.assertEqual(202, active.max_hp)
        self.assertEqual(
            {
                "attack": 111,
                "defense": 112,
                "special-attack": 113,
                "special-defense": 114,
                "speed": 115,
            },
            active.stats,
        )
        self.assertEqual(["tackle"], [move.name for move in active.moves])
        self.assertEqual(35, active.moves[0].current_pp)
        self.assertEqual("jolly", active.nature)
        self.assertEqual((4, 8, 12, 16, 20, 24), active.evs)
        self.assertEqual((31, 30, 29, 28, 27, 26), active.ivs)
        self.assertFalse(runtime.retains_selection)
        self.assertFalse(
            any(
                value.__class__.__name__ == "CanonicalTeamArtifact"
                for value in (*vars(battle).values(), *vars(battle.user).values())
            )
        )
        self.assertNotIn(PRIVATE_PACKED, repr(vars(battle)))
        intended_values = (*vars(battle).values(), *vars(battle.user).values())
        self.assertFalse(
            any(isinstance(value, BlindChallengeToken) for value in intended_values)
        )
        self.assertNotIn(EXACT_TOKEN, repr(vars(battle)) + repr(vars(battle.user)))


class CanonicalRuntimeFailureTests(RuntimeFixture):
    async def test_runtime_construction_requires_explicit_exact_protocol(self):
        transport = FakeTransport(())

        async def submit_team(_packed):
            return None

        with self.assertRaises(TypeError):
            CanonicalBlindRuntime(
                self.state_config,
                self.registry,
                transport,
                submit_team,
                self.successful_initializer,
            )

    async def assert_corruption_fails_preaccept(self, mutation):
        mutation()
        transport = FakeTransport((EXACT_CHALLENGE,))

        async def submit_team(_packed):
            self.fail("corrupt artifact reached submission")

        runtime = self.runtime(transport, submit_team, self.successful_initializer)
        with self.assertRaises(BlindPoolLifecycleError) as caught:
            await runtime.run_once()
        self.assert_safe_error(
            caught.exception,
            "team_artifact_verification_failed",
        )
        self.assertEqual([], transport.sent)
        self.assertFalse(runtime.retains_selection)
        state = self.state_document()
        self.assertIsNone(state["reservation"])
        self.assertEqual(0, state["next_index"])
        store = BlindPoolBagStore.from_canonical_registry(
            self.state_config,
            self.registry,
            random_source=IdentityRandom(),
            reservation_id_factory=lambda: next(self.identifiers),
            lock_timeout_seconds=1,
        )
        reservation = store.reserve_next()
        self.assertEqual("BL-001-v1", reservation.team_id)
        store.release_reservation(reservation.reservation_id)

    async def test_metadata_corruption_fails_before_accept(self):
        await self.assert_corruption_fails_preaccept(
            lambda: (self.first_directory / "metadata.json").write_bytes(b"invalid")
        )

    async def test_packed_corruption_fails_before_accept(self):
        await self.assert_corruption_fails_preaccept(
            lambda: (self.first_directory / "packed.txt").write_bytes(b"changed")
        )

    async def test_sidecar_corruption_fails_before_accept(self):
        await self.assert_corruption_fails_preaccept(
            lambda: (self.first_directory / "team.json").write_bytes(b"changed")
        )

    async def test_artifact_directory_corruption_fails_before_accept(self):
        await self.assert_corruption_fails_preaccept(
            lambda: (self.first_directory / "unexpected").write_text(
                "synthetic", encoding="utf-8"
            )
        )

    async def test_submitter_failure_releases_before_accept_and_is_sanitized(self):
        transport = FakeTransport((EXACT_CHALLENGE,))

        async def submit_team(_packed):
            raise RuntimeError(PRIVATE_ERROR)

        runtime = self.runtime(transport, submit_team, self.successful_initializer)
        with self.assertRaises(BlindPoolLifecycleError) as caught:
            await runtime.run_once()
        self.assert_safe_error(caught.exception, "team_submission_failed")
        self.assertEqual([], transport.sent)
        self.assertIsNone(self.state_document()["reservation"])
        self.assertFalse(runtime.retains_selection)

    async def test_submitter_cancellation_releases_and_clears_selection(self):
        transport = FakeTransport((EXACT_CHALLENGE,))

        async def submit_team(_packed):
            raise asyncio.CancelledError(PRIVATE_ERROR)

        runtime = self.runtime(transport, submit_team, self.successful_initializer)
        with self.assertRaises(asyncio.CancelledError) as caught:
            await runtime.run_once()
        self.assertNotIn(PRIVATE_ERROR, str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        self.assertEqual([], transport.sent)
        self.assertIsNone(self.state_document()["reservation"])
        self.assertFalse(runtime.retains_selection)

    async def test_transport_cancellation_is_sanitized_before_reservation(self):
        transport = FakeTransport(())

        async def cancelled_receive():
            raise asyncio.CancelledError(PRIVATE_ERROR)

        transport.receive_message = cancelled_receive

        async def submit_team(_packed):
            self.fail("team submission reached after transport cancellation")

        runtime = self.runtime(transport, submit_team, self.successful_initializer)
        with self.assertRaises(asyncio.CancelledError) as caught:
            await runtime.run_once()
        self.assertNotIn(PRIVATE_ERROR, str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        self.assertIsNone(self.state_document()["reservation"])
        self.assertFalse(runtime.retains_selection)

    async def test_submit_failure_restarts_with_the_same_team_id(self):
        attempted = []

        async def failed_submit(packed):
            attempted.append(packed)
            raise RuntimeError(PRIVATE_ERROR)

        first_runtime = self.runtime(
            FakeTransport((EXACT_CHALLENGE,)),
            failed_submit,
            self.successful_initializer,
        )
        with self.assertRaises(BlindPoolLifecycleError):
            await first_runtime.run_once()

        async def successful_submit(packed):
            attempted.append(packed)

        restarted_runtime = self.runtime(
            FakeTransport((EXACT_CHALLENGE, exact_room_message())),
            successful_submit,
            self.successful_initializer,
        )
        await restarted_runtime.run_once()
        self.assertEqual([PRIVATE_PACKED, PRIVATE_PACKED], attempted)
        self.assertEqual(1, self.state_document()["next_index"])

    async def test_accept_sent_persistence_failure_releases_and_clears(self):
        transport = FakeTransport((EXACT_CHALLENGE,))

        async def submit_team(_packed):
            return None

        runtime = self.runtime(transport, submit_team, self.successful_initializer)
        failure = BlindPoolValidationError(
            "synthetic_state_failure",
            "Synthetic state transition failed",
        )
        with (
            mock.patch.object(
                runtime._store,
                "mark_accept_sent",
                side_effect=failure,
            ),
            self.assertRaises(BlindPoolLifecycleError) as caught,
        ):
            await runtime.run_once()
        self.assert_safe_error(caught.exception, "acceptance_transition_failed")
        self.assertEqual([], transport.sent)
        self.assertIsNone(self.state_document()["reservation"])
        self.assertFalse(runtime.retains_selection)

    async def test_accept_send_ambiguity_quarantines_bag_but_clears_session(self):
        transport = FakeTransport(
            (EXACT_CHALLENGE,),
            send_error=RuntimeError(PRIVATE_ERROR),
        )

        async def submit_team(_packed):
            return None

        runtime = self.runtime(transport, submit_team, self.successful_initializer)
        with self.assertRaises(BlindPoolReconciliationRequired) as caught:
            await runtime.run_once()
        self.assert_safe_error(
            caught.exception,
            "acceptance_transmission_ambiguous",
        )
        state = self.state_document()
        self.assertEqual("accept_sent", state["reservation"]["phase"])
        self.assertFalse(runtime.retains_selection)
        runtime.reconcile_as_no_room(state["reservation"]["reservation_id"])
        self.assertIsNone(self.state_document()["reservation"])

    async def test_initializer_failure_occurs_after_commit_and_clears_selection(self):
        transport = FakeTransport((EXACT_CHALLENGE, exact_room_message()))

        async def submit_team(_packed):
            return None

        async def initialize(_room, _projection):
            raise RuntimeError(PRIVATE_ERROR)

        runtime = self.runtime(transport, submit_team, initialize)
        with self.assertRaises(BlindPoolLifecycleError) as caught:
            await runtime.run_once()
        self.assert_safe_error(caught.exception, "battle_initialization_failed")
        self.assertEqual(1, self.state_document()["next_index"])
        self.assertIsNone(self.state_document()["reservation"])
        self.assertFalse(runtime.retains_selection)

    async def test_initializer_cancellation_is_sanitized_after_commit(self):
        transport = FakeTransport((EXACT_CHALLENGE, exact_room_message()))

        async def submit_team(_packed):
            return None

        async def initialize(_room, _projection):
            raise asyncio.CancelledError(PRIVATE_ERROR)

        runtime = self.runtime(transport, submit_team, initialize)
        with self.assertRaises(asyncio.CancelledError) as caught:
            await runtime.run_once()
        self.assertNotIn(PRIVATE_ERROR, str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        self.assertEqual(1, self.state_document()["next_index"])
        self.assertIsNone(self.state_document()["reservation"])
        self.assertFalse(runtime.retains_selection)


class CanonicalRuntimeSessionTests(RuntimeFixture):
    def room(self):
        return BlindPoolBattleRoom(
            room_id="battle-gen9tugs-999",
            format_id="gen9tugs",
            opponent_id="syntheticopponent",
            opponent_name="Synthetic Opponent",
            bot_slot="p2",
            opponent_slot="p1",
        )

    async def session(self):
        async def submit_team(_packed):
            return None

        return CanonicalBlindRuntimeSession(
            self.registry,
            submit_team,
            self.successful_initializer,
        )

    async def test_initializer_before_prepare_fails_closed(self):
        session = await self.session()
        session.begin_attempt()
        with self.assertRaises(BlindPoolLifecycleError) as caught:
            await session.initialize_battle(self.room())
        self.assert_safe_error(
            caught.exception,
            "canonical_runtime_initializer_invalid",
        )
        session.end_attempt()

    async def test_double_prepare_same_or_different_id_fails_closed(self):
        for second_id in ("BL-001-v1", "BL-002-v1"):
            with self.subTest(second_id=second_id):
                session = await self.session()
                session.begin_attempt()
                await session.prepare_team("BL-001-v1")
                with self.assertRaises(BlindPoolLifecycleError) as caught:
                    await session.prepare_team(second_id)
                self.assert_safe_error(
                    caught.exception,
                    "canonical_runtime_prepare_invalid",
                )
                session.end_attempt()

    async def test_initializer_cannot_run_twice(self):
        session = await self.session()
        session.begin_attempt()
        await session.prepare_team("BL-001-v1")
        await session.initialize_battle(self.room())
        with self.assertRaises(BlindPoolLifecycleError) as caught:
            await session.initialize_battle(self.room())
        self.assert_safe_error(
            caught.exception,
            "canonical_runtime_initializer_invalid",
        )
        session.end_attempt()

    async def test_direct_artifact_failure_is_sanitized_and_cleared(self):
        (self.first_directory / "packed.txt").write_bytes(b"changed")
        session = await self.session()
        session.begin_attempt()
        with self.assertRaises(BlindPoolLifecycleError) as caught:
            await session.prepare_team("BL-001-v1")
        self.assert_safe_error(
            caught.exception,
            "canonical_runtime_artifact_rejected",
        )
        self.assertFalse(session.prepared)
        session.end_attempt()

    async def test_inactive_registered_team_cannot_be_prepared(self):
        submitted = []

        async def submit_team(packed):
            submitted.append(packed)

        session = CanonicalBlindRuntimeSession(
            self.registry,
            submit_team,
            self.successful_initializer,
        )
        session.begin_attempt()
        with self.assertRaises(BlindPoolLifecycleError) as caught:
            await session.prepare_team("BL-003-v1")
        self.assert_safe_error(
            caught.exception,
            "canonical_runtime_selection_invalid",
        )
        self.assertEqual([], submitted)
        self.assertFalse(session.prepared)
        session.end_attempt()

    async def test_unexpected_verifier_failure_is_distinct_and_sanitized(self):
        session = await self.session()
        session.begin_attempt()
        with (
            mock.patch(
                "fp.data.blind_pool.canonical_runtime.load_canonical_team_artifact",
                side_effect=RuntimeError(PRIVATE_ERROR),
            ),
            self.assertRaises(BlindPoolLifecycleError) as caught,
        ):
            await session.prepare_team("BL-001-v1")
        self.assert_safe_error(
            caught.exception,
            "canonical_runtime_internal_failure",
        )
        self.assertFalse(session.prepared)
        session.end_attempt()

    async def test_direct_initializer_failure_is_sanitized_and_cleared(self):
        async def submit_team(_packed):
            return None

        async def initialize(_room, _projection):
            raise RuntimeError(PRIVATE_ERROR)

        session = CanonicalBlindRuntimeSession(
            self.registry,
            submit_team,
            initialize,
        )
        session.begin_attempt()
        await session.prepare_team("BL-001-v1")
        with self.assertRaises(BlindPoolLifecycleError) as caught:
            await session.initialize_battle(self.room())
        self.assert_safe_error(
            caught.exception,
            "canonical_runtime_initialization_failed",
        )
        self.assertFalse(session.prepared)
        session.end_attempt()

    async def test_ended_attempt_cannot_reuse_stale_preparation(self):
        session = await self.session()
        session.begin_attempt()
        await session.prepare_team("BL-001-v1")
        self.assertTrue(session.prepared)
        session.end_attempt()
        self.assertFalse(session.prepared)
        with self.assertRaises(BlindPoolLifecycleError) as caught:
            await session.initialize_battle(self.room())
        self.assert_safe_error(
            caught.exception,
            "canonical_runtime_initializer_invalid",
        )

    async def test_inflight_old_prepare_cannot_cross_attempt_boundary(self):
        submission_started = asyncio.Event()
        submission_release = asyncio.Event()

        async def submit_team(_packed):
            submission_started.set()
            await submission_release.wait()

        session = CanonicalBlindRuntimeSession(
            self.registry,
            submit_team,
            self.successful_initializer,
        )
        session.begin_attempt()
        old_prepare = asyncio.create_task(session.prepare_team("BL-001-v1"))
        await submission_started.wait()
        session.end_attempt()
        session.begin_attempt()
        submission_release.set()
        with self.assertRaises(BlindPoolLifecycleError) as caught:
            await old_prepare
        self.assert_safe_error(
            caught.exception,
            "canonical_runtime_attempt_stale",
        )
        self.assertFalse(session.prepared)
        self.assertEqual("ready", session.phase)
        await session.prepare_team("BL-002-v1")
        self.assertTrue(session.prepared)
        session.end_attempt()

    async def test_repr_str_and_callback_failures_hide_private_values(self):
        async def submit_team(_packed):
            raise RuntimeError(PRIVATE_ERROR)

        session = CanonicalBlindRuntimeSession(
            self.registry,
            submit_team,
            self.successful_initializer,
        )
        output = io.StringIO()
        logs = io.StringIO()
        handler = logging.StreamHandler(logs)
        logging.getLogger().addHandler(handler)
        self.addCleanup(logging.getLogger().removeHandler, handler)
        session.begin_attempt()
        with redirect_stdout(output), redirect_stderr(output):
            with self.assertRaises(BlindPoolLifecycleError) as caught:
                await session.prepare_team("BL-001-v1")
        self.assert_safe_error(
            caught.exception,
            "canonical_runtime_submission_failed",
        )
        rendered = repr(session) + str(session) + output.getvalue() + logs.getvalue()
        for sentinel in (
            PRIVATE_PACKED,
            PRIVATE_ERROR,
            PRIVATE_SIDECAR,
            str(self.fixture.private_root),
        ):
            self.assertNotIn(sentinel, rendered)
        session.end_attempt()

    async def test_runtime_objects_disable_serialization(self):
        session = await self.session()
        runtime = self.runtime(
            FakeTransport(()),
            session._submit_team,
            self.successful_initializer,
        )
        for value in (session, runtime):
            with self.subTest(value=value.__class__.__name__):
                with self.assertRaises(TypeError):
                    pickle.dumps(value)

    async def test_runtime_callback_requires_current_durable_reservation(self):
        async def submit_team(_packed):
            return None

        runtime = self.runtime(
            FakeTransport(()),
            submit_team,
            self.successful_initializer,
        )
        await runtime.startup()
        runtime._session.begin_attempt()
        with self.assertRaises(BlindPoolLifecycleError) as missing:
            await runtime._prepare_reserved_team("BL-001-v1")
        self.assert_safe_error(
            missing.exception,
            "canonical_runtime_reservation_invalid",
        )

        reservation = runtime._store.reserve_next(BlindChallengeToken("a" * 32))
        with self.assertRaises(BlindPoolLifecycleError) as mismatched:
            await runtime._prepare_reserved_team("BL-002-v1")
        self.assert_safe_error(
            mismatched.exception,
            "canonical_runtime_reservation_invalid",
        )
        await runtime._prepare_reserved_team(reservation.team_id)
        self.assertTrue(runtime.retains_selection)
        runtime.close_attempt()
        runtime._store.release_reservation(reservation.reservation_id)

    async def test_outer_close_clears_prepared_selection(self):
        async def submit_team(_packed):
            return None

        runtime = self.runtime(
            FakeTransport(()),
            submit_team,
            self.successful_initializer,
        )
        await runtime.startup()
        runtime._session.begin_attempt()
        reservation = runtime._store.reserve_next(BlindChallengeToken("a" * 32))
        await runtime._prepare_reserved_team(reservation.team_id)
        self.assertTrue(runtime.retains_selection)
        runtime.close_attempt()
        self.assertFalse(runtime.retains_selection)
        self.assertEqual("idle", runtime._session.phase)
        runtime._store.release_reservation(reservation.reservation_id)
