from __future__ import annotations

import json
import logging
from unittest import mock

from fp.data.blind_pool.canonical_runtime import CanonicalBlindRuntime
from fp.data.blind_pool.errors import (
    BlindPoolLifecycleError,
    BlindPoolValidationError,
)
from fp.data.blind_pool.lifecycle import BlindExactChallengeProtocol
from fp.data.blind_pool.rating_state import (
    BlindRatingStateStore,
    validate_rating_state_config,
)
from fp.data.blind_pool.result_ledger import (
    OUTCOME_PLAYER_LOSS,
    OUTCOME_PLAYER_WIN,
    OUTCOME_TIE,
    PENDING_AWAITING_TERMINAL,
    PENDING_ROOM_VALIDATED,
    BlindResultLedgerStore,
    validate_result_ledger_config,
)
from tests.test_blind_canonical_runtime import RuntimeFixture
from tests.test_blind_pool_lifecycle import (
    EXACT_CHALLENGE,
    FakeTransport,
    IdentityRandom,
    exact_room_message,
)


PRIVATE_SENTINEL = "PHASE6A-RUNTIME-PRIVATE-SENTINEL"


class ResultRuntimeFixture(RuntimeFixture):
    def setUp(self):
        super().setUp()
        result_directory = self.fixture.base / "ladder"
        result_directory.mkdir()
        self.result_path = result_directory / "results.json"
        result_config = validate_result_ledger_config(
            self.result_path,
            private_root=self.fixture.private_root,
            registry_path=self.fixture.registry_path,
            selection_state_path=self.state_path,
        )
        self.result_store = BlindResultLedgerStore(
            result_config,
            lock_timeout_seconds=1,
        )
        self.result_store.initialize_empty()
        self.rating_path = result_directory / "ratings.json"
        rating_config = validate_rating_state_config(
            self.rating_path,
            private_root=self.fixture.private_root,
            registry_path=self.fixture.registry_path,
            selection_state_path=self.state_path,
            result_ledger_path=self.result_path,
        )
        self.rating_store = BlindRatingStateStore(
            rating_config,
            lock_timeout_seconds=1,
        )
        self.rating_store.initialize(self.result_store.require_ready())

    def result_runtime(self, transport, initialize_battle):
        return CanonicalBlindRuntime(
            self.state_config,
            self.registry,
            transport,
            mock.AsyncMock(),
            initialize_battle,
            exact_protocol=BlindExactChallengeProtocol.from_transport(transport),
            random_source=IdentityRandom(),
            reservation_id_factory=lambda: next(self.identifiers),
            lock_timeout_seconds=1,
            room_timeout_seconds=0.1,
            result_store=self.result_store,
            rating_store=self.rating_store,
        )

    def selection_state(self):
        return json.loads(self.state_path.read_text(encoding="utf-8"))


class CanonicalResultRuntimeTests(ResultRuntimeFixture):
    async def test_write_ahead_and_selection_commit_order_precedes_battle(self):
        events = []
        transport = FakeTransport((EXACT_CHALLENGE, exact_room_message()))
        holder = {}

        async def initialize(_room, _projection):
            events.append("battle")
            ledger = self.result_store.load()
            self.assertEqual(PENDING_AWAITING_TERMINAL, ledger.pending_result.phase)
            self.assertIsNone(self.selection_state()["reservation"])
            holder["runtime"].record_terminal_result("Blind Bot", tied=False)
            return "Blind Bot"

        runtime = self.result_runtime(transport, initialize)
        holder["runtime"] = runtime
        original_intent = self.result_store.create_pending_intent
        original_mark = self.result_store.mark_selection_committed
        original_commit = runtime._store.commit_room_created

        def intent(**kwargs):
            events.append("intent")
            return original_intent(**kwargs)

        def commit(reservation_id):
            events.append("selection_commit")
            return original_commit(reservation_id)

        def mark(battle_id):
            events.append("result_commit")
            return original_mark(battle_id)

        with (
            mock.patch.object(
                self.result_store,
                "create_pending_intent",
                side_effect=intent,
            ),
            mock.patch.object(
                runtime._store,
                "commit_room_created",
                side_effect=commit,
            ),
            mock.patch.object(
                self.result_store,
                "mark_selection_committed",
                side_effect=mark,
            ),
        ):
            winner = await runtime.run_once()
        self.assertEqual("Blind Bot", winner)
        self.assertEqual(
            ["intent", "selection_commit", "result_commit", "battle"],
            events,
        )
        ledger = self.result_store.load()
        self.assertEqual(1, ledger.completed_count)
        self.assertEqual(0, ledger.pending_count)
        rating = self.rating_store.verify(self.result_store.require_ready())
        self.assertEqual((1, 1), (rating.processed_sequence, rating.rated_results))
        self.assertEqual(-16, runtime.last_rating_update.rating_delta)

    async def test_player_bot_tie_and_forfeit_winner_events_normalize(self):
        cases = (
            ("Synthetic Opponent", False, OUTCOME_PLAYER_WIN),
            ("Blind Bot", False, OUTCOME_PLAYER_LOSS),
            (None, True, OUTCOME_TIE),
            ("Synthetic Opponent", False, OUTCOME_PLAYER_WIN),
        )
        for index, (winner, tied, outcome) in enumerate(cases):
            with self.subTest(index=index, outcome=outcome):
                if index:
                    self.tearDown()
                    self.setUp()
                transport = FakeTransport((EXACT_CHALLENGE, exact_room_message()))
                holder = {}

                async def initialize(_room, _projection):
                    holder["runtime"].record_terminal_result(winner, tied=tied)
                    return winner

                runtime = self.result_runtime(transport, initialize)
                holder["runtime"] = runtime
                await runtime.run_once()
                ledger = self.result_store.load()
                self.assertEqual(outcome, ledger.completed_results[0].outcome)

    async def test_duplicate_terminal_event_is_idempotent(self):
        transport = FakeTransport((EXACT_CHALLENGE, exact_room_message()))
        holder = {}

        async def initialize(_room, _projection):
            holder["runtime"].record_terminal_result("Blind Bot", tied=False)
            holder["runtime"].record_terminal_result("Blind Bot", tied=False)
            return "Blind Bot"

        runtime = self.result_runtime(transport, initialize)
        holder["runtime"] = runtime
        await runtime.run_once()
        self.assertEqual(1, self.result_store.load().completed_count)
        self.assertEqual(1, self.rating_store.load().rated_results)

    async def test_result_finalization_failure_never_invokes_rating_sync(self):
        transport = FakeTransport((EXACT_CHALLENGE, exact_room_message()))
        holder = {}

        async def initialize(_room, _projection):
            holder["runtime"].record_terminal_result("Blind Bot", tied=False)

        runtime = self.result_runtime(transport, initialize)
        holder["runtime"] = runtime
        with mock.patch.object(
            self.result_store,
            "finalize_terminal",
            side_effect=BlindPoolValidationError("synthetic", "synthetic failure"),
        ), mock.patch.object(self.rating_store, "sync") as rating_sync:
            with self.assertRaises(BlindPoolLifecycleError):
                await runtime.run_once()
        rating_sync.assert_not_called()
        self.assertEqual(0, self.rating_store.load().processed_sequence)
        self.assertEqual(1, self.result_store.load().pending_count)

    async def test_rating_failure_after_result_finalization_stops_runtime_behind(self):
        transport = FakeTransport((EXACT_CHALLENGE, exact_room_message()))
        holder = {}

        async def initialize(_room, _projection):
            holder["runtime"].record_terminal_result("Synthetic Opponent", tied=False)

        runtime = self.result_runtime(transport, initialize)
        holder["runtime"] = runtime
        with mock.patch.object(
            self.rating_store,
            "sync",
            side_effect=BlindPoolValidationError("synthetic", "synthetic failure"),
        ):
            with self.assertRaises(BlindPoolLifecycleError) as caught:
                await runtime.run_once()
        self.assertEqual("canonical_rating_sync_failed", caught.exception.code)
        ledger = self.result_store.require_ready()
        self.assertEqual(1, ledger.completed_count)
        self.assertEqual(0, self.rating_store.load().processed_sequence)
        recovered = self.rating_store.sync(ledger)
        self.assertEqual(1, recovered.state.processed_sequence)
        self.assertEqual(1, recovered.state.rated_results)

    async def test_conflicting_duplicate_terminal_event_fails_closed(self):
        transport = FakeTransport((EXACT_CHALLENGE, exact_room_message()))
        holder = {}

        async def initialize(_room, _projection):
            holder["runtime"].record_terminal_result(
                "Synthetic Opponent",
                tied=False,
            )
            holder["runtime"].record_terminal_result("Blind Bot", tied=False)

        runtime = self.result_runtime(transport, initialize)
        holder["runtime"] = runtime
        with self.assertRaises(BlindPoolLifecycleError):
            await runtime.run_once()
        ledger = self.result_store.load()
        self.assertEqual(1, ledger.completed_count)
        self.assertEqual(OUTCOME_PLAYER_WIN, ledger.completed_results[0].outcome)

    async def test_unexpected_winner_fails_closed_with_pending_result(self):
        transport = FakeTransport((EXACT_CHALLENGE, exact_room_message()))
        holder = {}

        async def initialize(_room, _projection):
            holder["runtime"].record_terminal_result("Unexpected User", tied=False)

        runtime = self.result_runtime(transport, initialize)
        holder["runtime"] = runtime
        with self.assertRaises(BlindPoolLifecycleError):
            await runtime.run_once()
        ledger = self.result_store.load()
        self.assertEqual(0, ledger.completed_count)
        self.assertEqual(PENDING_AWAITING_TERMINAL, ledger.pending_result.phase)

    async def test_disconnect_deinit_exception_and_missing_terminal_record_nothing(
        self,
    ):
        failures = (
            ConnectionError(PRIVATE_SENTINEL),
            RuntimeError(PRIVATE_SENTINEL),
            ValueError(PRIVATE_SENTINEL),
            None,
        )
        for index, failure in enumerate(failures):
            with self.subTest(index=index):
                if index:
                    self.tearDown()
                    self.setUp()
                transport = FakeTransport((EXACT_CHALLENGE, exact_room_message()))

                async def initialize(_room, _projection):
                    if failure is not None:
                        raise failure
                    return None

                runtime = self.result_runtime(transport, initialize)
                with self.assertRaises(BlindPoolLifecycleError):
                    await runtime.run_once()
                ledger = self.result_store.load()
                self.assertEqual(0, ledger.completed_count)
                self.assertEqual(PENDING_AWAITING_TERMINAL, ledger.pending_result.phase)

    async def test_intent_failure_prevents_selection_commit_and_battle(self):
        transport = FakeTransport((EXACT_CHALLENGE, exact_room_message()))
        initialize = mock.AsyncMock()
        runtime = self.result_runtime(transport, initialize)
        with mock.patch.object(
            self.result_store,
            "create_pending_intent",
            side_effect=BlindPoolValidationError("synthetic", "safe"),
        ), self.assertRaises(BlindPoolLifecycleError) as caught:
            await runtime.run_once()
        self.assertEqual("result_intent_failed", caught.exception.code)
        self.assertEqual("accept_sent", self.selection_state()["reservation"]["phase"])
        self.assertEqual(0, self.result_store.load().pending_count)
        initialize.assert_not_awaited()

    async def test_selection_commit_failure_leaves_room_validated_intent(self):
        transport = FakeTransport((EXACT_CHALLENGE, exact_room_message()))
        initialize = mock.AsyncMock()
        runtime = self.result_runtime(transport, initialize)
        with mock.patch.object(
            runtime._store,
            "commit_room_created",
            side_effect=BlindPoolValidationError("synthetic", "safe"),
        ), self.assertRaises(BlindPoolLifecycleError):
            await runtime.run_once()
        self.assertEqual("accept_sent", self.selection_state()["reservation"]["phase"])
        self.assertEqual(
            PENDING_ROOM_VALIDATED,
            self.result_store.load().pending_result.phase,
        )
        initialize.assert_not_awaited()

    async def test_result_commit_phase_failure_after_selection_commit_stops_battle(
        self,
    ):
        transport = FakeTransport((EXACT_CHALLENGE, exact_room_message()))
        initialize = mock.AsyncMock()
        runtime = self.result_runtime(transport, initialize)
        with mock.patch.object(
            self.result_store,
            "mark_selection_committed",
            side_effect=BlindPoolValidationError("synthetic", "safe"),
        ), self.assertRaises(BlindPoolLifecycleError) as caught:
            await runtime.run_once()
        self.assertEqual("result_selection_commit_failed", caught.exception.code)
        self.assertIsNone(self.selection_state()["reservation"])
        self.assertEqual(
            PENDING_ROOM_VALIDATED,
            self.result_store.load().pending_result.phase,
        )
        initialize.assert_not_awaited()

    async def test_result_logs_and_representations_hide_private_associations(self):
        transport = FakeTransport((EXACT_CHALLENGE, exact_room_message()))
        holder = {}

        async def initialize(_room, _projection):
            holder["runtime"].record_terminal_result("Blind Bot", tied=False)
            return "Blind Bot"

        runtime = self.result_runtime(transport, initialize)
        holder["runtime"] = runtime
        with self.assertLogs(level=logging.INFO) as captured:
            await runtime.run_once()
        rendered = "\n".join(captured.output)
        state = self.result_store.load()
        private_values = (
            state.completed_results[0].battle_id,
            state.completed_results[0].team_id,
            state.completed_results[0].reservation_id,
        )
        for value in private_values:
            self.assertNotIn(value, rendered)
            self.assertNotIn(value, repr(state))


if __name__ == "__main__":
    import unittest

    unittest.main()
