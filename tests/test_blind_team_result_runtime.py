from __future__ import annotations

import json
from unittest import mock

from fp.data.blind_pool import (
    BlindChallengeToken,
    BlindPoolBattleRoom,
    BlindPoolChallenge,
    BlindPoolLifecycleError,
    BlindPoolValidationError,
    BlindTeamPublicIdentity,
    BlindTeamPublicRegistryStore,
    BlindTeamRatingStateStore,
    BlindTeamResultLedgerStore,
    PUBLIC_TEAM_KIND_PLAYER,
    TEAM_OUTCOME_A_LOSS,
    TEAM_OUTCOME_A_WIN,
    TEAM_OUTCOME_TIE,
    validate_team_public_registry_config,
    validate_team_rating_state_config,
    validate_team_result_ledger_config,
)
from fp.data.blind_pool.canonical_runtime import CanonicalBlindRuntime
from fp.data.blind_pool.lifecycle import BlindExactChallengeProtocol
from tests.test_blind_canonical_runtime import RuntimeFixture
from tests.test_blind_challenge_tokens import PLAYER_TEAM_ID, team_identity_event
from tests.test_blind_pool_lifecycle import (
    EXACT_CHALLENGE,
    FakeTransport,
    IdentityRandom,
    exact_room_message,
)


class TeamResultRuntimeFixture(RuntimeFixture):
    def setUp(self):
        super().setUp()
        ladder = self.fixture.base / "team-ladder"
        ladder.mkdir()
        self.public_path = ladder / "public.json"
        self.result_path = ladder / "results.json"
        self.rating_path = ladder / "ratings.json"
        self.public_store = BlindTeamPublicRegistryStore(
            validate_team_public_registry_config(
                self.public_path,
                private_root=self.fixture.private_root,
                canonical_registry_path=self.fixture.registry_path,
                selection_state_path=self.state_path,
                result_ledger_path=self.result_path,
                rating_state_path=self.rating_path,
            )
        )
        self.public_store.initialize(
            self.registry.active_ids, random_source=IdentityRandom()
        )
        self.result_store = BlindTeamResultLedgerStore(
            validate_team_result_ledger_config(
                self.result_path,
                private_root=self.fixture.private_root,
                registry_path=self.fixture.registry_path,
                selection_state_path=self.state_path,
                rating_state_path=self.rating_path,
                public_registry_path=self.public_path,
            )
        )
        ledger = self.result_store.initialize_empty()
        self.rating_store = BlindTeamRatingStateStore(
            validate_team_rating_state_config(
                self.rating_path,
                private_root=self.fixture.private_root,
                canonical_registry_path=self.fixture.registry_path,
                selection_state_path=self.state_path,
                result_ledger_path=self.result_path,
                public_registry_path=self.public_path,
            )
        )
        self.rating_store.initialize(ledger)

    def transport(self):
        return FakeTransport(
            (
                "\n".join((EXACT_CHALLENGE, team_identity_event())),
                exact_room_message(),
            )
        )

    def runtime(self, transport, initialize_battle):
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
            team_public_store=self.public_store,
            team_mode=True,
        )


class TeamResultRuntimeTests(TeamResultRuntimeFixture):
    async def test_team_mode_persists_identity_before_result_and_rates_both_teams(self):
        transport = self.transport()
        holder = {}

        async def initialize(_room, _projection):
            holder["runtime"].record_terminal_result("Synthetic Opponent", tied=False)
            return "done"

        runtime = self.runtime(transport, initialize)
        holder["runtime"] = runtime
        original_intent = self.result_store.create_pending_intent

        def observe(**kwargs):
            selection = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.assertEqual(3, selection["schema_version"])
            self.assertEqual(PLAYER_TEAM_ID, selection["reservation"]["player_team_id"])
            self.assertEqual(
                "Synthetic Offense",
                selection["reservation"]["player_team_display_name"],
            )
            return original_intent(**kwargs)

        with mock.patch.object(
            self.result_store, "create_pending_intent", side_effect=observe
        ):
            self.assertEqual("done", await runtime.run_once())
        record = self.result_store.require_ready().completed_results[0]
        self.assertEqual(PLAYER_TEAM_ID, record.player_team_id)
        self.assertEqual("BL-001-v1", record.bot_team_id)
        self.assertEqual(TEAM_OUTCOME_A_WIN, record.outcome)
        rating = self.rating_store.verify(self.result_store.require_ready())
        self.assertEqual(1516, rating.team(PLAYER_TEAM_ID).rating)
        self.assertEqual(1484, rating.team("BL-001-v1").rating)

    async def test_player_identity_is_registered_before_reservation(self):
        transport = self.transport()
        runtime = self.runtime(transport, mock.AsyncMock())
        original_reserve = runtime._store.reserve_next

        def observe(*args):
            identity = self.public_store.load().identity(PLAYER_TEAM_ID)
            self.assertEqual("Synthetic Offense", identity.display_name)
            return original_reserve(*args)

        with mock.patch.object(runtime._store, "reserve_next", side_effect=observe):
            with self.assertRaises(BlindPoolLifecycleError):
                await runtime.run_once()

    async def test_bot_alias_is_not_exposed_until_terminal_persistence(self):
        transport = self.transport()
        holder = {}

        async def initialize(_room, _projection):
            self.assertIsNone(holder["runtime"].last_team_rating_update)
            holder["runtime"].record_terminal_result("Blind Bot", tied=False)

        runtime = self.runtime(transport, initialize)
        holder["runtime"] = runtime
        await runtime.run_once()
        update = runtime.last_team_rating_update
        self.assertEqual("Synthetic Offense", update.player_name)
        self.assertEqual("Bot team 01", update.bot_name)
        self.assertFalse(hasattr(update, "player_team_id"))
        self.assertFalse(hasattr(update, "bot_team_id"))

    async def test_terminal_mapping_covers_bot_win_and_tie(self):
        cases = (
            ("Blind Bot", False, TEAM_OUTCOME_A_LOSS),
            (None, True, TEAM_OUTCOME_TIE),
        )
        for index, (winner, tied, expected) in enumerate(cases):
            with self.subTest(expected=expected):
                if index:
                    self.tearDown()
                    self.setUp()
                transport = self.transport()
                holder = {}

                async def initialize(_room, _projection):
                    holder["runtime"].record_terminal_result(winner, tied=tied)

                runtime = self.runtime(transport, initialize)
                holder["runtime"] = runtime
                await runtime.run_once()
                self.assertEqual(
                    expected,
                    self.result_store.require_ready().completed_results[0].outcome,
                )

    async def test_duplicate_terminal_event_is_idempotent(self):
        transport = self.transport()
        holder = {}

        async def initialize(_room, _projection):
            holder["runtime"].record_terminal_result("Blind Bot", tied=False)
            first = holder["runtime"].last_team_rating_update
            holder["runtime"].record_terminal_result("Blind Bot", tied=False)
            self.assertEqual(first, holder["runtime"].last_team_rating_update)

        runtime = self.runtime(transport, initialize)
        holder["runtime"] = runtime
        await runtime.run_once()
        self.assertEqual(1, self.result_store.load().completed_count)
        self.assertEqual(1, self.rating_store.load().rated_results)

    async def test_unexpected_terminal_winner_fails_with_pending_intent(self):
        transport = self.transport()
        holder = {}

        async def initialize(_room, _projection):
            holder["runtime"].record_terminal_result("Unexpected User", tied=False)

        runtime = self.runtime(transport, initialize)
        holder["runtime"] = runtime
        with self.assertRaises(BlindPoolLifecycleError):
            await runtime.run_once()
        self.assertEqual(1, self.result_store.load().pending_count)

    async def test_rating_failure_keeps_authoritative_result_for_restart_catchup(self):
        transport = self.transport()
        holder = {}

        async def initialize(_room, _projection):
            holder["runtime"].record_terminal_result("Synthetic Opponent", tied=False)

        runtime = self.runtime(transport, initialize)
        holder["runtime"] = runtime
        with mock.patch.object(
            self.rating_store,
            "sync",
            side_effect=BlindPoolValidationError("synthetic", "safe"),
        ), self.assertRaises(BlindPoolLifecycleError) as caught:
            await runtime.run_once()
        self.assertEqual("canonical_rating_sync_failed", caught.exception.code)
        ledger = self.result_store.require_ready()
        self.assertEqual(1, ledger.completed_count)
        self.assertEqual(0, self.rating_store.load().processed_sequence)
        self.assertEqual(1, self.rating_store.sync(ledger).state.processed_sequence)

    def test_durable_and_process_identity_mismatch_fails_before_intent(self):
        transport = FakeTransport(())
        runtime = self.runtime(transport, mock.AsyncMock())
        runtime._store.initialize_or_load()
        durable = BlindTeamPublicIdentity(
            PLAYER_TEAM_ID, "Synthetic Offense", PUBLIC_TEAM_KIND_PLAYER
        )
        reservation = runtime._store.reserve_next(
            BlindChallengeToken("a" * 32),
            durable,
        )
        conflicting = BlindPoolChallenge(
            "syntheticopponent",
            "Synthetic Opponent",
            "gen9tugs",
            "server",
            reservation.challenge_token,
            BlindTeamPublicIdentity(
                "player-team:" + "9" * 32,
                "Other Team",
                PUBLIC_TEAM_KIND_PLAYER,
            ),
        )
        room = BlindPoolBattleRoom(
            "battle-gen9tugs-401",
            "gen9tugs",
            "syntheticopponent",
            "Synthetic Opponent",
            "p2",
            "p1",
        )
        with self.assertRaises(BlindPoolLifecycleError) as caught:
            runtime._create_result_intent(reservation, conflicting, room)
        self.assertEqual("canonical_team_identity_mismatch", caught.exception.code)
        self.assertEqual(0, self.result_store.load().pending_count)


if __name__ == "__main__":
    import unittest

    unittest.main()
