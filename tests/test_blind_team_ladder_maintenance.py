from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest import mock

from fp.data.blind_pool.bag import BlindPoolBagStore
from fp.data.blind_pool.config import validate_blind_pool_state_config
from fp.data.blind_pool.errors import BlindPoolValidationError
from fp.data.blind_pool.leaderboard import (
    BlindTeamPublicIdentity,
    PUBLIC_TEAM_KIND_PLAYER,
)
from fp.data.blind_pool.models import (
    BlindChallengeToken,
    BlindPoolConfig,
    BlindPoolStateConfig,
)
from fp.data.blind_pool.ownership import acquire_blind_pool_deployment_owner
from fp.data.blind_pool.reconciliation import derive_reconciliation_case
from fp.data.blind_pool.selection import create_canonical_selection_snapshot
from fp.data.blind_pool.team_ladder_maintenance import main
from fp.data.blind_pool.team_public_registry import (
    BlindTeamPublicRegistryStore,
    validate_team_public_registry_config,
)
from fp.data.blind_pool.team_rating_state import (
    BlindTeamRatingStateStore,
    validate_team_rating_state_config,
)
from fp.data.blind_pool.team_result_ledger import (
    TEAM_OUTCOME_A_WIN,
    BlindTeamResultLedgerStore,
    validate_team_result_ledger_config,
)
from tests.test_blind_canonical_registry import SyntheticCanonicalFixture


ROOT = Path(__file__).resolve().parents[1]
PLAYER_TEAM_ID = "player-team:" + "1" * 32


class IdentityRandom:
    def shuffle(self, values):
        return None

    def randrange(self, start, stop=None):
        return start


class TeamLadderMaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = SyntheticCanonicalFixture()
        self.addCleanup(self.fixture.close)
        for team_id in ("BL-001-v1", "BL-002-v1"):
            self.fixture.add_artifact(team_id)
        state_dir = self.fixture.private_root / "state"
        state_dir.mkdir()
        self.selection = state_dir / "selection.json"
        self.registry = self.fixture.load()
        self.selection_snapshot = create_canonical_selection_snapshot(self.registry)
        self.state_config = validate_blind_pool_state_config(
            BlindPoolStateConfig(
                BlindPoolConfig(
                    self.fixture.private_root,
                    self.fixture.registry_path,
                ),
                self.selection,
            ),
            repository_root=ROOT,
        )
        self.team_store().initialize_or_load()
        self.external = self.fixture.base / "team-ladder"
        self.external.mkdir()
        self.results = self.external / "results.json"
        self.ratings = self.external / "ratings.json"
        self.public = self.external / "public.json"
        self.arguments = [
            "--private-root",
            str(self.fixture.private_root),
            "--canonical-registry",
            str(self.fixture.registry_path),
            "--selection-state",
            str(self.selection),
            "--team-result-ledger",
            str(self.results),
            "--team-rating-state",
            str(self.ratings),
            "--team-public-registry",
            str(self.public),
            "--repository-root",
            str(ROOT),
        ]

    def team_store(self):
        return BlindPoolBagStore.from_selection_snapshot(
            self.state_config,
            self.selection_snapshot,
            random_source=IdentityRandom(),
            reservation_id_factory=lambda: "c" * 32,
            team_mode=True,
        )

    def legacy_store(self):
        if self.selection.exists():
            self.selection.unlink()
        return BlindPoolBagStore.from_selection_snapshot(
            self.state_config,
            self.selection_snapshot,
            random_source=IdentityRandom(),
            reservation_id_factory=lambda: "c" * 32,
            team_mode=False,
        )

    def stores(self):
        public = BlindTeamPublicRegistryStore(
            validate_team_public_registry_config(
                self.public,
                private_root=self.fixture.private_root,
                canonical_registry_path=self.fixture.registry_path,
                selection_state_path=self.selection,
                result_ledger_path=self.results,
                rating_state_path=self.ratings,
                repository_root=ROOT,
            )
        )
        results = BlindTeamResultLedgerStore(
            validate_team_result_ledger_config(
                self.results,
                private_root=self.fixture.private_root,
                registry_path=self.fixture.registry_path,
                selection_state_path=self.selection,
                rating_state_path=self.ratings,
                public_registry_path=self.public,
                repository_root=ROOT,
            )
        )
        ratings = BlindTeamRatingStateStore(
            validate_team_rating_state_config(
                self.ratings,
                private_root=self.fixture.private_root,
                canonical_registry_path=self.fixture.registry_path,
                selection_state_path=self.selection,
                result_ledger_path=self.results,
                public_registry_path=self.public,
                repository_root=ROOT,
            )
        )
        return public, results, ratings

    def create_pending_result(self):
        public, results, _ratings = self.stores()
        public.register_player(
            BlindTeamPublicIdentity(
                PLAYER_TEAM_ID,
                "Synthetic player team",
                PUBLIC_TEAM_KIND_PLAYER,
            )
        )
        return results, results.create_pending_intent(
            player_account_id="player",
            bot_account_id="bot",
            player_team_id=PLAYER_TEAM_ID,
            bot_team_id=self.registry.active_ids[0],
            reservation_id="a" * 32,
            room_id="battle-gen9tugs-1",
            format_id="gen9tugs",
            registry_fingerprint=self.registry.registry_fingerprint,
        )

    def create_accept_sent(self):
        self.initialize()
        player_identity = BlindTeamPublicIdentity(
            PLAYER_TEAM_ID,
            "Synthetic player team",
            PUBLIC_TEAM_KIND_PLAYER,
        )
        public, _results, _ratings = self.stores()
        public.register_player(player_identity)
        store = self.team_store()
        reservation = store.reserve_next(
            BlindChallengeToken("a" * 32),
            player_identity,
        )
        state = store.mark_accept_sent(reservation.reservation_id)
        return store, state, derive_reconciliation_case(state)

    def run_command(self, command, *extra):
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            result = main([command, *self.arguments, *extra])
        return result, output.getvalue()

    def initialize(self):
        with mock.patch(
            "fp.data.blind_pool.team_public_registry.random.SystemRandom"
        ) as source:
            source.return_value.shuffle.side_effect = lambda values: None
            return self.run_command("initialize")

    def test_initialize_creates_only_three_new_team_era_files(self):
        legacy = self.external / "legacy-account-results.json"
        legacy.write_bytes(b"frozen")
        result, output = self.initialize()
        self.assertEqual(0, result)
        self.assertEqual(b"frozen", legacy.read_bytes())
        self.assertTrue(
            all(path.exists() for path in (self.results, self.ratings, self.public))
        )
        self.assertIn("completed results: 0", output)
        self.assertIn("bot identities registered: 2", output)
        self.assertNotIn("BL-001-v1", output)

    def test_initialize_refuses_overwrite(self):
        self.initialize()
        with self.assertRaises(BlindPoolValidationError):
            self.initialize()

    def test_initialize_preflights_all_outputs_before_creating_any(self):
        self.ratings.write_text("existing", encoding="utf-8")
        with self.assertRaises(BlindPoolValidationError):
            self.initialize()
        self.assertFalse(self.public.exists())
        self.assertFalse(self.results.exists())
        self.assertEqual("existing", self.ratings.read_text(encoding="utf-8"))

    def test_status_and_verify_are_public_aggregate_only(self):
        self.initialize()
        _, status = self.run_command("status")
        _, verified = self.run_command("verify")
        self.assertIn("rated results: 0", status)
        self.assertIn("team ladder state: verified", verified)
        self.assertNotIn("BL-", status + verified)

    def test_migration_command_preserves_idle_bag_semantics_and_hides_ids(self):
        legacy = self.legacy_store()
        legacy.initialize_or_load()
        reservation = legacy.reserve_next()
        legacy.commit_reservation(reservation.reservation_id)
        before = legacy.snapshot()

        _, output = self.run_command("migrate-selection-state")
        after = self.team_store().snapshot()

        self.assertIn("selection state migration: complete", output)
        self.assertIn("selection schema: 3", output)
        self.assertNotIn("BL-", output)
        self.assertEqual(before.cycle_number, after.cycle_number)
        self.assertEqual(before.cycle_order, after.cycle_order)
        self.assertEqual(before.next_index, after.next_index)
        self.assertEqual(before.last_consumed_id, after.last_consumed_id)

    def test_migration_command_requires_and_releases_deployment_owner(self):
        legacy = self.legacy_store()
        legacy.initialize_or_load()
        owners = []

        def acquire_owner(*args, **kwargs):
            owner = acquire_blind_pool_deployment_owner(*args, **kwargs)
            owners.append(owner)
            return owner

        with mock.patch(
            "fp.data.blind_pool.team_ladder_maintenance."
            "acquire_blind_pool_deployment_owner",
            side_effect=acquire_owner,
        ) as acquire:
            self.run_command("migrate-selection-state")
        acquire.assert_called_once()
        self.assertEqual(1, len(owners))
        self.assertFalse(owners[0].held)

    def test_active_owner_blocks_migration(self):
        legacy = self.legacy_store()
        legacy.initialize_or_load()
        with acquire_blind_pool_deployment_owner(
            self.state_config,
            repository_root=ROOT,
        ):
            with self.assertRaises(BlindPoolValidationError) as caught:
                self.run_command("migrate-selection-state")
        self.assertEqual("deployment_ownership_unavailable", caught.exception.code)

    def test_migration_refuses_unresolved_legacy_reservation(self):
        legacy = self.legacy_store()
        legacy.initialize_or_load()
        legacy.reserve_next(BlindChallengeToken("a" * 32))
        with self.assertRaises(BlindPoolValidationError) as caught:
            self.run_command("migrate-selection-state")
        self.assertEqual(
            "state_schema_migration_reservation_unresolved",
            caught.exception.code,
        )

    def test_repeated_migration_is_safe_and_does_not_rewrite_state(self):
        before = self.selection.read_bytes()
        _, output = self.run_command("migrate-selection-state")
        self.assertEqual(before, self.selection.read_bytes())
        self.assertIn("selection state migration: already complete", output)

    def test_preflight_accepts_complete_idle_team_state(self):
        self.initialize()
        _, output = self.run_command("preflight")
        self.assertIn("team ladder preflight: ready", output)
        self.assertIn("selection schema: 3", output)
        self.assertIn("pending results: 0", output)
        self.assertNotIn("BL-", output)

    def test_preflight_rejects_schema_two(self):
        self.initialize()
        self.legacy_store().initialize_or_load()
        with self.assertRaises(BlindPoolValidationError) as caught:
            self.run_command("preflight")
        self.assertEqual("state_schema_unsupported", caught.exception.code)

    def test_preflight_rejects_unresolved_selection_reservation(self):
        self.initialize()
        self.team_store().reserve_next(
            BlindChallengeToken("a" * 32),
            BlindTeamPublicIdentity(
                PLAYER_TEAM_ID,
                "Synthetic player team",
                PUBLIC_TEAM_KIND_PLAYER,
            ),
        )
        with self.assertRaises(BlindPoolValidationError) as caught:
            self.run_command("preflight")
        self.assertEqual("team_selection_recovery_required", caught.exception.code)

    def test_preflight_rejects_pending_result(self):
        self.initialize()
        self.create_pending_result()
        with self.assertRaises(BlindPoolValidationError) as caught:
            self.run_command("preflight")
        self.assertEqual("team_result_recovery_required", caught.exception.code)

    def test_preflight_rejects_missing_bot_public_identity(self):
        self.initialize()
        document = json.loads(self.public.read_text(encoding="utf-8"))
        document["identities"] = document["identities"][1:]
        self.public.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaises(BlindPoolValidationError) as caught:
            self.run_command("preflight")
        self.assertEqual("team_public_registry_bot_missing", caught.exception.code)

    def test_preflight_rejects_rating_state_behind_history(self):
        self.initialize()
        results, pending = self.create_pending_result()
        results.mark_selection_committed(pending.battle_id)
        results.finalize_terminal(pending.battle_id, TEAM_OUTCOME_A_WIN)
        with self.assertRaises(BlindPoolValidationError) as caught:
            self.run_command("preflight")
        self.assertEqual("team_rating_state_behind", caught.exception.code)

    def test_preflight_rejects_divergent_rating_state(self):
        self.initialize()
        results, pending = self.create_pending_result()
        results.mark_selection_committed(pending.battle_id)
        results.finalize_terminal(pending.battle_id, TEAM_OUTCOME_A_WIN)
        _public, _results, ratings = self.stores()
        ratings.sync(results.require_ready())
        document = json.loads(self.ratings.read_text(encoding="utf-8"))
        document["teams"][PLAYER_TEAM_ID]["rating"] = 1515
        document["teams"][self.registry.active_ids[0]]["rating"] = 1485
        self.ratings.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaises(BlindPoolValidationError) as caught:
            self.run_command("preflight")
        self.assertEqual("team_rating_state_diverged", caught.exception.code)

    def test_preflight_rejects_missing_historical_public_identity(self):
        self.initialize()
        results, pending = self.create_pending_result()
        results.mark_selection_committed(pending.battle_id)
        results.finalize_terminal(pending.battle_id, TEAM_OUTCOME_A_WIN)
        document = json.loads(self.public.read_text(encoding="utf-8"))
        document["identities"] = [
            identity
            for identity in document["identities"]
            if identity["private_team_id"] != PLAYER_TEAM_ID
        ]
        self.public.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaises(BlindPoolValidationError) as caught:
            self.run_command("preflight")
        self.assertEqual(
            "team_public_registry_history_invalid",
            caught.exception.code,
        )

    def test_verify_and_status_require_schema_three(self):
        self.initialize()
        self.legacy_store().initialize_or_load()
        for command in ("verify", "status"):
            with self.subTest(command=command):
                with self.assertRaises(BlindPoolValidationError) as caught:
                    self.run_command(command)
                self.assertEqual("state_schema_unsupported", caught.exception.code)

    def test_leaderboard_contains_public_aliases_without_private_ids(self):
        self.initialize()
        _, output = self.run_command("leaderboard")
        rows = json.loads(output)
        self.assertEqual(["Bot team 01", "Bot team 02"], [row["name"] for row in rows])
        self.assertNotIn("BL-", output)
        self.assertTrue(all(row["rating"] == 1500 for row in rows))

    def test_rebuild_requires_exact_confirmation(self):
        self.initialize()
        with self.assertRaises(RuntimeError):
            self.run_command("rebuild-ratings")
        _, output = self.run_command("rebuild-ratings", "--confirm", "rebuild")
        self.assertIn("team rating state: rebuilt", output)

    def test_sync_and_bot_sync_have_safe_output(self):
        self.initialize()
        _, rating = self.run_command("sync")
        _, bots = self.run_command("sync-bot-identities")
        self.assertIn("processed results: 0", rating)
        self.assertIn("bot identities registered: 2", bots)
        self.assertNotIn("BL-", rating + bots)

    def test_recovery_status_reports_accept_sent_case_without_private_identity(self):
        _store, _state, reconciliation_case = self.create_accept_sent()

        _, output = self.run_command("recovery-status")

        self.assertIn("team selection recovery: required", output)
        self.assertIn("selection schema: 3", output)
        self.assertIn("reservation phase: accept_sent", output)
        self.assertIn("completed results: 0", output)
        self.assertIn("pending results: 0", output)
        self.assertIn("pending phase: none", output)
        self.assertIn("processed results: 0", output)
        self.assertIn("rated results: 0", output)
        self.assertIn("reconciliation case: {}".format(reconciliation_case), output)
        for private_value in (
            "BL-001-v1",
            "BL-002-v1",
            PLAYER_TEAM_ID,
            "Synthetic player team",
            "c" * 32,
            "a" * 32,
        ):
            self.assertNotIn(private_value, output)

    def test_recovery_status_distinguishes_idle_and_pre_accept_state(self):
        self.initialize()
        _, idle = self.run_command("recovery-status")
        self.assertEqual("team selection recovery: none\n", idle)

        player_identity = BlindTeamPublicIdentity(
            PLAYER_TEAM_ID,
            "Synthetic player team",
            PUBLIC_TEAM_KIND_PLAYER,
        )
        public, _results, _ratings = self.stores()
        public.register_player(player_identity)
        self.team_store().reserve_next(
            BlindChallengeToken("a" * 32),
            player_identity,
        )
        _, reserved = self.run_command("recovery-status")
        self.assertIn("team selection recovery: startup-release", reserved)
        self.assertIn("reservation phase: reserved", reserved)
        self.assertNotIn("reconciliation case:", reserved)

    def test_resolve_not_consumed_preserves_bag_and_team_history(self):
        store, before, reconciliation_case = self.create_accept_sent()
        public_before = self.public.read_bytes()
        results_before = self.results.read_bytes()
        ratings_before = self.ratings.read_bytes()

        _, output = self.run_command(
            "resolve-not-consumed",
            "--case",
            reconciliation_case,
            "--confirm",
            "not-consumed",
        )
        after = store.snapshot()

        self.assertEqual(
            "reconciliation resolved: not-consumed\nreservation pending: 0\n",
            output,
        )
        self.assertIsNone(after.reservation)
        self.assertEqual(before.cycle_number, after.cycle_number)
        self.assertEqual(before.cycle_order, after.cycle_order)
        self.assertEqual(before.next_index, after.next_index)
        self.assertEqual(before.last_consumed_id, after.last_consumed_id)
        self.assertEqual(public_before, self.public.read_bytes())
        self.assertEqual(results_before, self.results.read_bytes())
        self.assertEqual(ratings_before, self.ratings.read_bytes())
        public, results, ratings = self.stores()
        self.assertIsNotNone(public.load().identity(PLAYER_TEAM_ID))
        self.assertEqual(0, results.load().completed_count)
        self.assertEqual(0, results.load().pending_count)
        self.assertEqual(0, ratings.verify(results.load()).rated_results)
        self.assertNotIn("BL-", output)
        self.assertNotIn(PLAYER_TEAM_ID, output)

    def test_resolve_not_consumed_rejects_stale_malformed_and_missing_confirmation(
        self,
    ):
        _store, _state, reconciliation_case = self.create_accept_sent()
        stale_case = ("0" if reconciliation_case[0] != "0" else "1") + (
            reconciliation_case[1:]
        )
        with self.assertRaises(BlindPoolValidationError) as caught:
            self.run_command(
                "resolve-not-consumed",
                "--case",
                stale_case,
                "--confirm",
                "not-consumed",
            )
        self.assertEqual("reconciliation_case_mismatch", caught.exception.code)
        with self.assertRaises(BlindPoolValidationError) as caught:
            self.run_command(
                "resolve-not-consumed",
                "--case",
                "not-a-case",
                "--confirm",
                "not-consumed",
            )
        self.assertEqual("reconciliation_case_invalid", caught.exception.code)
        with self.assertRaises(BlindPoolValidationError) as caught:
            self.run_command(
                "resolve-not-consumed",
                "--case",
                reconciliation_case,
            )
        self.assertEqual(
            "team_recovery_confirmation_required",
            caught.exception.code,
        )

    def test_resolve_not_consumed_refuses_pending_team_result(self):
        _store, state, reconciliation_case = self.create_accept_sent()
        public, results, _ratings = self.stores()
        reservation = state.reservation
        self.assertIsNotNone(reservation)
        results.create_pending_intent(
            player_account_id="player",
            bot_account_id="bot",
            player_team_id=PLAYER_TEAM_ID,
            bot_team_id=reservation.team_id,
            reservation_id=reservation.reservation_id,
            room_id="battle-gen9tugs-2",
            format_id="gen9tugs",
            registry_fingerprint=self.registry.registry_fingerprint,
        )
        public_before = public.load()

        with self.assertRaises(BlindPoolValidationError) as caught:
            self.run_command(
                "resolve-not-consumed",
                "--case",
                reconciliation_case,
                "--confirm",
                "not-consumed",
            )
        self.assertEqual(
            "team_recovery_pending_result_conflict",
            caught.exception.code,
        )
        self.assertIsNotNone(self.team_store().snapshot().reservation)
        self.assertEqual(public_before, public.load())

    def test_resolve_not_consumed_rejects_completed_result_evidence(self):
        _store, state, reconciliation_case = self.create_accept_sent()
        _public, results, ratings = self.stores()
        reservation = state.reservation
        self.assertIsNotNone(reservation)
        pending = results.create_pending_intent(
            player_account_id="player",
            bot_account_id="bot",
            player_team_id=PLAYER_TEAM_ID,
            bot_team_id=reservation.team_id,
            reservation_id=reservation.reservation_id,
            room_id="battle-gen9tugs-3",
            format_id="gen9tugs",
            registry_fingerprint=self.registry.registry_fingerprint,
        )
        results.mark_selection_committed(pending.battle_id)
        results.finalize_terminal(pending.battle_id, TEAM_OUTCOME_A_WIN)
        ratings.sync(results.require_ready())

        with self.assertRaises(BlindPoolValidationError) as caught:
            self.run_command(
                "resolve-not-consumed",
                "--case",
                reconciliation_case,
                "--confirm",
                "not-consumed",
            )
        self.assertEqual(
            "team_recovery_result_evidence_conflict",
            caught.exception.code,
        )
        self.assertIsNotNone(self.team_store().snapshot().reservation)

    def test_team_recovery_rejects_schema_two_and_active_owner(self):
        self.initialize()
        self.legacy_store().initialize_or_load()
        with self.assertRaises(BlindPoolValidationError) as caught:
            self.run_command("recovery-status")
        self.assertEqual("state_schema_unsupported", caught.exception.code)

        self.selection.unlink()
        self.team_store().initialize_or_load()
        with acquire_blind_pool_deployment_owner(
            self.state_config,
            repository_root=ROOT,
        ):
            with self.assertRaises(BlindPoolValidationError) as caught:
                self.run_command("recovery-status")
        self.assertEqual("deployment_ownership_unavailable", caught.exception.code)

    def test_resolved_case_is_not_reusable(self):
        _store, _state, reconciliation_case = self.create_accept_sent()
        arguments = (
            "--case",
            reconciliation_case,
            "--confirm",
            "not-consumed",
        )
        self.run_command("resolve-not-consumed", *arguments)
        with self.assertRaises(BlindPoolValidationError) as caught:
            self.run_command("resolve-not-consumed", *arguments)
        self.assertEqual("reconciliation_not_applicable", caught.exception.code)


if __name__ == "__main__":
    unittest.main()
