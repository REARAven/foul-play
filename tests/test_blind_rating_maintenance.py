from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import unittest

from fp.data.blind_pool.rating_maintenance import main
from fp.data.blind_pool.models import BlindPoolConfig, BlindPoolStateConfig
from fp.data.blind_pool.ownership import acquire_blind_pool_deployment_owner
from fp.data.blind_pool.result_ledger import (
    OUTCOME_PLAYER_WIN,
    BlindResultLedgerStore,
    validate_result_ledger_config,
)
from tests.test_blind_canonical_registry import SyntheticCanonicalFixture


PRIVATE_SENTINEL = "PHASE6B-MAINTENANCE-PRIVATE-SENTINEL"


class RatingMaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = SyntheticCanonicalFixture()
        self.addCleanup(self.fixture.close)
        for team_id in ("BL-001-v1", "BL-002-v1"):
            self.fixture.add_artifact(team_id)
        state_directory = self.fixture.private_root / "state"
        state_directory.mkdir()
        self.state_path = state_directory / "bag.json"
        ladder_directory = self.fixture.base / "ladder"
        ladder_directory.mkdir()
        self.result_path = ladder_directory / "results.json"
        self.rating_path = ladder_directory / "ratings.json"
        result_config = validate_result_ledger_config(
            self.result_path,
            private_root=self.fixture.private_root,
            registry_path=self.fixture.registry_path,
            selection_state_path=self.state_path,
        )
        self.result_store = BlindResultLedgerStore(result_config)
        self.result_store.initialize_empty()
        self.base_args = [
            "--private-root",
            str(self.fixture.private_root),
            "--registry",
            str(self.fixture.registry_path),
            "--state",
            str(self.state_path),
            "--ledger",
            str(self.result_path),
            "--ratings",
            str(self.rating_path),
        ]

    def run_cli(self, command, *extra):
        output = io.StringIO()
        errors = io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            code = main([command, *self.base_args, *extra])
        rendered = output.getvalue() + errors.getvalue()
        for forbidden in (
            PRIVATE_SENTINEL,
            "BL-001-v1",
            "BL-002-v1",
            str(self.fixture.private_root),
            str(self.result_path),
            str(self.rating_path),
        ):
            self.assertNotIn(forbidden, rendered)
        return code, output.getvalue(), errors.getvalue()

    def complete_win(self):
        registry = self.fixture.load()
        pending = self.result_store.create_pending_intent(
            player_id="syntheticplayer",
            bot_id="syntheticbot",
            team_id="BL-001-v1",
            reservation_id="1" * 32,
            room_id="battle-gen9tugs-601",
            format_id="gen9tugs",
            registry_fingerprint=registry.registry_fingerprint,
        )
        self.result_store.mark_selection_committed(pending.battle_id)
        self.result_store.finalize_terminal(pending.battle_id, OUTCOME_PLAYER_WIN)

    def test_initialize_from_nonempty_verify_and_status_are_aggregate_only(self):
        self.complete_win()
        code, output, _errors = self.run_cli("initialize")
        self.assertEqual(0, code)
        self.assertIn("rating state: initialized", output)
        self.assertIn("processed results: 1", output)
        self.assertIn("rated results: 1", output)
        code, output, _errors = self.run_cli("verify")
        self.assertEqual(0, code)
        self.assertIn("rating state: verified", output)
        code, output, _errors = self.run_cli("status")
        self.assertEqual(0, code)
        self.assertIn("rating state: ready", output)

    def test_status_reports_behind_and_sync_catches_up(self):
        self.run_cli("initialize")
        self.complete_win()
        code, output, _errors = self.run_cli("status")
        self.assertEqual(0, code)
        self.assertIn("rating state: behind", output)
        code, output, _errors = self.run_cli("sync")
        self.assertEqual(0, code)
        self.assertIn("rating state: synchronized", output)
        self.assertIn("processed results: 1", output)

    def test_rebuild_requires_exact_confirmation_and_preserves_result_bytes(self):
        self.complete_win()
        self.run_cli("initialize")
        ledger_before = self.result_path.read_bytes()
        document = json.loads(self.rating_path.read_text(encoding="utf-8"))
        document["players"]["syntheticplayer"]["rating"] += 1
        document["players"]["syntheticplayer"]["peak_rating"] += 1
        document["opponents"]["BL-001-v1"]["rating"] -= 1
        self.rating_path.write_text(json.dumps(document) + "\n", encoding="utf-8")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["rebuild", *self.base_args])
        code, output, _errors = self.run_cli("rebuild", "--confirm", "rebuild")
        self.assertEqual(0, code)
        self.assertIn("rating state: rebuilt", output)
        self.assertEqual(ledger_before, self.result_path.read_bytes())

    def test_player_status_is_public_safe_and_unseen_player_is_baseline(self):
        self.complete_win()
        self.run_cli("initialize")
        code, output, _errors = self.run_cli(
            "player-status", "--player", "Synthetic Player"
        )
        self.assertEqual(0, code)
        self.assertIn("player rating: 1516", output)
        self.assertIn("record: 1-0-0", output)
        code, output, _errors = self.run_cli(
            "player-status", "--player", "Unseen Player"
        )
        self.assertEqual(0, code)
        self.assertIn("player rating: 1500", output)
        self.assertIn("rating status: unrated", output)

    def test_invalid_player_and_missing_state_fail_safely(self):
        code, _output, errors = self.run_cli("status")
        self.assertEqual(2, code)
        self.assertIn("rating_not_initialized", errors)
        self.run_cli("initialize")
        code, _output, errors = self.run_cli("player-status", "--player", "!!!")
        self.assertEqual(2, code)
        self.assertIn("rating_player_id_invalid", errors)

    def test_no_hidden_team_status_command_or_network_dependency_exists(self):
        source = Path("fp/data/blind_pool/rating_maintenance.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("team-status", source)
        self.assertNotIn("websocket", source.casefold())
        self.assertNotIn("requests", source.casefold())

    def test_live_deployment_owner_blocks_rating_maintenance_without_mutation(self):
        self.run_cli("initialize")
        rating_before = self.rating_path.read_bytes()
        ledger_before = self.result_path.read_bytes()
        owner = acquire_blind_pool_deployment_owner(
            BlindPoolStateConfig(
                BlindPoolConfig(
                    self.fixture.private_root,
                    self.fixture.registry_path,
                ),
                self.state_path,
            ),
            timeout_seconds=0.05,
        )
        try:
            code, _output, errors = self.run_cli("status")
        finally:
            owner.close()
        self.assertEqual(2, code)
        self.assertIn("deployment_ownership", errors)
        self.assertEqual(rating_before, self.rating_path.read_bytes())
        self.assertEqual(ledger_before, self.result_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
