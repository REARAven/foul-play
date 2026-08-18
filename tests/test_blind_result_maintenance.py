from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import unittest

from fp.data.blind_pool.result_ledger import (
    BlindResultLedgerStore,
    validate_result_ledger_config,
)
from fp.data.blind_pool.result_maintenance import main
from tests.test_blind_canonical_registry import SyntheticCanonicalFixture


PRIVATE_SENTINEL = "PHASE6A-MAINTENANCE-PRIVATE-SENTINEL"


class ResultMaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = SyntheticCanonicalFixture()
        self.addCleanup(self.fixture.close)
        for team_id in ("BL-001-v1", "BL-002-v1"):
            self.fixture.add_artifact(team_id)
        state_directory = self.fixture.private_root / "state"
        state_directory.mkdir()
        self.state_path = state_directory / "bag.json"
        result_directory = self.fixture.base / "ladder"
        result_directory.mkdir()
        self.result_path = result_directory / "results.json"
        self.base_args = [
            "--private-root",
            str(self.fixture.private_root),
            "--registry",
            str(self.fixture.registry_path),
            "--state",
            str(self.state_path),
            "--ledger",
            str(self.result_path),
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
            str(self.fixture.private_root),
            str(self.result_path),
        ):
            self.assertNotIn(forbidden, rendered)
        return code, output.getvalue(), errors.getvalue()

    def store(self):
        config = validate_result_ledger_config(
            self.result_path,
            private_root=self.fixture.private_root,
            registry_path=self.fixture.registry_path,
            selection_state_path=self.state_path,
        )
        return BlindResultLedgerStore(config)

    def create_pending(self):
        registry = self.fixture.load()
        return self.store().create_pending_intent(
            player_id="syntheticplayer",
            bot_id="syntheticbot",
            team_id="BL-001-v1",
            reservation_id="1" * 32,
            room_id="battle-gen9tugs-401",
            format_id="gen9tugs",
            registry_fingerprint=registry.registry_fingerprint,
        )

    def test_initialize_verify_and_ready_status_are_aggregate_only(self):
        code, output, _errors = self.run_cli("initialize")
        self.assertEqual(0, code)
        self.assertIn("result ledger: initialized", output)
        code, output, _errors = self.run_cli("verify")
        self.assertEqual(0, code)
        self.assertIn("completed results: 0", output)
        code, output, _errors = self.run_cli("status")
        self.assertEqual(0, code)
        self.assertIn("result ledger: ready", output)
        self.assertIn("pending results: 0", output)

    def test_pending_status_and_recovery_case_reveal_no_association(self):
        self.run_cli("initialize")
        pending = self.create_pending()
        code, output, _errors = self.run_cli("status")
        self.assertEqual(0, code)
        self.assertIn("result ledger: recovery_required", output)
        self.assertIn("pending results: 1", output)
        code, case_output, _errors = self.run_cli("recovery-case")
        self.assertEqual(0, code)
        recovery_case = case_output.strip().split()[-1]
        self.assertEqual(64, len(recovery_case))
        self.assertNotIn(pending.battle_id, output + case_output)

    def test_explicit_resolution_requires_matching_confirmation(self):
        self.run_cli("initialize")
        self.create_pending()
        recovery_case = self.store().recovery_case()
        code, _output, errors = self.run_cli(
            "resolve",
            "--case",
            recovery_case,
            "--outcome",
            "tie",
            "--confirm",
            "player-win",
        )
        self.assertEqual(2, code)
        self.assertIn("result_recovery_confirmation_mismatch", errors)
        self.assertEqual(1, self.store().load().pending_count)

    def test_explicit_resolution_is_atomic_and_clears_pending(self):
        self.run_cli("initialize")
        self.create_pending()
        recovery_case = self.store().recovery_case()
        code, output, _errors = self.run_cli(
            "resolve",
            "--case",
            recovery_case,
            "--outcome",
            "no-result",
            "--confirm",
            "no-result",
        )
        self.assertEqual(0, code)
        self.assertIn("result recovery: complete", output)
        state = self.store().load()
        self.assertEqual(1, state.completed_count)
        self.assertEqual(0, state.pending_count)

    def test_stale_case_fails_without_mutating_pending(self):
        self.run_cli("initialize")
        self.create_pending()
        code, _output, errors = self.run_cli(
            "resolve",
            "--case",
            "0" * 64,
            "--outcome",
            "player-loss",
            "--confirm",
            "player-loss",
        )
        self.assertEqual(2, code)
        self.assertIn("result_recovery_case_mismatch", errors)
        self.assertEqual(1, self.store().load().pending_count)

    def test_maintenance_module_has_no_network_dependency(self):
        source = Path("fp/data/blind_pool/result_maintenance.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("websocket", source.casefold())
        self.assertNotIn("requests", source.casefold())


if __name__ == "__main__":
    unittest.main()
