from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest import mock

from fp.data.blind_pool.errors import BlindPoolValidationError
from fp.data.blind_pool.team_ladder_maintenance import main
from tests.test_blind_canonical_registry import SyntheticCanonicalFixture


ROOT = Path(__file__).resolve().parents[1]


class TeamLadderMaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = SyntheticCanonicalFixture()
        self.addCleanup(self.fixture.close)
        for team_id in ("BL-001-v1", "BL-002-v1"):
            self.fixture.add_artifact(team_id)
        state_dir = self.fixture.private_root / "state"
        state_dir.mkdir()
        self.selection = state_dir / "selection.json"
        self.selection.write_text("{}", encoding="utf-8")
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

    def run_command(self, command, *extra):
        output = io.StringIO()
        with redirect_stdout(output):
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


if __name__ == "__main__":
    unittest.main()
