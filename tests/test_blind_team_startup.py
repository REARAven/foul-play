from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from fp.data.blind_pool import (
    BlindCanonicalActivationError,
    BlindCanonicalStartupConfig,
    BlindPoolBagStore,
    BlindPoolConfig,
    BlindPoolStateConfig,
    BlindTeamPublicRegistryStore,
    BlindTeamRatingStateStore,
    BlindTeamResultLedgerStore,
    STATE_SCHEMA_VERSION,
    TEAM_LADDER_ENABLED_ENV,
    TEAM_OUTCOME_A_WIN,
    create_canonical_selection_snapshot,
    load_blind_canonical_startup_config,
    prepare_blind_canonical_deployment,
    validate_team_public_registry_config,
    validate_team_rating_state_config,
    validate_team_result_ledger_config,
)
from fp.data.blind_pool.config import PRIVATE_ROOT_ENV, STATE_PATH_ENV
from fp.data.blind_pool.startup import CANONICAL_REGISTRY_PATH_ENV
from fp.data.blind_pool.team_public_registry import TEAM_PUBLIC_REGISTRY_PATH_ENV
from fp.data.blind_pool.team_rating_state import TEAM_RATING_STATE_PATH_ENV
from fp.data.blind_pool.team_result_ledger import TEAM_RESULT_LEDGER_PATH_ENV
from tests.test_blind_canonical_registry import SyntheticCanonicalFixture
from tests.test_blind_pool_lifecycle import IdentityRandom


ROOT = Path(__file__).resolve().parents[1]


class TeamStartupFixture(unittest.TestCase):
    def setUp(self):
        self.fixture = SyntheticCanonicalFixture()
        self.addCleanup(self.fixture.close)
        for team_id in ("BL-001-v1", "BL-002-v1"):
            self.fixture.add_artifact(team_id)
        self.registry = self.fixture.load()
        state_dir = self.fixture.private_root / "state"
        state_dir.mkdir()
        self.state_path = state_dir / "selection.json"
        self.state_config = BlindPoolStateConfig(
            BlindPoolConfig(
                self.fixture.private_root.resolve(),
                self.fixture.registry_path.resolve(),
            ),
            self.state_path.resolve(),
        )
        BlindPoolBagStore.from_selection_snapshot(
            self.state_config,
            create_canonical_selection_snapshot(self.registry),
            random_source=IdentityRandom(),
            team_mode=True,
        ).initialize_or_load()
        self.external = self.fixture.base / "team-ladder"
        self.external.mkdir()
        self.public_path = self.external / "public.json"
        self.result_path = self.external / "results.json"
        self.rating_path = self.external / "ratings.json"
        self.public_store = BlindTeamPublicRegistryStore(
            validate_team_public_registry_config(
                self.public_path,
                private_root=self.fixture.private_root,
                canonical_registry_path=self.fixture.registry_path,
                selection_state_path=self.state_path,
                result_ledger_path=self.result_path,
                rating_state_path=self.rating_path,
                repository_root=ROOT,
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
                repository_root=ROOT,
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
                repository_root=ROOT,
            )
        )
        self.rating_store.initialize(ledger)
        self.config = BlindCanonicalStartupConfig(
            self.fixture.private_root,
            self.fixture.registry_path,
            self.state_path,
            team_result_ledger_path=self.result_path,
            team_rating_state_path=self.rating_path,
            team_public_registry_path=self.public_path,
            team_ladder_enabled=True,
        )


class TeamStartupTests(TeamStartupFixture):
    def test_enabled_startup_requires_and_returns_only_team_era_stores(self):
        prepared = prepare_blind_canonical_deployment(self.config, repository_root=ROOT)
        self.addCleanup(prepared.close)
        self.assertTrue(prepared._team_mode)
        self.assertIs(prepared._result_store.__class__, BlindTeamResultLedgerStore)
        self.assertIs(prepared._rating_store.__class__, BlindTeamRatingStateStore)
        self.assertEqual(
            STATE_SCHEMA_VERSION, prepared._store.snapshot().schema_version
        )

    def test_team_startup_does_not_read_or_write_legacy_account_files(self):
        old_results = self.external / "old-results.json"
        old_ratings = self.external / "old-ratings.json"
        old_results.write_bytes(b"frozen-account-results")
        old_ratings.write_bytes(b"frozen-account-ratings")
        configured = BlindCanonicalStartupConfig(
            self.fixture.private_root,
            self.fixture.registry_path,
            self.state_path,
            old_results,
            old_ratings,
            self.result_path,
            self.rating_path,
            self.public_path,
            True,
        )
        prepared = prepare_blind_canonical_deployment(configured, repository_root=ROOT)
        prepared.close()
        self.assertEqual(b"frozen-account-results", old_results.read_bytes())
        self.assertEqual(b"frozen-account-ratings", old_ratings.read_bytes())

    def test_team_startup_refuses_pending_result_before_network(self):
        self.result_store.create_pending_intent(
            player_account_id="player",
            bot_account_id="bot",
            player_team_id="player-team:" + "1" * 32,
            bot_team_id="BL-001-v1",
            reservation_id="a" * 32,
            room_id="battle-gen9tugs-1",
            format_id="gen9tugs",
            registry_fingerprint=self.registry.registry_fingerprint,
        )
        with self.assertRaises(BlindCanonicalActivationError) as caught:
            prepare_blind_canonical_deployment(self.config, repository_root=ROOT)
        self.assertEqual("blind_result_recovery_required", caught.exception.code)

    def test_team_startup_refuses_missing_rating_initialization(self):
        self.rating_path.unlink()
        with self.assertRaises(BlindCanonicalActivationError) as caught:
            prepare_blind_canonical_deployment(self.config, repository_root=ROOT)
        self.assertEqual("blind_rating_initialization_required", caught.exception.code)

    def test_team_startup_refuses_public_registry_missing_active_bot(self):
        document = self.public_store.load()
        raw = json.loads(self.public_path.read_text(encoding="utf-8"))
        raw["identities"] = raw["identities"][1:]
        self.public_path.write_text(json.dumps(raw), encoding="utf-8")
        self.assertGreater(document.bot_count, 1)
        with self.assertRaises(BlindCanonicalActivationError) as caught:
            prepare_blind_canonical_deployment(self.config, repository_root=ROOT)
        self.assertEqual("blind_team_public_registry_invalid", caught.exception.code)

    def test_team_startup_refuses_historical_team_missing_public_identity(self):
        pending = self.result_store.create_pending_intent(
            player_account_id="player",
            bot_account_id="bot",
            player_team_id="player-team:" + "1" * 32,
            bot_team_id="BL-001-v1",
            reservation_id="a" * 32,
            room_id="battle-gen9tugs-1",
            format_id="gen9tugs",
            registry_fingerprint=self.registry.registry_fingerprint,
        )
        self.result_store.mark_selection_committed(pending.battle_id)
        self.result_store.finalize_terminal(pending.battle_id, TEAM_OUTCOME_A_WIN)
        with self.assertRaises(BlindCanonicalActivationError) as caught:
            prepare_blind_canonical_deployment(self.config, repository_root=ROOT)
        self.assertEqual("blind_team_public_registry_invalid", caught.exception.code)


class TeamStartupEnvironmentTests(unittest.TestCase):
    def paths(self):
        base = Path(tempfile.gettempdir()).resolve()
        return {
            PRIVATE_ROOT_ENV: str(base / "private"),
            CANONICAL_REGISTRY_PATH_ENV: str(base / "canonical.json"),
            STATE_PATH_ENV: str(base / "state.json"),
            TEAM_RESULT_LEDGER_PATH_ENV: str(base / "team-results.json"),
            TEAM_RATING_STATE_PATH_ENV: str(base / "team-ratings.json"),
            TEAM_PUBLIC_REGISTRY_PATH_ENV: str(base / "team-public.json"),
        }

    def test_gate_one_selects_only_new_team_paths(self):
        environment = self.paths()
        environment[TEAM_LADDER_ENABLED_ENV] = "1"
        config = load_blind_canonical_startup_config(environment)
        self.assertTrue(config.team_ladder_enabled)
        self.assertIsNone(config.result_ledger_path)
        self.assertIsNone(config.rating_state_path)
        self.assertEqual(
            Path(environment[TEAM_RESULT_LEDGER_PATH_ENV]),
            config.team_result_ledger_path,
        )

    def test_gate_absent_preserves_legacy_configuration_contract(self):
        environment = self.paths()
        environment["TUGS_BLIND_RESULT_LEDGER"] = str(
            Path(tempfile.gettempdir()) / "old-results.json"
        )
        environment["TUGS_BLIND_RATING_STATE"] = str(
            Path(tempfile.gettempdir()) / "old-ratings.json"
        )
        config = load_blind_canonical_startup_config(environment)
        self.assertFalse(config.team_ladder_enabled)
        self.assertIsNone(config.team_result_ledger_path)
        self.assertIsNone(config.team_public_registry_path)

    def test_invalid_gate_fails_without_filesystem_access(self):
        environment = self.paths()
        environment[TEAM_LADDER_ENABLED_ENV] = "true"
        with self.assertRaises(BlindCanonicalActivationError) as caught:
            load_blind_canonical_startup_config(environment)
        self.assertEqual("blind_team_ladder_gate_invalid", caught.exception.code)


if __name__ == "__main__":
    unittest.main()
