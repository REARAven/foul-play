from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import gc
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock
import weakref

import requests

import fp.data.blind_pool.canonical_artifacts as canonical_artifacts
import fp.data.blind_pool.canonical_registry as canonical_registry
import fp.data.blind_pool.maintenance as maintenance
from fp.data.blind_pool.bag import BlindPoolBagStore
from fp.data.blind_pool.canonical_registry import load_canonical_runtime_registry
from fp.data.blind_pool.maintenance import (
    BlindCanonicalInspectionStatus,
    inspect_blind_canonical_deployment,
    verify_active_blind_canonical_artifacts,
)
from fp.data.blind_pool.models import (
    BlindChallengeToken,
    BlindPoolConfig,
    BlindPoolStateConfig,
)
from fp.data.blind_pool.ownership import acquire_blind_pool_deployment_owner
from fp.data.blind_pool.startup import (
    BlindCanonicalActivationError,
    BlindCanonicalErrorCategory,
    BlindCanonicalStartupConfig,
)
from fp.websocket_client import PSWebsocketClient
from tests.test_blind_canonical_registry import SyntheticCanonicalFixture


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_SENTINEL = "PHASE5E2A-SYNTHETIC-PRIVATE-SENTINEL"
TOKEN = "a" * 32


class MaintenanceFixture(unittest.TestCase):
    def setUp(self):
        self.fixture = SyntheticCanonicalFixture()
        self.addCleanup(self.fixture.close)
        self.fixture.add_artifact("BL-002-v1")
        self.fixture.add_artifact("BL-001-v1")
        self.state_directory = self.fixture.private_root / "state"
        self.state_directory.mkdir()
        self.state_path = self.state_directory / "bag.json"
        self.config = BlindCanonicalStartupConfig(
            self.fixture.private_root,
            self.fixture.registry_path,
            self.state_path,
        )

    def state_config(self):
        return BlindPoolStateConfig(
            BlindPoolConfig(
                self.fixture.private_root,
                self.fixture.registry_path,
            ),
            self.state_path,
        )

    def store(self):
        return BlindPoolBagStore.from_canonical_registry(
            self.state_config(),
            load_canonical_runtime_registry(
                self.fixture.private_root,
                self.fixture.registry_path,
            ),
        )

    def initialize(self):
        store = self.store()
        store.initialize_or_load()
        return store

    def assert_safe_error(self, error, *, category=None, code=None):
        if category is not None:
            self.assertEqual(category, error.category)
        if code is not None:
            self.assertEqual(code, error.code)
        rendered = str(error) + repr(error)
        for forbidden in (
            PRIVATE_SENTINEL,
            TOKEN,
            str(self.fixture.private_root),
            str(self.fixture.registry_path),
            str(self.state_path),
            "metadata_sha256",
            "packed",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertIsNone(error.__cause__)
        self.assertIsNone(error.__context__)

    def cli_args(self, command, *, registry=None, plan=None):
        args = [
            command,
            "--private-root",
            str(self.fixture.private_root),
            "--registry",
            str(registry or self.fixture.registry_path),
            "--state",
            str(self.state_path),
        ]
        if plan is not None:
            args.extend(("--plan", str(plan)))
        return args


class TestBlindCanonicalReadOnlyPreflight(MaintenanceFixture):
    def test_absent_state_reports_first_start_without_creating_state_or_full_read(self):
        real_read = canonical_registry._stable_read_bytes

        def guarded(path, **options):
            if Path(path).name in {"packed.txt", "team.json"}:
                raise AssertionError("full artifact read")
            return real_read(path, **options)

        with mock.patch.object(canonical_registry, "_stable_read_bytes", guarded):
            result = inspect_blind_canonical_deployment(self.config)
        self.assertEqual(
            BlindCanonicalInspectionStatus.READY_FOR_FIRST_START,
            result.status,
        )
        self.assertTrue(result.ready)
        self.assertFalse(self.state_path.exists())
        self.assertEqual(2, result.active_count)

    def test_idle_state_reports_ready_without_rewriting_bytes(self):
        self.initialize()
        before = self.state_path.read_bytes()
        result = inspect_blind_canonical_deployment(self.config)
        self.assertEqual(BlindCanonicalInspectionStatus.READY, result.status)
        self.assertEqual(before, self.state_path.read_bytes())

    def test_reserved_state_is_observational_and_startup_recoverable(self):
        store = self.initialize()
        store.reserve_next(BlindChallengeToken(TOKEN))
        before = self.state_path.read_bytes()
        with mock.patch.object(
            BlindPoolBagStore,
            "release_reservation",
            side_effect=AssertionError("reserved recovery attempted"),
        ), mock.patch.object(
            BlindPoolBagStore,
            "initialize_or_load",
            side_effect=AssertionError("startup factory called"),
        ):
            result = inspect_blind_canonical_deployment(self.config)
        self.assertEqual(
            BlindCanonicalInspectionStatus.STARTUP_RECOVERY_AVAILABLE,
            result.status,
        )
        self.assertTrue(result.ready)
        self.assertEqual(before, self.state_path.read_bytes())

    def test_accept_sent_requires_recovery_without_mutation_or_token_output(self):
        store = self.initialize()
        reservation = store.reserve_next(BlindChallengeToken(TOKEN))
        store.mark_accept_sent(reservation.reservation_id, BlindChallengeToken(TOKEN))
        before = self.state_path.read_bytes()
        result = inspect_blind_canonical_deployment(self.config)
        self.assertEqual(
            BlindCanonicalInspectionStatus.RECOVERY_REQUIRED,
            result.status,
        )
        self.assertFalse(result.ready)
        self.assertEqual(before, self.state_path.read_bytes())

        output = io.StringIO()
        errors = io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            exit_code = maintenance.main(self.cli_args("preflight"))
        self.assertEqual(2, exit_code)
        rendered = output.getvalue() + errors.getvalue()
        self.assertIn("recovery_required", rendered)
        self.assertNotIn(TOKEN, rendered)
        self.assertEqual(before, self.state_path.read_bytes())

    def test_schema_v1_malformed_and_fingerprint_mismatch_are_safe_and_immutable(self):
        self.initialize()
        valid = json.loads(self.state_path.read_bytes())
        cases = (
            (
                {**valid, "schema_version": 1},
                BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
            ),
            (
                {**valid, "schema_version": 3},
                BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
            ),
            (
                {**valid, "registry_fingerprint": "f" * 64},
                BlindCanonicalErrorCategory.RECOVERY_REQUIRED,
            ),
            (None, BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY),
        )
        for index, (document, category) in enumerate(cases):
            payload = (
                (json.dumps(document, separators=(",", ":")) + "\n").encode()
                if document is not None
                else ("{" + PRIVATE_SENTINEL).encode()
            )
            self.state_path.write_bytes(payload)
            with self.subTest(index=index):
                with self.assertRaises(BlindCanonicalActivationError) as caught:
                    inspect_blind_canonical_deployment(self.config)
                self.assert_safe_error(caught.exception, category=category)
                self.assertEqual(payload, self.state_path.read_bytes())

    def test_runtime_owner_contention_reports_deployment_in_use_then_recovers(self):
        owner = acquire_blind_pool_deployment_owner(
            self.state_config(),
            timeout_seconds=0.05,
        )
        try:
            with self.assertRaises(BlindCanonicalActivationError) as caught:
                inspect_blind_canonical_deployment(
                    self.config,
                    owner_timeout_seconds=0.05,
                )
            self.assert_safe_error(
                caught.exception,
                category=BlindCanonicalErrorCategory.DEPLOYMENT_OWNERSHIP,
                code="blind_canonical_deployment_in_use",
            )
            self.assertFalse(self.state_path.exists())
        finally:
            owner.close()
        result = inspect_blind_canonical_deployment(self.config)
        self.assertEqual(
            BlindCanonicalInspectionStatus.READY_FOR_FIRST_START,
            result.status,
        )

    def test_preflight_result_repr_is_aggregate_only(self):
        result = inspect_blind_canonical_deployment(self.config)
        rendered = repr(result)
        for forbidden in (
            "BL-001-v1",
            "BL-002-v1",
            str(self.fixture.private_root),
            str(self.state_path),
            TOKEN,
        ):
            self.assertNotIn(forbidden, rendered)

    def test_unexpected_inspection_error_remains_distinct_and_releases_owner(self):
        with mock.patch.object(
            maintenance,
            "_inspect_state_read_only",
            side_effect=AttributeError(PRIVATE_SENTINEL),
        ):
            with self.assertRaisesRegex(AttributeError, PRIVATE_SENTINEL):
                inspect_blind_canonical_deployment(self.config)
        owner = acquire_blind_pool_deployment_owner(
            self.state_config(),
            timeout_seconds=0.05,
        )
        owner.close()


class TestBlindCanonicalArtifactVerification(MaintenanceFixture):
    def test_verifies_every_active_artifact_once_in_deterministic_order(self):
        loaded = []
        real_loader = canonical_artifacts.load_canonical_team_artifact

        def record(registry, team_id):
            loaded.append(team_id)
            return real_loader(registry, team_id)

        with mock.patch.object(maintenance, "load_canonical_team_artifact", record):
            result = verify_active_blind_canonical_artifacts(self.config)
        self.assertEqual(2, result.active_count)
        self.assertEqual(["BL-001-v1", "BL-002-v1"], loaded)
        self.assertFalse(self.state_path.exists())
        self.assertNotIn("BL-001-v1", repr(result))

    def test_absent_idle_reserved_and_accept_sent_state_are_never_mutated(self):
        result = verify_active_blind_canonical_artifacts(self.config)
        self.assertEqual(2, result.active_count)
        self.assertFalse(self.state_path.exists())

        store = self.initialize()
        idle = self.state_path.read_bytes()
        verify_active_blind_canonical_artifacts(self.config)
        self.assertEqual(idle, self.state_path.read_bytes())

        reservation = store.reserve_next(BlindChallengeToken(TOKEN))
        reserved = self.state_path.read_bytes()
        verify_active_blind_canonical_artifacts(self.config)
        self.assertEqual(reserved, self.state_path.read_bytes())

        store.mark_accept_sent(
            reservation.reservation_id,
            BlindChallengeToken(TOKEN),
        )
        accept_sent = self.state_path.read_bytes()
        verify_active_blind_canonical_artifacts(self.config)
        self.assertEqual(accept_sent, self.state_path.read_bytes())

    def test_inactive_full_artifact_is_not_read(self):
        self.fixture.add_artifact("BL-003-v1")
        self.fixture.entries[1]["active"] = False
        self.fixture.write_registry()
        inactive_id = self.fixture.entries[1]["team_id"]
        inactive_packed = self.fixture.schema_root / inactive_id / "packed.txt"
        inactive_packed.write_bytes(PRIVATE_SENTINEL.encode())
        loaded = []
        real_loader = canonical_artifacts.load_canonical_team_artifact

        def record(registry, team_id):
            loaded.append(team_id)
            return real_loader(registry, team_id)

        with mock.patch.object(maintenance, "load_canonical_team_artifact", record):
            result = verify_active_blind_canonical_artifacts(self.config)
        self.assertEqual(2, result.active_count)
        self.assertNotIn(inactive_id, loaded)

    def test_first_corrupt_active_artifact_stops_and_releases_owner(self):
        self.fixture.add_artifact("BL-003-v1")
        corrupt = self.fixture.schema_root / "BL-002-v1" / "packed.txt"
        corrupt.write_bytes(PRIVATE_SENTINEL.encode())
        before = self.state_path.read_bytes() if self.state_path.exists() else None
        loaded = []
        real_loader = canonical_artifacts.load_canonical_team_artifact

        def record(registry, team_id):
            loaded.append(team_id)
            return real_loader(registry, team_id)

        with mock.patch.object(maintenance, "load_canonical_team_artifact", record):
            with self.assertRaises(BlindCanonicalActivationError) as caught:
                verify_active_blind_canonical_artifacts(self.config)
        self.assert_safe_error(
            caught.exception,
            category=BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
            code="blind_canonical_artifact_verification_failed",
        )
        self.assertEqual(["BL-001-v1", "BL-002-v1"], loaded)
        self.assertNotIn("BL-003-v1", loaded)
        after = self.state_path.read_bytes() if self.state_path.exists() else None
        self.assertEqual(before, after)
        owner = acquire_blind_pool_deployment_owner(
            self.state_config(),
            timeout_seconds=0.05,
        )
        owner.close()

    def test_previous_artifact_reference_is_dropped_before_next_load(self):
        real_loader = canonical_artifacts.load_canonical_team_artifact
        references = []

        class TrackedArtifact:
            pass

        def tracked(registry, team_id):
            if references:
                gc.collect()
                self.assertIsNone(references[-1]())
            wrapper = TrackedArtifact()
            wrapper.artifact = real_loader(registry, team_id)
            references.append(weakref.ref(wrapper))
            return wrapper

        with mock.patch.object(maintenance, "load_canonical_team_artifact", tracked):
            verify_active_blind_canonical_artifacts(self.config)
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))

    def test_verifier_owner_contention_does_not_read_artifacts(self):
        owner = acquire_blind_pool_deployment_owner(
            self.state_config(),
            timeout_seconds=0.05,
        )
        loader = mock.Mock(side_effect=AssertionError("artifact read"))
        try:
            with mock.patch.object(
                maintenance,
                "load_canonical_team_artifact",
                loader,
            ):
                with self.assertRaises(BlindCanonicalActivationError) as caught:
                    verify_active_blind_canonical_artifacts(
                        self.config,
                        owner_timeout_seconds=0.05,
                    )
            self.assert_safe_error(
                caught.exception,
                category=BlindCanonicalErrorCategory.DEPLOYMENT_OWNERSHIP,
            )
            loader.assert_not_called()
        finally:
            owner.close()


class TestBlindCanonicalMaintenanceCli(MaintenanceFixture):
    def test_command_surface_contains_only_reviewed_operations(self):
        help_text = maintenance._parser().format_help()
        for command in (
            "build-registry",
            "preflight",
            "verify-artifacts",
            "status",
            "resolve-consumed",
            "resolve-not-consumed",
        ):
            self.assertIn(command, help_text)
        for forbidden in (
            "release-accept-sent",
            "commit-accept-sent",
            "reset-state",
            "edit-state",
            "force-release",
        ):
            self.assertNotIn(forbidden, help_text)

    def test_success_output_is_aggregate_and_private_safe(self):
        output = io.StringIO()
        errors = io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            exit_code = maintenance.main(self.cli_args("preflight"))
        self.assertEqual(0, exit_code)
        rendered = output.getvalue() + errors.getvalue()
        self.assertIn("ready_for_first_start", rendered)
        for forbidden in (
            "BL-001-v1",
            "BL-002-v1",
            str(self.fixture.private_root),
            TOKEN,
            "sha256",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_all_commands_are_network_free(self):
        plan_path = self.fixture.base / "plan.json"
        output_registry = self.fixture.private_root / "generated-registry.json"
        plan_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "registry_version": "1.0.0",
                    "format_id": "gen9tugs",
                    "entries": [
                        {"team_id": "BL-001-v1", "active": True},
                        {"team_id": "BL-002-v1", "active": True},
                    ],
                }
            ),
            encoding="utf-8",
        )
        websocket = mock.AsyncMock(side_effect=AssertionError("network"))
        request = mock.Mock(side_effect=AssertionError("network"))
        with mock.patch.object(
            PSWebsocketClient, "create", websocket
        ), mock.patch.object(
            requests.sessions.Session,
            "request",
            request,
        ):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(
                    0,
                    maintenance.main(
                        self.cli_args(
                            "build-registry",
                            registry=output_registry,
                            plan=plan_path,
                        )
                    ),
                )
                self.assertEqual(
                    0,
                    maintenance.main(
                        self.cli_args("preflight", registry=output_registry)
                    ),
                )
                self.assertEqual(
                    0,
                    maintenance.main(
                        self.cli_args("verify-artifacts", registry=output_registry)
                    ),
                )
        websocket.assert_not_awaited()
        request.assert_not_called()

    def test_failure_output_has_only_stable_category_and_code(self):
        invalid = BlindCanonicalStartupConfig(
            self.fixture.private_root,
            self.fixture.base / PRIVATE_SENTINEL,
            self.state_path,
        )
        output = io.StringIO()
        errors = io.StringIO()
        args = self.cli_args("preflight", registry=invalid.canonical_registry_path)
        with redirect_stdout(output), redirect_stderr(errors):
            exit_code = maintenance.main(args)
        self.assertEqual(2, exit_code)
        rendered = output.getvalue() + errors.getvalue()
        self.assertIn("deployment_integrity", rendered)
        self.assertNotIn(PRIVATE_SENTINEL, rendered)
        self.assertNotIn(str(self.fixture.private_root), rendered)

    def test_module_entrypoint_help_is_offline_and_import_safe(self):
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "TUGS_BLIND_POOL_ROOT": PRIVATE_SENTINEL,
                "TUGS_BLIND_POOL_STATE": PRIVATE_SENTINEL,
                "TUGS_BLIND_CANONICAL_REGISTRY": PRIVATE_SENTINEL,
            }
        )
        process = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "fp.data.blind_pool.maintenance",
                "--help",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(0, process.returncode)
        self.assertIn("build-registry", process.stdout)
        self.assertNotIn(PRIVATE_SENTINEL, process.stdout + process.stderr)
