from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import unittest
from unittest import mock

import fp.data.blind_pool.canonical_registry as canonical_registry
import fp.data.blind_pool.ownership as ownership
import fp.data.blind_pool.registry_generation as registry_generation
from fp.data.blind_pool.canonical_models import CanonicalArtifactError
from fp.data.blind_pool.canonical_registry import load_canonical_runtime_registry
from fp.data.blind_pool.errors import BlindPoolValidationError
from fp.data.blind_pool.models import BlindPoolConfig, BlindPoolStateConfig
from fp.data.blind_pool.ownership import acquire_blind_pool_deployment_owner
from fp.data.blind_pool.registry_generation import (
    CANONICAL_DEPLOYMENT_PLAN_SCHEMA_VERSION,
    CanonicalRegistryBuildConfig,
    build_canonical_runtime_registry,
    parse_canonical_deployment_plan_bytes,
)
from fp.data.blind_pool.startup import (
    BlindCanonicalActivationError,
    BlindCanonicalErrorCategory,
)
from tests.test_blind_canonical_registry import SyntheticCanonicalFixture


PRIVATE_SENTINEL = "PHASE5E2A-SYNTHETIC-PRIVATE-SENTINEL"


def plan_document(entries, *, registry_version="1.0.0", **overrides):
    document = {
        "schema_version": CANONICAL_DEPLOYMENT_PLAN_SCHEMA_VERSION,
        "registry_version": registry_version,
        "format_id": "gen9tugs",
        "entries": entries,
    }
    document.update(overrides)
    return document


def plan_bytes(entries, *, registry_version="1.0.0", **overrides):
    return (
        json.dumps(
            plan_document(
                entries,
                registry_version=registry_version,
                **overrides,
            ),
            ensure_ascii=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


class RegistryGenerationFixture(unittest.TestCase):
    def setUp(self):
        self.fixture = SyntheticCanonicalFixture()
        self.addCleanup(self.fixture.close)
        self.fixture.add_artifact("BL-002-v1")
        self.fixture.add_artifact("BL-001-v1")
        self.state_directory = self.fixture.private_root / "state"
        self.state_directory.mkdir()
        self.state_path = self.state_directory / "bag.json"
        self.plan_path = self.fixture.base / "deployment-plan.json"
        self.output_path = self.fixture.private_root / "runtime-registry.json"
        self.entries = [
            {"team_id": "BL-002-v1", "active": True},
            {"team_id": "BL-001-v1", "active": True},
        ]
        self.write_plan(self.entries)

    def write_plan(self, entries, **options):
        self.plan_path.write_bytes(plan_bytes(entries, **options))

    def config(self, *, output_path=None, state_path=None, plan_path=None):
        return CanonicalRegistryBuildConfig(
            self.fixture.private_root,
            plan_path or self.plan_path,
            output_path or self.output_path,
            state_path or self.state_path,
        )

    def build(self, **options):
        return build_canonical_runtime_registry(self.config(), **options)

    def assert_safe_error(self, error, *, category=None, code=None):
        if category is not None:
            self.assertEqual(category, error.category)
        if code is not None:
            self.assertEqual(code, error.code)
        rendered = str(error) + repr(error)
        for forbidden in (
            PRIVATE_SENTINEL,
            str(self.fixture.private_root),
            str(self.plan_path),
            str(self.output_path),
            "metadata_sha256",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertIsNone(error.__cause__)
        self.assertIsNone(error.__context__)


class TestCanonicalDeploymentPlan(RegistryGenerationFixture):
    def test_exact_plan_schema_is_membership_only_and_sorted(self):
        plan = parse_canonical_deployment_plan_bytes(plan_bytes(self.entries))
        self.assertEqual(1, plan.schema_version)
        self.assertEqual("1.0.0", plan.registry_version)
        self.assertEqual("gen9tugs", plan.format_id)
        self.assertEqual(
            ("BL-001-v1", "BL-002-v1"),
            tuple(entry.team_id for entry in plan.entries),
        )
        self.assertEqual(2, plan.active_count)
        rendered = repr(plan) + "".join(repr(entry) for entry in plan.entries)
        self.assertNotIn("BL-001-v1", rendered)
        self.assertNotIn("BL-002-v1", rendered)

    def test_multiple_committed_registry_versions_are_accepted_exactly(self):
        for version in ("1", "1.0.0", "12.003.4"):
            with self.subTest(version=version):
                plan = parse_canonical_deployment_plan_bytes(
                    plan_bytes(self.entries, registry_version=version)
                )
                self.assertEqual(version, plan.registry_version)

    def test_plan_rejects_strict_schema_failures_without_raw_plan_leakage(self):
        valid = plan_document(self.entries)
        cases = []
        unknown = dict(valid, unexpected=PRIVATE_SENTINEL)
        cases.append(json.dumps(unknown).encode())
        missing = dict(valid)
        del missing["format_id"]
        cases.append(json.dumps(missing).encode())
        unknown_entry = plan_document(
            [{"team_id": "BL-001-v1", "active": True, "extra": PRIVATE_SENTINEL}]
        )
        cases.append(json.dumps(unknown_entry).encode())
        cases.extend(
            (
                b'{"schema_version":1,"schema_version":1}',
                plan_bytes(
                    [
                        {"team_id": "BL-001-v1", "active": True},
                        {"team_id": "BL-001-v1", "active": False},
                    ]
                ),
                plan_bytes([{"team_id": "descriptive", "active": True}]),
                plan_bytes([{"team_id": "../BL-001-v1", "active": True}]),
                plan_bytes([{"team_id": "BL-001-v1", "active": 1}]),
                plan_bytes(self.entries, registry_version="01.0"),
                plan_bytes(self.entries, format_id="gen9ou"),
                plan_bytes(self.entries, schema_version=2),
                b"\xff",
                b"\xef\xbb\xbf{}",
                b"{",
                b'{"schema_version":NaN}',
            )
        )
        for index, raw in enumerate(cases):
            with self.subTest(index=index):
                with self.assertRaises(BlindCanonicalActivationError) as caught:
                    parse_canonical_deployment_plan_bytes(raw)
                self.assert_safe_error(
                    caught.exception,
                    category=BlindCanonicalErrorCategory.CONFIGURATION,
                )

    def test_plan_requires_nonempty_membership_and_one_active_entry(self):
        for entries in ([], [{"team_id": "BL-001-v1", "active": False}]):
            with self.subTest(entries=len(entries)):
                with self.assertRaises(BlindCanonicalActivationError) as caught:
                    parse_canonical_deployment_plan_bytes(plan_bytes(entries))
                self.assert_safe_error(caught.exception)

    def test_plan_config_and_result_reprs_are_path_and_id_safe(self):
        plan = parse_canonical_deployment_plan_bytes(plan_bytes(self.entries))
        config = self.config()
        result = self.build()
        rendered = repr(plan) + repr(config) + repr(result)
        for forbidden in (
            str(self.fixture.private_root),
            str(self.plan_path),
            str(self.output_path),
            "BL-001-v1",
            "BL-002-v1",
        ):
            self.assertNotIn(forbidden, rendered)


class TestCanonicalRegistryGeneration(RegistryGenerationFixture):
    def test_generated_registry_has_exact_schema_order_and_self_loads(self):
        result = self.build()
        self.assertEqual("1.0.0", result.registry_version)
        self.assertEqual(2, result.entry_count)
        self.assertEqual(2, result.active_count)
        raw = self.output_path.read_bytes()
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\r", raw)
        document = json.loads(raw)
        self.assertEqual(
            [
                "schema_version",
                "registry_version",
                "format_id",
                "artifact_schema_version",
                "metadata_schema_version",
                "entries",
            ],
            list(document),
        )
        self.assertEqual(
            ["team_id", "active", "metadata_sha256"],
            list(document["entries"][0]),
        )
        self.assertEqual(
            ["BL-001-v1", "BL-002-v1"],
            [entry["team_id"] for entry in document["entries"]],
        )
        loaded = load_canonical_runtime_registry(
            self.fixture.private_root,
            self.output_path,
        )
        self.assertEqual(("BL-001-v1", "BL-002-v1"), loaded.active_ids)

    def test_membership_comes_only_from_plan_not_artifact_enumeration(self):
        self.fixture.add_artifact("BL-999-v1")
        self.write_plan([{"team_id": "BL-001-v1", "active": True}])
        result = self.build()
        self.assertEqual(1, result.entry_count)
        document = json.loads(self.output_path.read_bytes())
        self.assertEqual(
            ["BL-001-v1"], [item["team_id"] for item in document["entries"]]
        )

    def test_generation_never_reads_packed_or_sidecar_bodies(self):
        real_read = canonical_registry._stable_read_bytes

        def guarded(path, **options):
            if Path(path).name in {"packed.txt", "team.json"}:
                raise AssertionError("full artifact read")
            return real_read(path, **options)

        with mock.patch.object(canonical_registry, "_stable_read_bytes", guarded):
            self.build()

    def test_metadata_digest_uses_exact_bytes_without_reserialization(self):
        metadata_path = self.fixture.schema_root / "BL-001-v1" / "metadata.json"
        metadata_path.write_bytes(metadata_path.read_bytes() + b" \n")
        self.build()
        document = json.loads(self.output_path.read_bytes())
        entry = next(
            item for item in document["entries"] if item["team_id"] == "BL-001-v1"
        )
        self.assertEqual(
            hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            entry["metadata_sha256"],
        )

    def test_active_common_provenance_mismatch_fails_before_publication(self):
        metadata_path = self.fixture.schema_root / "BL-002-v1" / "metadata.json"
        document = json.loads(metadata_path.read_bytes())
        document["showdown_commit"] = "a" * 40
        metadata_path.write_text(json.dumps(document, separators=(",", ":")))
        with self.assertRaises(BlindCanonicalActivationError) as caught:
            self.build()
        self.assert_safe_error(
            caught.exception,
            category=BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
            code="blind_canonical_registry_provenance_mismatch",
        )
        self.assertFalse(self.output_path.exists())

    def test_inactive_historical_metadata_may_use_other_supported_provenance(self):
        metadata_path = self.fixture.schema_root / "BL-002-v1" / "metadata.json"
        document = json.loads(metadata_path.read_bytes())
        document["showdown_commit"] = "a" * 40
        metadata_path.write_text(json.dumps(document, separators=(",", ":")))
        self.write_plan(
            [
                {"team_id": "BL-001-v1", "active": True},
                {"team_id": "BL-002-v1", "active": False},
            ]
        )
        result = self.build()
        self.assertEqual(1, result.active_count)
        registry = load_canonical_runtime_registry(
            self.fixture.private_root,
            self.output_path,
        )
        self.assertEqual(("BL-001-v1",), registry.active_ids)

    def test_equivalent_plan_order_whitespace_and_layout_produce_identical_bytes(self):
        first = self.fixture
        second = SyntheticCanonicalFixture()
        self.addCleanup(second.close)
        second.add_artifact("BL-001-v1")
        second.add_artifact("BL-002-v1")
        second_state_dir = second.private_root / "state"
        second_state_dir.mkdir()
        second_plan = second.base / "plan.json"
        second_plan.write_text(
            json.dumps(
                plan_document(list(reversed(self.entries))),
                ensure_ascii=True,
                indent=4,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        first_result = self.build()
        second_output = second.private_root / "runtime-registry.json"
        second_result = build_canonical_runtime_registry(
            CanonicalRegistryBuildConfig(
                second.private_root,
                second_plan,
                second_output,
                second_state_dir / "bag.json",
            )
        )
        self.assertEqual(first_result, second_result)
        self.assertEqual(self.output_path.read_bytes(), second_output.read_bytes())
        self.assertNotEqual(first.base, second.base)

    def test_randomized_plan_order_has_zero_output_authority(self):
        self.fixture.add_artifact("BL-003-v1")
        entries = self.entries + [{"team_id": "BL-003-v1", "active": False}]
        outputs = []
        for seed in range(5):
            randomized = list(entries)
            random.Random(seed).shuffle(randomized)
            self.write_plan(randomized)
            output = self.fixture.private_root / "runtime-registry-{}.json".format(seed)
            build_canonical_runtime_registry(self.config(output_path=output))
            outputs.append(output.read_bytes())
        self.assertTrue(all(payload == outputs[0] for payload in outputs))

    def test_existing_output_is_refused_without_byte_change_or_temp_residue(self):
        sentinel = PRIVATE_SENTINEL.encode()
        self.output_path.write_bytes(sentinel)
        with self.assertRaises(BlindCanonicalActivationError) as caught:
            self.build()
        self.assert_safe_error(
            caught.exception,
            category=BlindCanonicalErrorCategory.CONFIGURATION,
            code="blind_canonical_registry_output_exists",
        )
        self.assertEqual(sentinel, self.output_path.read_bytes())
        self.assertEqual([], list(self.output_path.parent.glob(".*.tmp")))

    def test_zero_byte_and_identical_existing_outputs_are_never_accepted(self):
        self.build()
        desired = self.output_path.read_bytes()
        self.output_path.unlink()
        for existing in (b"", desired):
            with self.subTest(size=len(existing)):
                self.output_path.write_bytes(existing)
                with self.assertRaises(BlindCanonicalActivationError) as caught:
                    self.build()
                self.assert_safe_error(
                    caught.exception,
                    category=BlindCanonicalErrorCategory.CONFIGURATION,
                )
                self.assertEqual(existing, self.output_path.read_bytes())
                self.assertEqual(
                    [],
                    list(self.output_path.parent.glob(".runtime-registry.json-*.tmp")),
                )
                self.output_path.unlink()

    def test_atomic_publication_failure_matrix_leaves_no_owned_residue(self):
        def boundary_failure(boundary):
            def fail(name):
                if name == boundary:
                    raise OSError(PRIVATE_SENTINEL)

            return mock.patch.object(
                registry_generation,
                "_atomic_boundary",
                side_effect=fail,
            )

        cases = (
            (
                "temp_create",
                lambda: mock.patch.object(
                    registry_generation.tempfile,
                    "mkstemp",
                    side_effect=OSError(PRIVATE_SENTINEL),
                ),
            ),
            ("temp_write", lambda: boundary_failure("temp_write")),
            ("flush", lambda: boundary_failure("flush")),
            (
                "fsync",
                lambda: mock.patch.object(
                    registry_generation.os,
                    "fsync",
                    side_effect=OSError(PRIVATE_SENTINEL),
                ),
            ),
            (
                "parent_fsync",
                lambda: mock.patch.object(
                    registry_generation,
                    "_fsync_directory",
                    side_effect=OSError(PRIVATE_SENTINEL),
                ),
            ),
            (
                "os_link",
                lambda: mock.patch.object(
                    registry_generation.os,
                    "link",
                    side_effect=OSError(PRIVATE_SENTINEL),
                ),
            ),
            (
                "final_reopen",
                lambda: mock.patch.object(
                    registry_generation,
                    "_stable_read_bytes",
                    side_effect=CanonicalArtifactError(
                        "SYNTHETIC",
                        PRIVATE_SENTINEL,
                    ),
                ),
            ),
            (
                "final_byte_verification",
                lambda: mock.patch.object(
                    registry_generation,
                    "_stable_read_bytes",
                    return_value=b"synthetic-mismatch",
                ),
            ),
        )
        for name, patcher in cases:
            with self.subTest(name=name):
                self.output_path.unlink(missing_ok=True)
                with patcher():
                    with self.assertRaises(BlindCanonicalActivationError) as caught:
                        self.build()
                self.assert_safe_error(caught.exception)
                self.assertFalse(self.output_path.exists())
                self.assertEqual(
                    [],
                    list(self.output_path.parent.glob(".runtime-registry.json-*.tmp")),
                )

    def test_racing_destination_is_preserved_and_only_owned_temp_is_removed(self):
        competitor = PRIVATE_SENTINEL.encode()

        def create_competitor(name):
            if name == "publish":
                self.output_path.write_bytes(competitor)

        with mock.patch.object(
            registry_generation,
            "_atomic_boundary",
            side_effect=create_competitor,
        ):
            with self.assertRaises(BlindCanonicalActivationError) as caught:
                self.build()
        self.assert_safe_error(
            caught.exception,
            category=BlindCanonicalErrorCategory.CONFIGURATION,
            code="blind_canonical_registry_output_exists",
        )
        self.assertEqual(competitor, self.output_path.read_bytes())
        self.assertEqual(
            [],
            list(self.output_path.parent.glob(".runtime-registry.json-*.tmp")),
        )

    def test_post_publish_competitor_is_not_deleted_during_safe_rollback(self):
        competitor = PRIVATE_SENTINEL.encode()

        def replace_final(name):
            if name == "readback":
                self.output_path.unlink()
                self.output_path.write_bytes(competitor)
                raise OSError(PRIVATE_SENTINEL)

        with mock.patch.object(
            registry_generation,
            "_atomic_boundary",
            side_effect=replace_final,
        ):
            with self.assertRaises(BlindCanonicalActivationError) as caught:
                self.build()
        self.assert_safe_error(
            caught.exception,
            category=BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
            code="blind_canonical_registry_publication_failed",
        )
        self.assertEqual(competitor, self.output_path.read_bytes())
        self.assertEqual(
            [],
            list(self.output_path.parent.glob(".runtime-registry.json-*.tmp")),
        )

    def test_successful_hardlink_publication_leaves_one_complete_final_name(self):
        self.build()
        final_info = os.lstat(self.output_path)
        self.assertEqual(1, int(final_info.st_nlink))
        self.assertGreater(final_info.st_size, 0)
        self.assertEqual(
            [],
            list(self.output_path.parent.glob(".runtime-registry.json-*.tmp")),
        )
        load_canonical_runtime_registry(
            self.fixture.private_root,
            self.output_path,
        )

    def test_self_load_failure_removes_complete_publication_and_releases_owner(self):
        with mock.patch.object(
            registry_generation,
            "load_canonical_runtime_registry",
            side_effect=CanonicalArtifactError("SYNTHETIC", PRIVATE_SENTINEL),
        ):
            with self.assertRaises(BlindCanonicalActivationError) as caught:
                self.build()
        self.assert_safe_error(
            caught.exception,
            category=BlindCanonicalErrorCategory.DEPLOYMENT_INTEGRITY,
            code="blind_canonical_registry_self_validation_failed",
        )
        self.assertFalse(self.output_path.exists())
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
        owner.close()

    def test_temporary_descriptor_identity_failure_leaves_no_publication(self):
        real_same_file_object = registry_generation._same_file_object
        calls = 0

        def fail_first_identity(left, right):
            nonlocal calls
            calls += 1
            if calls == 1:
                return False
            return real_same_file_object(left, right)

        with mock.patch.object(
            registry_generation,
            "_same_file_object",
            side_effect=fail_first_identity,
        ):
            with self.assertRaises(BlindCanonicalActivationError) as caught:
                self.build()
        self.assert_safe_error(caught.exception)
        self.assertFalse(self.output_path.exists())
        self.assertEqual(
            [],
            list(self.output_path.parent.glob(".runtime-registry.json-*.tmp")),
        )

    def test_replaced_temporary_name_is_not_deleted_as_owned_residue(self):
        competitor = PRIVATE_SENTINEL.encode()
        replacement = None

        def replace_temp(name):
            nonlocal replacement
            if name == "publish":
                candidates = list(
                    self.output_path.parent.glob(".runtime-registry.json-*.tmp")
                )
                self.assertEqual(1, len(candidates))
                replacement = candidates[0]
                replacement.unlink()
                replacement.write_bytes(competitor)
                raise OSError(PRIVATE_SENTINEL)

        with mock.patch.object(
            registry_generation,
            "_atomic_boundary",
            side_effect=replace_temp,
        ):
            with self.assertRaises(BlindCanonicalActivationError) as caught:
                self.build()
        self.assert_safe_error(caught.exception)
        self.assertFalse(self.output_path.exists())
        self.assertIsNotNone(replacement)
        assert replacement is not None
        self.assertEqual(competitor, replacement.read_bytes())
        replacement.unlink()

    def test_output_collisions_are_rejected(self):
        cases = (
            self.fixture.schema_root / "generated.json",
            self.state_path,
            self.state_path.with_name(self.state_path.name + ".lock"),
            self.state_path.with_name(self.state_path.name + ".owner.lock"),
            Path(__file__).resolve().parents[1] / "generated-registry.json",
        )
        for output in cases:
            with self.subTest(name=output.name):
                with self.assertRaises(BlindCanonicalActivationError) as caught:
                    build_canonical_runtime_registry(self.config(output_path=output))
                self.assert_safe_error(caught.exception)

    def test_plan_state_and_plan_output_collisions_are_rejected(self):
        internal_plan = self.state_directory / "plan.json"
        internal_plan.write_bytes(plan_bytes(self.entries))
        cases = (
            self.config(state_path=internal_plan, plan_path=internal_plan),
            self.config(output_path=self.plan_path),
        )
        for config in cases:
            with self.subTest(config=repr(config)):
                with self.assertRaises(BlindCanonicalActivationError) as caught:
                    build_canonical_runtime_registry(config)
                self.assert_safe_error(caught.exception)

    def test_existing_hardlinked_output_is_rejected_unchanged(self):
        sentinel = self.fixture.base / "output-sentinel.bin"
        sentinel.write_bytes(PRIVATE_SENTINEL.encode())
        os.link(sentinel, self.output_path)
        before = self.output_path.read_bytes()
        with self.assertRaises(BlindCanonicalActivationError) as caught:
            self.build()
        self.assert_safe_error(caught.exception)
        self.assertEqual(before, self.output_path.read_bytes())

    def test_hardlink_aliases_to_plan_state_and_control_files_are_preserved(self):
        sources = (
            self.plan_path,
            self.state_path,
            self.state_path.with_name(self.state_path.name + ".lock"),
            self.state_path.with_name(self.state_path.name + ".owner.lock"),
        )
        for index, source in enumerate(sources):
            with self.subTest(index=index):
                if source != self.plan_path:
                    source.write_bytes(PRIVATE_SENTINEL.encode())
                before = source.read_bytes()
                os.link(source, self.output_path)
                with self.assertRaises(BlindCanonicalActivationError) as caught:
                    self.build()
                self.assert_safe_error(
                    caught.exception,
                    category=BlindCanonicalErrorCategory.CONFIGURATION,
                )
                self.assertEqual(before, source.read_bytes())
                self.output_path.unlink()
                if source != self.plan_path:
                    source.unlink()

    @unittest.skipUnless(os.name == "nt", "Windows case-alias proof")
    def test_windows_case_alias_collision_is_rejected(self):
        alias = Path(str(self.state_path).upper())
        with self.assertRaises(BlindCanonicalActivationError) as caught:
            build_canonical_runtime_registry(self.config(output_path=alias))
        self.assert_safe_error(caught.exception)

    def test_output_parent_symlink_is_rejected_when_supported(self):
        target = self.fixture.private_root / "registry-target"
        target.mkdir()
        alias = self.fixture.private_root / "registry-alias"
        try:
            alias.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(
                "directory symlink unavailable: {}".format(type(error).__name__)
            )
        with self.assertRaises(BlindCanonicalActivationError) as caught:
            build_canonical_runtime_registry(
                self.config(output_path=alias / "registry.json")
            )
        self.assert_safe_error(caught.exception)

    def test_existing_state_fails_after_owner_without_reading_or_mutating_it(self):
        sentinel = PRIVATE_SENTINEL.encode()
        self.state_path.write_bytes(sentinel)
        with mock.patch.object(
            Path, "read_text", side_effect=AssertionError("state read")
        ):
            with self.assertRaises(BlindCanonicalActivationError) as caught:
                self.build()
        self.assert_safe_error(
            caught.exception,
            category=BlindCanonicalErrorCategory.RECOVERY_REQUIRED,
            code="blind_canonical_registry_existing_state",
        )
        self.assertEqual(sentinel, self.state_path.read_bytes())
        self.assertFalse(self.output_path.exists())

    def test_runtime_owner_excludes_registry_publication(self):
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
            with self.assertRaises(BlindCanonicalActivationError) as caught:
                self.build(owner_timeout_seconds=0.05)
            self.assert_safe_error(
                caught.exception,
                category=BlindCanonicalErrorCategory.DEPLOYMENT_OWNERSHIP,
                code="blind_canonical_deployment_in_use",
            )
            self.assertFalse(self.output_path.exists())
        finally:
            owner.close()

    def test_runtime_owner_path_still_requires_existing_registry_after_validation(self):
        state_config = BlindPoolStateConfig(
            BlindPoolConfig(
                self.fixture.private_root,
                self.fixture.registry_path,
            ),
            self.state_path,
        )

        def remove_registry(config, **_options):
            self.fixture.registry_path.unlink()
            return config

        with mock.patch.object(
            ownership,
            "validate_blind_pool_state_config",
            side_effect=remove_registry,
        ):
            with self.assertRaises(BlindPoolValidationError):
                acquire_blind_pool_deployment_owner(
                    state_config,
                    timeout_seconds=0.05,
                )
        self.assertFalse(
            self.state_path.with_name(self.state_path.name + ".owner.lock").exists()
        )
