from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from fp.data.blind_pool import canonical_registry
from fp.data.blind_pool.canonical_artifacts import load_canonical_team_artifact
from fp.data.blind_pool.canonical_models import (
    CanonicalArtifactError,
    CanonicalBattleTeamRecord,
    CanonicalTeamArtifact,
)
from fp.data.blind_pool.canonical_registry import _stable_read_bytes
from fp.config import FoulPlayConfig
from fp.teams import load_team
from fp.teams.team_converter import export_to_dict, export_to_packed

from tests.test_blind_canonical_registry import (
    SyntheticCanonicalFixture,
    compact_json,
    synthetic_sidecar,
)


PRIVATE_SENTINEL = "synthetic-private-sentinel-5c1"


def stat_with(info, **overrides):
    values = {
        "st_dev": info.st_dev,
        "st_ino": info.st_ino,
        "st_mode": info.st_mode,
        "st_size": info.st_size,
        "st_mtime": info.st_mtime,
        "st_ctime": info.st_ctime,
        "st_mtime_ns": getattr(info, "st_mtime_ns", int(info.st_mtime * 1_000_000_000)),
        "st_ctime_ns": getattr(info, "st_ctime_ns", int(info.st_ctime * 1_000_000_000)),
        "st_file_attributes": getattr(info, "st_file_attributes", 0),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def rebind_metadata(fixture, team_id, *, metadata_updates=None):
    directory = fixture.schema_root / team_id
    metadata = json.loads((directory / "metadata.json").read_bytes())
    metadata["packed_sha256"] = hashlib.sha256(
        (directory / "packed.txt").read_bytes()
    ).hexdigest()
    metadata["sidecar_sha256"] = hashlib.sha256(
        (directory / "team.json").read_bytes()
    ).hexdigest()
    metadata.update(metadata_updates or {})
    raw = compact_json(metadata)
    (directory / "metadata.json").write_bytes(raw)
    for entry in fixture.entries:
        if entry["team_id"] == team_id:
            entry["metadata_sha256"] = hashlib.sha256(raw).hexdigest()
    fixture.write_registry()


class CanonicalArtifactTestCase(unittest.TestCase):
    def setUp(self):
        self.fixture = SyntheticCanonicalFixture()

    def tearDown(self):
        self.fixture.close()

    def assert_code(self, code, operation):
        with self.assertRaises(CanonicalArtifactError) as caught:
            operation()
        self.assertEqual(code, caught.exception.code)
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        return str(caught.exception)

    def load_valid(self, *, packed=b"synthetic-opaque-wire"):
        self.fixture.add_artifact("BL-001-v1", packed=packed)
        registry = self.fixture.load()
        return registry, load_canonical_team_artifact(registry, "BL-001-v1")

    def test_exact_three_file_artifact_loads_and_preserves_packed_value(self):
        packed = "synthetic-opaque-π-wire".encode()
        _, artifact = self.load_valid(packed=packed)
        self.assertEqual(packed, artifact.packed_for_submission().encode())
        self.assertEqual(6, artifact.set_count)
        self.assertEqual("gen9tugs", artifact.format_id)

    def test_registry_and_selected_load_ignore_unrelated_artifact_directories(self):
        self.fixture.add_artifact("BL-001-v1")
        unrelated = self.fixture.schema_root / ".synthetic-staging"
        unrelated.mkdir()
        (unrelated / "unexpected.tmp").write_bytes(b"synthetic")
        original_scandir = os.scandir

        def guarded_scandir(path):
            if Path(path) in (self.fixture.schema_root, unrelated):
                raise AssertionError("unrelated directory enumerated")
            return original_scandir(path)

        with mock.patch(
            "fp.data.blind_pool.canonical_registry.os.scandir",
            side_effect=guarded_scandir,
        ):
            registry = self.fixture.load()
            artifact = load_canonical_team_artifact(registry, "BL-001-v1")
        self.assertEqual(6, artifact.set_count)

    def test_registry_load_does_not_full_verify_every_artifact(self):
        self.fixture.add_artifact("BL-001-v1")
        second_directory, _ = self.fixture.add_artifact("BL-002-v1")
        (second_directory / "packed.txt").write_bytes(b"changed-after-metadata")
        registry = self.fixture.load()
        artifact = load_canonical_team_artifact(registry, "BL-001-v1")
        self.assertEqual("BL-001-v1", artifact.team_id)
        self.assert_code(
            "CANONICAL_PACKED_INTEGRITY_MISMATCH",
            lambda: load_canonical_team_artifact(registry, "BL-002-v1"),
        )

    def test_each_missing_artifact_file_and_extra_entries_fail(self):
        for missing in ("metadata.json", "packed.txt", "team.json"):
            fixture = SyntheticCanonicalFixture()
            try:
                directory, _ = fixture.add_artifact("BL-001-v1")
                (directory / missing).unlink()
                with self.subTest(missing=missing):
                    self.assert_code(
                        "CANONICAL_ARTIFACT_DIRECTORY_INVALID",
                        fixture.load,
                    )
            finally:
                fixture.close()

        for extra, is_directory in (("backup.txt", False), ("nested", True)):
            fixture = SyntheticCanonicalFixture()
            try:
                directory, _ = fixture.add_artifact("BL-001-v1")
                target = directory / extra
                target.mkdir() if is_directory else target.write_bytes(b"temporary")
                with self.subTest(extra=extra):
                    self.assert_code(
                        "CANONICAL_ARTIFACT_DIRECTORY_INVALID",
                        fixture.load,
                    )
            finally:
                fixture.close()

    def test_symlink_artifact_is_rejected_when_supported(self):
        directory, _ = self.fixture.add_artifact("BL-001-v1")
        packed = directory / "packed.txt"
        target = self.fixture.base / "outside-packed.txt"
        target.write_bytes(packed.read_bytes())
        packed.unlink()
        try:
            packed.symlink_to(target)
        except OSError as error:
            self.skipTest(
                "File symlink creation is unavailable: {}".format(type(error).__name__)
            )
        self.assert_code(
            "CANONICAL_ARTIFACT_DIRECTORY_INVALID",
            self.fixture.load,
        )

    def test_directory_symlink_escape_is_rejected_when_supported(self):
        directory, _ = self.fixture.add_artifact("BL-001-v1")
        outside = self.fixture.base / "outside-artifact"
        directory.rename(outside)
        try:
            directory.symlink_to(outside, target_is_directory=True)
        except OSError as error:
            outside.rename(directory)
            self.skipTest(
                "Directory symlink creation is unavailable: {}".format(
                    type(error).__name__
                )
            )
        self.assert_code("CANONICAL_ARTIFACT_PATH_INVALID", self.fixture.load)

    def test_windows_reparse_attribute_is_rejected_when_capability_exists(self):
        directory, _ = self.fixture.add_artifact("BL-001-v1")
        target = self.fixture.base / "reparse-target.txt"
        target.write_bytes(b"synthetic")
        artifact = directory / "packed.txt"
        artifact.unlink()
        try:
            artifact.symlink_to(target)
        except OSError as error:
            self.skipTest(
                "Reparse-point creation is unavailable: {}".format(type(error).__name__)
            )
        attributes = getattr(os.lstat(artifact), "st_file_attributes", 0)
        if not attributes:
            self.skipTest("Filesystem does not expose reparse attributes")
        self.assert_code(
            "CANONICAL_ARTIFACT_DIRECTORY_INVALID",
            self.fixture.load,
        )

    def test_unregistered_and_traversal_team_ids_cannot_select_paths(self):
        self.fixture.add_artifact("BL-001-v1")
        registry = self.fixture.load()
        for team_id in ("../BL-001-v1", "BL-999-v1", "BL-001-v1/extra"):
            with self.subTest(team_id=team_id):
                self.assert_code(
                    "CANONICAL_ARTIFACT_PATH_INVALID",
                    lambda team_id=team_id: load_canonical_team_artifact(
                        registry,
                        team_id,
                    ),
                )

    def test_stable_read_detects_file_mutation_during_read(self):
        target = self.fixture.base / "stable.bin"
        target.write_bytes(b"stable-content")
        original_open = Path.open

        class MutatingReader:
            def __init__(self, source):
                self.source = source

            def __enter__(self):
                self.source.__enter__()
                return self

            def __exit__(self, *args):
                return self.source.__exit__(*args)

            def fileno(self):
                return self.source.fileno()

            def read(self):
                raw = self.source.read()
                with original_open(target, "ab") as destination:
                    destination.write(b"changed")
                return raw

        def patched_open(path, *args, **kwargs):
            source = original_open(path, *args, **kwargs)
            return MutatingReader(source) if path == target else source

        with mock.patch.object(Path, "open", patched_open):
            self.assert_code(
                "CANONICAL_ARTIFACT_CHANGED",
                lambda: _stable_read_bytes(target, code="CANONICAL_PACKED_INVALID"),
            )

    def test_artifact_directory_snapshot_ignores_volatile_directory_fields(self):
        directory, _ = self.fixture.add_artifact("BL-001-v1")
        original = os.lstat(directory)
        variants = (
            {"st_size": original.st_size + 4096},
            {
                "st_mtime": original.st_mtime + 10,
                "st_ctime": original.st_ctime + 20,
                "st_mtime_ns": original.st_mtime_ns + 10_000_000_000,
                "st_ctime_ns": original.st_ctime_ns + 20_000_000_000,
            },
        )
        for changes in variants:
            with self.subTest(fields=tuple(changes)):
                with mock.patch.object(
                    canonical_registry.os,
                    "lstat",
                    side_effect=(original, stat_with(original, **changes)),
                ):
                    snapshot = canonical_registry._artifact_directory_snapshot(
                        directory,
                        team_id="BL-001-v1",
                    )
                self.assertEqual(3, len(snapshot.entries))

    def test_artifact_directory_snapshot_rejects_identity_transitions(self):
        directory, _ = self.fixture.add_artifact("BL-001-v1")
        original = os.lstat(directory)
        transitions = (
            {"st_ino": original.st_ino + 1},
            {
                "st_file_attributes": getattr(original, "st_file_attributes", 0)
                | canonical_registry._REPARSE_POINT,
            },
        )
        for changes in transitions:
            with self.subTest(fields=tuple(changes)):
                with mock.patch.object(
                    canonical_registry.os,
                    "lstat",
                    side_effect=(original, stat_with(original, **changes)),
                ):
                    self.assert_code(
                        "CANONICAL_ARTIFACT_CHANGED",
                        lambda: canonical_registry._artifact_directory_snapshot(
                            directory,
                            team_id="BL-001-v1",
                        ),
                    )

    def test_child_file_object_replacement_between_snapshots_fails_closed(self):
        directory, _ = self.fixture.add_artifact("BL-001-v1")
        metadata_path = directory / "metadata.json"
        packed_path = directory / "packed.txt"
        real_read = canonical_registry._stable_read_bytes
        replaced = False

        def replacing_read(path, **options):
            nonlocal replaced
            raw = real_read(path, **options)
            if path == metadata_path and not replaced:
                replacement = directory / "packed.replacement"
                replacement.write_bytes(packed_path.read_bytes())
                os.replace(replacement, packed_path)
                replaced = True
            return raw

        with mock.patch.object(
            canonical_registry,
            "_stable_read_bytes",
            side_effect=replacing_read,
        ):
            self.assert_code("CANONICAL_ARTIFACT_CHANGED", self.fixture.load)

    def test_stable_read_detects_regular_file_object_replacement(self):
        target = self.fixture.base / "stable-replacement.bin"
        target.write_bytes(b"stable-content")
        original = os.lstat(target)
        replacement = stat_with(original, st_ino=original.st_ino + 1)
        with mock.patch.object(
            canonical_registry.os,
            "lstat",
            side_effect=(original, replacement),
        ):
            self.assert_code(
                "CANONICAL_ARTIFACT_CHANGED",
                lambda: _stable_read_bytes(target, code="CANONICAL_PACKED_INVALID"),
            )

    def test_metadata_replacement_after_hash_fails_closed(self):
        directory, _ = self.fixture.add_artifact("BL-001-v1")
        registry = self.fixture.load()

        def boundary(name, _team_id):
            if name == "metadata_verified":
                path = directory / "metadata.json"
                path.write_bytes(path.read_bytes() + b" ")

        with mock.patch(
            "fp.data.blind_pool.canonical_artifacts._verification_boundary",
            side_effect=boundary,
        ):
            self.assert_code(
                "CANONICAL_ARTIFACT_CHANGED",
                lambda: load_canonical_team_artifact(registry, "BL-001-v1"),
            )

    def test_packed_replacement_before_final_recheck_fails_closed(self):
        directory, _ = self.fixture.add_artifact("BL-001-v1")
        registry = self.fixture.load()

        def boundary(name, _team_id):
            if name == "packed_verified":
                path = directory / "packed.txt"
                path.write_bytes(path.read_bytes() + b"changed")

        with mock.patch(
            "fp.data.blind_pool.canonical_artifacts._verification_boundary",
            side_effect=boundary,
        ):
            self.assert_code(
                "CANONICAL_ARTIFACT_CHANGED",
                lambda: load_canonical_team_artifact(registry, "BL-001-v1"),
            )

    def test_sidecar_replacement_before_final_recheck_fails_closed(self):
        directory, _ = self.fixture.add_artifact("BL-001-v1")
        registry = self.fixture.load()

        def boundary(name, _team_id):
            if name == "sidecar_verified":
                path = directory / "team.json"
                path.write_bytes(path.read_bytes() + b" ")

        with mock.patch(
            "fp.data.blind_pool.canonical_artifacts._verification_boundary",
            side_effect=boundary,
        ):
            self.assert_code(
                "CANONICAL_ARTIFACT_CHANGED",
                lambda: load_canonical_team_artifact(registry, "BL-001-v1"),
            )

    def test_directory_entry_appearing_during_load_fails_closed(self):
        directory, _ = self.fixture.add_artifact("BL-001-v1")
        registry = self.fixture.load()

        def boundary(name, _team_id):
            if name == "sidecar_verified":
                (directory / "temporary.lock").write_bytes(b"temporary")

        with mock.patch(
            "fp.data.blind_pool.canonical_artifacts._verification_boundary",
            side_effect=boundary,
        ):
            self.assert_code(
                "CANONICAL_ARTIFACT_DIRECTORY_INVALID",
                lambda: load_canonical_team_artifact(registry, "BL-001-v1"),
            )

    def test_packed_integrity_is_verified_before_decoding(self):
        directory, entry = self.fixture.add_artifact("BL-001-v1")
        (directory / "packed.txt").write_bytes(b"\xff")
        registry = self.fixture.load()
        error = self.assert_code(
            "CANONICAL_PACKED_INTEGRITY_MISMATCH",
            lambda: load_canonical_team_artifact(registry, "BL-001-v1"),
        )
        self.assertNotIn(entry["metadata_sha256"], error)

    def test_packed_byte_contract_rejects_invalid_bound_values(self):
        invalid_values = (
            b"",
            b"\xff",
            b"\xef\xbb\xbfopaque",
            b"opaque\x00wire",
            b"opaque\nwire",
            b"opaque\rwire",
            b"opaque\n",
        )
        for packed in invalid_values:
            fixture = SyntheticCanonicalFixture()
            try:
                fixture.add_artifact("BL-001-v1", packed=packed)
                registry = fixture.load()
                with self.subTest(packed_length=len(packed)):
                    self.assert_code(
                        "CANONICAL_PACKED_INVALID",
                        lambda: load_canonical_team_artifact(
                            registry,
                            "BL-001-v1",
                        ),
                    )
            finally:
                fixture.close()

    def test_sidecar_integrity_is_verified_before_parsing(self):
        directory, _ = self.fixture.add_artifact("BL-001-v1")
        (directory / "team.json").write_bytes(b"not-json")
        registry = self.fixture.load()
        self.assert_code(
            "CANONICAL_SIDECAR_INTEGRITY_MISMATCH",
            lambda: load_canonical_team_artifact(registry, "BL-001-v1"),
        )
        rebind_metadata(self.fixture, "BL-001-v1")
        registry = self.fixture.load()
        self.assert_code(
            "CANONICAL_SIDECAR_INVALID",
            lambda: load_canonical_team_artifact(registry, "BL-001-v1"),
        )

    def test_loader_never_repacks_or_uses_pokemon_semantics(self):
        self.fixture.add_artifact("BL-001-v1")
        registry = self.fixture.load()
        with (
            mock.patch(
                "fp.teams.team_converter.export_to_packed",
                side_effect=AssertionError("legacy packer called"),
            ),
            mock.patch(
                "fp.teams.team_converter.json_to_packed",
                side_effect=AssertionError("legacy packer called"),
            ),
            mock.patch(
                "fp.battle.helpers.normalize_name",
                side_effect=AssertionError("normalizer called"),
            ),
            mock.patch(
                "subprocess.run",
                side_effect=AssertionError("subprocess called"),
            ),
        ):
            artifact = load_canonical_team_artifact(registry, "BL-001-v1")
        self.assertEqual(6, artifact.set_count)

    def test_artifact_domain_representation_and_errors_do_not_leak(self):
        packed = PRIVATE_SENTINEL.encode()
        sidecar = synthetic_sidecar("BL-001-v1")
        sidecar["sets"][0]["species"] = PRIVATE_SENTINEL
        self.fixture.add_artifact(
            "BL-001-v1",
            packed=packed,
            sidecar_document=sidecar,
        )
        registry = self.fixture.load()
        output = io.StringIO()
        logs = io.StringIO()
        handler = logging.StreamHandler(logs)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        try:
            with redirect_stdout(output), redirect_stderr(output):
                artifact = load_canonical_team_artifact(registry, "BL-001-v1")
                representations = repr(artifact) + str(artifact) + repr(registry)
                self.assertNotIn(PRIVATE_SENTINEL, representations)
                self.assertEqual(PRIVATE_SENTINEL, artifact.packed_for_submission())

                bad_path = self.fixture.base / PRIVATE_SENTINEL
                error = self.assert_code(
                    "CANONICAL_ROOT_INVALID",
                    lambda: __import__(
                        "fp.data.blind_pool.canonical_registry",
                        fromlist=["load_canonical_runtime_registry"],
                    ).load_canonical_runtime_registry(
                        bad_path,
                        self.fixture.registry_path,
                    ),
                )
                self.assertNotIn(PRIVATE_SENTINEL, error)

                rebind_metadata(
                    self.fixture,
                    "BL-001-v1",
                    metadata_updates={
                        "node_version": PRIVATE_SENTINEL,
                    },
                )
                metadata_error = self.assert_code(
                    "CANONICAL_METADATA_INVALID",
                    self.fixture.load,
                )
                self.assertNotIn(PRIVATE_SENTINEL, metadata_error)
        finally:
            root_logger.removeHandler(handler)
        self.assertNotIn(PRIVATE_SENTINEL, output.getvalue())
        self.assertNotIn(PRIVATE_SENTINEL, logs.getvalue())

    def test_filesystem_error_context_is_not_retained(self):
        target = self.fixture.base / "unreadable.bin"
        target.write_bytes(b"synthetic")
        with mock.patch(
            "fp.data.blind_pool.canonical_registry.os.lstat",
            side_effect=OSError(PRIVATE_SENTINEL),
        ):
            with self.assertRaises(CanonicalArtifactError) as caught:
                _stable_read_bytes(target, code="CANONICAL_PACKED_INVALID")
        self.assertEqual("CANONICAL_PACKED_INVALID", caught.exception.code)
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        self.assertNotIn(PRIVATE_SENTINEL, repr(caught.exception))

    def test_artifact_construction_requires_the_verified_loader_boundary(self):
        records = (
            CanonicalBattleTeamRecord(
                species_id="privatevalue",
                nature_id="privatenature",
                evs=(0, 0, 0, 0, 0, 0),
                ivs=(31, 31, 31, 31, 31, 31),
            ),
        )
        self.assert_code(
            "CANONICAL_INTERNAL_ERROR",
            lambda: CanonicalTeamArtifact(
                team_id="BL-001-v1",
                format_id="gen9tugs",
                artifact_schema_version=1,
                metadata_schema_version=1,
                sidecar_schema_version=1,
                packed_wire=PRIVATE_SENTINEL,
                records=records,
                construction_token=object(),
            ),
        )

    def test_module_imports_have_no_external_side_effects_or_showdown_dependency(self):
        repository = Path(__file__).resolve().parents[1]
        script = """
import importlib, logging, os, socket, subprocess, sys, threading
from pathlib import Path
import fp.data.blind_pool.canonical_models as canonical_models
import fp.data.blind_pool.canonical_registry as canonical_registry
import fp.data.blind_pool.canonical_sidecar as canonical_sidecar
import fp.data.blind_pool.canonical_artifacts as canonical_artifacts
import fp.data.blind_pool.selection as selection
import fp.data.blind_pool.canonical_runtime as canonical_runtime
def blocked(*args, **kwargs):
    raise AssertionError('side effect')
os.scandir = blocked
Path.mkdir = blocked
socket.socket = blocked
subprocess.Popen = blocked
threading.Thread.start = blocked
logging.Logger._log = blocked
importlib.reload(canonical_models)
importlib.reload(canonical_registry)
importlib.reload(canonical_sidecar)
importlib.reload(canonical_artifacts)
importlib.reload(selection)
importlib.reload(canonical_runtime)
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(repository)
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=self.fixture.base,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual("", result.stdout)
        self.assertEqual("", result.stderr)

    def test_missing_or_malformed_canonical_registry_cannot_affect_legacy_config(self):
        argv = [
            "--websocket-uri",
            "wss://synthetic.invalid/showdown/websocket",
            "--ps-username",
            "SyntheticUser",
            "--bot-mode",
            "search_ladder",
            "--pokemon-format",
            "gen9tugs",
            "--team-name",
            "my-human-test-team",
        ]
        with mock.patch(
            "fp.data.blind_pool.canonical_registry.load_canonical_runtime_registry",
            side_effect=AssertionError("dormant loader was called"),
        ):
            options = FoulPlayConfig.configure(argv)
        self.assertIsNone(options)
        self.assertEqual("my-human-test-team", FoulPlayConfig.team_name)

        list_argv = argv[:-2] + ["--team-list", "synthetic-list.txt"]
        with mock.patch(
            "fp.data.blind_pool.canonical_registry.load_canonical_runtime_registry",
            side_effect=AssertionError("dormant loader was called"),
        ):
            options = FoulPlayConfig.configure(list_argv)
        self.assertIsNone(options)
        self.assertEqual("synthetic-list.txt", FoulPlayConfig.team_list)

    def test_legacy_converter_and_manual_team_loader_are_unchanged(self):
        team_export = (
            "Synthetic Mon @ Synthetic Item\n"
            "Ability: Synthetic Ability\n"
            "Synthetic Nature\n"
            "EVs: 4 HP / 252 SpA / 252 Spe\n"
            "IVs: 0 Atk\n"
            "- Synthetic Move"
        )
        expected_packed = export_to_packed(team_export)
        expected_projection = export_to_dict(team_export)
        with (
            mock.patch("fp.teams.load_team.os.path.isdir", return_value=False),
            mock.patch("fp.teams.load_team.os.path.isfile", return_value=True),
            mock.patch("builtins.open", mock.mock_open(read_data=team_export)),
        ):
            packed, projection, filename = load_team("my-human-test-team.txt")
        self.assertEqual(expected_packed, packed)
        self.assertEqual(expected_projection, projection)
        self.assertEqual("my-human-test-team.txt", filename)

    def test_metadata_change_between_registry_and_selected_load_is_rejected(self):
        directory, _ = self.fixture.add_artifact("BL-001-v1")
        registry = self.fixture.load()
        old = json.loads((directory / "metadata.json").read_bytes())
        old["source_sha256"] = "a" * 64
        raw = compact_json(old)
        (directory / "metadata.json").write_bytes(raw)
        self.assert_code(
            "CANONICAL_METADATA_INTEGRITY_MISMATCH",
            lambda: load_canonical_team_artifact(registry, "BL-001-v1"),
        )

    def test_artifact_object_is_immutable_and_not_pickle_serializable(self):
        _, artifact = self.load_valid()
        with self.assertRaises(AttributeError):
            artifact._team_id = "BL-999-v1"
        with self.assertRaises(AttributeError):
            del artifact._packed_wire
        import pickle

        with self.assertRaises(TypeError):
            pickle.dumps(artifact)
        for private_field in (
            "metadata",
            "provenance",
            "path",
            "item",
            "ability",
            "moves",
            "sidecar",
        ):
            self.assertFalse(hasattr(artifact, private_field), private_field)


if __name__ == "__main__":
    unittest.main()
