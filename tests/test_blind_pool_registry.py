from __future__ import annotations

import hashlib
import io
import json
import logging
from dataclasses import FrozenInstanceError
from pathlib import Path
import tempfile
import traceback
from unittest import mock
import unittest

from fp.config import BotModes, FoulPlayConfig
from fp.data.blind_pool import (
    PRIVATE_ROOT_ENV,
    REGISTRY_PATH_ENV,
    BlindPoolConfig,
    BlindPoolValidationError,
    get_active_blind_pool_entries,
    get_blind_pool_entry_by_id,
    load_blind_pool_config,
    load_blind_pool_registry,
)
from fp.config import CustomFormatter


ROOT = Path(__file__).resolve().parents[1]
TEAM_BYTES = b"synthetic-team-file-for-registry-tests\n"


def _sha256(value: bytes = TEAM_BYTES) -> str:
    return hashlib.sha256(value).hexdigest()


def _entry(
    team_id: str = "BL-001-v1",
    *,
    active: bool = True,
    team_file: str = "teams/BL-001-v1.team",
    sha256: str | None = None,
) -> dict[str, object]:
    return {
        "team_id": team_id,
        "active": active,
        "team_file": team_file,
        "sha256": sha256 or _sha256(),
    }


def _document(
    *,
    schema_version: object = 1,
    registry_version: object = "1.0",
    format_id: object = "gen9tugs",
    entries: object = None,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "registry_version": registry_version,
        "format_id": format_id,
        "entries": [_entry()] if entries is None else entries,
    }


def _formatted_traceback(error: BaseException) -> str:
    return "".join(traceback.format_exception(type(error), error, error.__traceback__))


class BlindPoolRegistryFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.base = Path(self.temporary_directory.name)
        self.private_root = self.base / "private-artifacts"
        self.team_directory = self.private_root / "teams"
        self.team_directory.mkdir(parents=True)
        self.team_file = self.team_directory / "BL-001-v1.team"
        self.team_file.write_bytes(TEAM_BYTES)
        self.registry_path = self.base / "registry.json"
        self.write_registry(_document())
        self.environ = {
            PRIVATE_ROOT_ENV: str(self.private_root),
            REGISTRY_PATH_ENV: str(self.registry_path),
        }
        self.config = load_blind_pool_config(
            self.environ,
            repository_root=ROOT,
        )
        self.assertIsNotNone(self.config)

    def write_registry(self, document: object) -> None:
        self.registry_path.write_text(
            json.dumps(document, separators=(",", ":")),
            encoding="utf-8",
        )

    def load(self):
        return load_blind_pool_registry(self.config)


class TestBlindPoolConfiguration(BlindPoolRegistryFixture):
    def test_absent_configuration_is_dormant(self):
        with mock.patch(
            "fp.data.blind_pool.config.Path.resolve",
            side_effect=AssertionError("filesystem access is not expected"),
        ):
            self.assertIsNone(load_blind_pool_config({}))

    def test_partial_configuration_is_rejected_without_echoing_values(self):
        private_value = "synthetic-private-root-value"
        with self.assertRaises(BlindPoolValidationError) as captured:
            load_blind_pool_config({PRIVATE_ROOT_ENV: private_value})
        self.assertEqual("config_incomplete", captured.exception.code)
        self.assertNotIn(private_value, str(captured.exception))

    def test_paths_must_be_absolute(self):
        with self.assertRaises(BlindPoolValidationError) as captured:
            BlindPoolConfig(Path("relative-root"), Path("relative-registry"))
        self.assertEqual("config_path_invalid", captured.exception.code)

    def test_configuration_values_must_be_strings(self):
        with self.assertRaises(BlindPoolValidationError) as captured:
            load_blind_pool_config(
                {
                    PRIVATE_ROOT_ENV: object(),
                    REGISTRY_PATH_ENV: str(self.registry_path),
                }
            )
        self.assertEqual("config_value_invalid", captured.exception.code)

    def test_private_root_must_be_outside_repository(self):
        environment = {
            PRIVATE_ROOT_ENV: str(ROOT),
            REGISTRY_PATH_ENV: str(ROOT / "requirements.txt"),
        }
        with self.assertRaises(BlindPoolValidationError) as captured:
            load_blind_pool_config(environment, repository_root=ROOT)
        self.assertEqual("private_root_not_external", captured.exception.code)
        self.assertNotIn(str(ROOT), str(captured.exception))

    def test_registry_must_be_outside_repository(self):
        environment = {
            PRIVATE_ROOT_ENV: str(self.private_root),
            REGISTRY_PATH_ENV: str(ROOT / "requirements.txt"),
        }
        with self.assertRaises(BlindPoolValidationError) as captured:
            load_blind_pool_config(environment, repository_root=ROOT)
        self.assertEqual("registry_not_external", captured.exception.code)
        self.assertNotIn(str(ROOT), str(captured.exception))

    def test_missing_registry_configuration_is_sanitized(self):
        environment = dict(self.environ)
        missing = self.base / "missing-registry.json"
        environment[REGISTRY_PATH_ENV] = str(missing)
        with self.assertRaises(BlindPoolValidationError) as captured:
            load_blind_pool_config(environment, repository_root=ROOT)
        self.assertEqual("registry_file_missing", captured.exception.code)
        self.assertNotIn(str(missing), _formatted_traceback(captured.exception))

    def test_config_repr_does_not_expose_private_paths(self):
        representation = repr(self.config)
        self.assertEqual("BlindPoolConfig(configured=True)", representation)
        self.assertNotIn(str(self.private_root), representation)
        self.assertNotIn(str(self.registry_path), representation)

    def test_legacy_configuration_remains_optional(self):
        config = type(FoulPlayConfig)()
        options = config.configure(
            [
                "--websocket-uri",
                "ws://synthetic.invalid",
                "--ps-username",
                "syntheticbot",
                "--bot-mode",
                BotModes.search_ladder.name,
                "--pokemon-format",
                "gen9tugs",
            ]
        )
        self.assertIsNone(options)
        self.assertEqual("gen9tugs", config.team_name)


class TestBlindPoolRegistryLoading(BlindPoolRegistryFixture):
    def test_valid_registry_load_and_exact_file_resolution(self):
        registry = self.load()
        self.assertEqual(1, registry.schema_version)
        self.assertEqual("1.0", registry.registry_version)
        self.assertEqual("gen9tugs", registry.format_id)
        self.assertEqual(1, len(registry))
        entry = registry.entries[0]
        self.assertEqual("BL-001-v1", entry.team_id)
        self.assertEqual(self.team_file.resolve(), entry.resolved_team_path)
        self.assertEqual("teams/BL-001-v1.team", entry.relative_team_path)

    def test_active_entries_preserve_registry_order(self):
        second_file = self.team_directory / "BL-002-v1.team"
        second_file.write_bytes(b"second-synthetic-team\n")
        self.write_registry(
            _document(
                entries=[
                    _entry(
                        "BL-002-v1",
                        team_file="teams/BL-002-v1.team",
                        sha256=_sha256(b"second-synthetic-team\n"),
                    ),
                    _entry("BL-001-v1"),
                ]
            )
        )
        active = get_active_blind_pool_entries(self.load())
        self.assertEqual(
            ("BL-002-v1", "BL-001-v1"),
            tuple(entry.team_id for entry in active),
        )

    def test_entry_lookup_uses_only_opaque_id(self):
        registry = self.load()
        self.assertIs(
            registry.entries[0],
            get_blind_pool_entry_by_id(registry, "BL-001-v1"),
        )
        self.assertIsNone(get_blind_pool_entry_by_id(registry, "BL-999-v1"))

    def test_models_and_indexes_are_immutable_with_safe_repr(self):
        registry = self.load()
        entry = registry.entries[0]
        with self.assertRaises(FrozenInstanceError):
            entry.active = False
        self.assertNotIn(entry.team_id, repr(entry))
        self.assertNotIn(entry.relative_team_path, repr(entry))
        self.assertNotIn(entry.sha256, repr(entry))
        self.assertNotIn(str(entry.resolved_team_path), repr(registry))

    def test_inactive_entries_remain_valid_but_are_not_active(self):
        self.write_registry(
            _document(entries=[_entry(), _entry("BL-002-v1", active=False)])
        )
        registry = self.load()
        self.assertEqual(2, len(registry))
        self.assertEqual(
            ("BL-001-v1",),
            tuple(entry.team_id for entry in registry.active_entries),
        )
        self.assertFalse(registry.get_entry("BL-002-v1").active)

    def test_safe_formatted_logging_contains_only_aggregate_identity(self):
        output = io.StringIO()
        handler = logging.StreamHandler(output)
        handler.setFormatter(CustomFormatter())
        registry_logger = logging.getLogger("fp.data.blind_pool.loader")
        previous_level = registry_logger.level
        registry_logger.setLevel(logging.INFO)
        registry_logger.addHandler(handler)
        try:
            self.load()
        finally:
            registry_logger.removeHandler(handler)
            registry_logger.setLevel(previous_level)
        logged = output.getvalue()
        self.assertIn("schema=1", logged)
        self.assertIn("entries=1", logged)
        self.assertNotIn(str(self.private_root), logged)
        self.assertNotIn(str(self.registry_path), logged)
        self.assertNotIn("BL-001-v1.team", logged)
        self.assertNotIn(_sha256(), logged)
        self.assertNotIn(TEAM_BYTES.decode().strip(), logged)

    def test_registry_and_team_files_are_not_modified_or_supplemented(self):
        before = {
            "registry": (
                self.registry_path.read_bytes(),
                self.registry_path.stat().st_mtime_ns,
            ),
            "team": (self.team_file.read_bytes(), self.team_file.stat().st_mtime_ns),
            "base_entries": tuple(sorted(path.name for path in self.base.iterdir())),
            "team_entries": tuple(
                sorted(path.name for path in self.team_directory.iterdir())
            ),
        }
        self.load()
        after = {
            "registry": (
                self.registry_path.read_bytes(),
                self.registry_path.stat().st_mtime_ns,
            ),
            "team": (self.team_file.read_bytes(), self.team_file.stat().st_mtime_ns),
            "base_entries": tuple(sorted(path.name for path in self.base.iterdir())),
            "team_entries": tuple(
                sorted(path.name for path in self.team_directory.iterdir())
            ),
        }
        self.assertEqual(before, after)


class TestBlindPoolRegistryValidation(BlindPoolRegistryFixture):
    def assert_error(self, code: str) -> BlindPoolValidationError:
        with self.assertRaises(BlindPoolValidationError) as captured:
            self.load()
        self.assertEqual(code, captured.exception.code)
        return captured.exception

    def test_duplicate_team_id_is_rejected(self):
        self.write_registry(_document(entries=[_entry(), _entry()]))
        error = self.assert_error("duplicate_team_id")
        self.assertIn("BL-001-v1", str(error))

    def test_invalid_schema_version_is_rejected(self):
        self.write_registry(_document(schema_version=2))
        self.assert_error("schema_version_unsupported")

    def test_boolean_schema_version_is_rejected(self):
        self.write_registry(_document(schema_version=True))
        self.assert_error("schema_version_unsupported")

    def test_non_tugs_and_missing_format_are_rejected(self):
        for format_id in ("gen9nationaldex", None):
            with self.subTest(format_id=format_id):
                self.write_registry(_document(format_id=format_id))
                self.assert_error("format_incompatible")

    def test_invalid_registry_version_is_rejected_without_echoing_value(self):
        private_value = "content-derived-version-label"
        self.write_registry(_document(registry_version=private_value))
        error = self.assert_error("registry_version_invalid")
        self.assertNotIn(private_value, str(error))

    def test_duplicate_json_field_is_rejected(self):
        self.registry_path.write_text(
            '{"schema_version":1,"schema_version":1}',
            encoding="utf-8",
        )
        self.assert_error("registry_duplicate_field")

    def test_malformed_json_is_rejected_without_payload_disclosure(self):
        private_value = "content-derived-malformed-value"
        self.registry_path.write_text("{" + private_value, encoding="utf-8")
        error = self.assert_error("registry_json_invalid")
        self.assertNotIn(private_value, _formatted_traceback(error))

    def test_invalid_utf8_registry_is_rejected(self):
        self.registry_path.write_bytes(b"\xff\xfe")
        self.assert_error("registry_encoding_invalid")

    def test_wrong_top_level_type_is_rejected(self):
        self.write_registry([])
        self.assert_error("registry_type_invalid")

    def test_missing_and_unexpected_fields_are_rejected(self):
        document = _document()
        del document["entries"]
        self.write_registry(document)
        self.assert_error("missing_required_field")

        document = _document()
        document["content_label"] = "not-allowed"
        self.write_registry(document)
        error = self.assert_error("unexpected_field")
        self.assertNotIn("content_label", str(error))

    def test_entry_must_be_object_with_exact_fields(self):
        self.write_registry(_document(entries=["not-an-object"]))
        self.assert_error("entry_type_invalid")

        entry = _entry()
        entry["label"] = "not-allowed"
        self.write_registry(_document(entries=[entry]))
        error = self.assert_error("unexpected_field")
        self.assertNotIn("label", str(error))

    def test_team_id_must_be_opaque_and_is_not_echoed_when_invalid(self):
        private_value = "content-derived-team-name"
        self.write_registry(_document(entries=[_entry(private_value)]))
        error = self.assert_error("team_id_invalid")
        self.assertNotIn(private_value, str(error))

    def test_active_flag_must_be_boolean(self):
        self.write_registry(_document(entries=[_entry(active=1)]))
        self.assert_error("active_flag_invalid")

    def test_empty_active_pool_is_rejected(self):
        self.write_registry(_document(entries=[_entry(active=False)]))
        self.assert_error("active_pool_empty")

    def test_inactive_entry_still_requires_existing_file(self):
        self.write_registry(
            _document(
                entries=[
                    _entry(),
                    _entry(
                        "BL-002-v1",
                        active=False,
                        team_file="teams/missing.team",
                    ),
                ]
            )
        )
        self.assert_error("team_file_missing")

    def test_missing_team_file_is_rejected_without_path_disclosure(self):
        private_value = "content-derived-missing.team"
        self.write_registry(
            _document(entries=[_entry(team_file="teams/" + private_value)])
        )
        error = self.assert_error("team_file_missing")
        self.assertNotIn(private_value, _formatted_traceback(error))
        self.assertNotIn(str(self.private_root), _formatted_traceback(error))

    def test_team_path_traversal_is_rejected(self):
        outside_file = self.base / "outside.team"
        outside_file.write_bytes(TEAM_BYTES)
        self.write_registry(_document(entries=[_entry(team_file="../outside.team")]))
        error = self.assert_error("team_path_traversal")
        self.assertNotIn(str(outside_file), str(error))

    def test_absolute_and_windows_style_paths_are_rejected(self):
        for path in ("/outside.team", "C:\\outside.team"):
            with self.subTest(path=path):
                self.write_registry(_document(entries=[_entry(team_file=path)]))
                error = self.assert_error(
                    "team_path_traversal"
                    if path.startswith("/")
                    else "team_path_invalid"
                )
                self.assertNotIn(path, str(error))

    def test_normalized_dot_segments_are_rejected_not_repaired(self):
        for path in ("teams/./BL-001-v1.team", "teams//BL-001-v1.team"):
            with self.subTest(path=path):
                self.write_registry(_document(entries=[_entry(team_file=path)]))
                self.assert_error("team_path_traversal")

    def test_resolved_path_escape_is_rejected(self):
        outside_file = self.base / "outside.team"
        outside_file.write_bytes(TEAM_BYTES)
        candidate = self.private_root / "teams" / "escape.team"
        original_resolve = Path.resolve

        def controlled_resolve(path, strict=False):
            if path == candidate:
                return outside_file
            return original_resolve(path, strict=strict)

        self.write_registry(
            _document(entries=[_entry(team_file="teams/escape.team")])
        )
        with mock.patch.object(Path, "resolve", new=controlled_resolve):
            self.assert_error("team_path_escape")

    def test_referenced_directory_is_rejected(self):
        directory = self.team_directory / "directory.team"
        directory.mkdir()
        self.write_registry(
            _document(entries=[_entry(team_file="teams/directory.team")])
        )
        self.assert_error("team_file_invalid")

    def test_hash_shape_and_hash_mismatch_are_rejected_safely(self):
        self.write_registry(_document(entries=[_entry(sha256="ABC")]))
        self.assert_error("sha256_invalid")

        expected = "0" * 64
        self.write_registry(_document(entries=[_entry(sha256=expected)]))
        error = self.assert_error("sha256_mismatch")
        rendered = _formatted_traceback(error)
        self.assertNotIn(expected, rendered)
        self.assertNotIn(_sha256(), rendered)
        self.assertNotIn(str(self.team_file), rendered)

    def test_active_entries_cannot_alias_the_same_file(self):
        self.write_registry(
            _document(entries=[_entry(), _entry("BL-002-v1")])
        )
        self.assert_error("duplicate_active_team_file")

    def test_unknown_lookup_id_validation_is_sanitized(self):
        private_value = "content-derived-lookup"
        with self.assertRaises(BlindPoolValidationError) as captured:
            get_blind_pool_entry_by_id(self.load(), private_value)
        self.assertEqual("team_id_invalid", captured.exception.code)
        self.assertNotIn(private_value, str(captured.exception))


if __name__ == "__main__":
    unittest.main()
