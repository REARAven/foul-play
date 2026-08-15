import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import fp.data.blind_pool.canonical_registry as canonical_registry
from fp.data.blind_pool.canonical_models import (
    CANONICAL_REGISTRY_FINGERPRINT_PROFILE_VERSION,
    CanonicalArtifactError,
    CanonicalRegistryEntry,
)
from fp.data.blind_pool.canonical_registry import (
    _compute_fingerprint,
    _parse_registry_document,
    compute_canonical_registry_fingerprint,
    load_canonical_runtime_registry,
)
from fp.data.blind_pool.fingerprint import compute_registry_fingerprint
from fp.data.blind_pool.models import BlindPoolEntry, BlindPoolRegistry


STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")
DIGEST_A = "1" * 64
DIGEST_B = "2" * 64


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def synthetic_sidecar(team_id, *, move_count=1):
    sets = []
    for slot in range(1, 7):
        sets.append(
            {
                "slot": slot,
                "name": "Synthetic Alias {}".format(slot),
                "species": "Synthetic Species {}".format(slot),
                "species_id": "syntheticspecies{}".format(slot),
                "item": None,
                "item_id": None,
                "ability": "Synthetic Ability",
                "ability_id": "syntheticability",
                "moves": [
                    {
                        "slot": move_slot,
                        "name": "Synthetic Move {}".format(move_slot),
                        "id": "syntheticmove{}".format(move_slot),
                    }
                    for move_slot in range(1, move_count + 1)
                ],
                "nature": "Synthetic Nature",
                "nature_id": "syntheticnature",
                "evs": {key: 0 for key in STAT_KEYS},
                "ivs": {key: 31 for key in STAT_KEYS},
                "gender": "",
                "level": 100,
                "happiness": 255,
                "shiny": False,
                "hidden_power_type": None,
                "hidden_power_type_id": None,
                "pokeball": None,
                "pokeball_id": None,
                "gigantamax": False,
                "dynamax_level": 10,
                "tera_type": None,
                "tera_type_id": None,
            }
        )
    return {
        "schema_version": 1,
        "team_id": team_id,
        "format_id": "gen9tugs",
        "sets": sets,
    }


def synthetic_metadata(team_id, packed, sidecar, **overrides):
    document = {
        "schema_version": 1,
        "team_id": team_id,
        "format_id": "gen9tugs",
        "format_fingerprint_sha256": "3" * 64,
        "validation_status": "validated",
        "team_size": 6,
        "showdown_commit": "4" * 40,
        "showdown_tree_clean": True,
        "showdown_package_version": "0.11.10",
        "package_lock_sha256": "5" * 64,
        "dist_tree_sha256": "6" * 64,
        "node_version": "v22.1.0",
        "npm_version": "10.8.0",
        "provisioner_version": 1,
        "provisioner_sha256": "7" * 64,
        "source_sha256": "8" * 64,
        "packed_sha256": hashlib.sha256(packed).hexdigest(),
        "sidecar_sha256": hashlib.sha256(sidecar).hexdigest(),
        "semantic_team_fingerprint_sha256": "9" * 64,
    }
    document.update(overrides)
    return document


class SyntheticCanonicalFixture:
    def __init__(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="canonical-synthetic-")
        self.base = Path(self.temporary.name)
        self.private_root = self.base / "private"
        self.schema_root = self.private_root / "canonical" / "schema-1"
        self.schema_root.mkdir(parents=True)
        self.registry_path = self.base / "canonical-registry.json"
        self.entries = []

    def close(self):
        self.temporary.cleanup()

    def add_artifact(
        self,
        team_id,
        *,
        active=True,
        packed=None,
        sidecar_document=None,
        metadata_overrides=None,
        metadata_raw=None,
    ):
        packed = packed if packed is not None else b"synthetic-opaque-wire"
        if sidecar_document is None:
            sidecar_document = synthetic_sidecar(team_id)
        sidecar = compact_json(sidecar_document)
        if metadata_raw is None:
            metadata_document = synthetic_metadata(team_id, packed, sidecar)
            metadata_document.update(metadata_overrides or {})
            metadata_raw = compact_json(metadata_document)
        directory = self.schema_root / team_id
        directory.mkdir()
        (directory / "packed.txt").write_bytes(packed)
        (directory / "team.json").write_bytes(sidecar)
        (directory / "metadata.json").write_bytes(metadata_raw)
        entry = {
            "team_id": team_id,
            "active": active,
            "metadata_sha256": hashlib.sha256(metadata_raw).hexdigest(),
        }
        self.entries.append(entry)
        self.write_registry()
        return directory, entry

    def write_registry(self, *, document=None, raw=None):
        if raw is None:
            if document is None:
                document = {
                    "schema_version": 1,
                    "registry_version": "1.0.0",
                    "format_id": "gen9tugs",
                    "artifact_schema_version": 1,
                    "metadata_schema_version": 1,
                    "entries": self.entries,
                }
            raw = compact_json(document)
        self.registry_path.write_bytes(raw)

    def load(self):
        return load_canonical_runtime_registry(
            self.private_root,
            self.registry_path,
        )


class CanonicalRegistryTestCase(unittest.TestCase):
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

    def valid_document(self):
        return {
            "schema_version": 1,
            "registry_version": "1.0.0",
            "format_id": "gen9tugs",
            "artifact_schema_version": 1,
            "metadata_schema_version": 1,
            "entries": [
                {
                    "team_id": "BL-001-v1",
                    "active": True,
                    "metadata_sha256": DIGEST_A,
                }
            ],
        }

    def test_valid_registry_binds_metadata_and_sorts_entries(self):
        self.fixture.add_artifact("BL-200-v1")
        self.fixture.add_artifact("BL-001-v1")
        registry = self.fixture.load()
        self.assertEqual(("BL-001-v1", "BL-200-v1"), registry.active_ids)
        self.assertEqual(2, len(registry))
        self.assertNotIn("canonical-synthetic", repr(registry))
        self.assertNotIn(self.fixture.entries[0]["metadata_sha256"], repr(registry))
        self.assertNotIn(registry.registry_fingerprint, repr(registry))

    def test_registry_strict_encoding_and_json_failures(self):
        invalid_values = (
            b"\xff",
            b"\xef\xbb\xbf{}",
            b'{"schema_version":1}\x00',
            b"{",
            b"{} trailing",
            b'{"schema_version":NaN}',
            b'{"schema_version":Infinity}',
            b'{"schema_version":-Infinity}',
            b"[]",
        )
        for raw in invalid_values:
            with self.subTest(raw=raw[:3]):
                self.fixture.write_registry(raw=raw)
                self.assert_code("CANONICAL_REGISTRY_INVALID", self.fixture.load)

    def test_registry_duplicate_keys_are_rejected_at_every_depth(self):
        raws = (
            b'{"schema_version":1,"schema_version":1}',
            (
                b'{"schema_version":1,"registry_version":"1","format_id":'
                b'"gen9tugs","artifact_schema_version":1,'
                b'"metadata_schema_version":1,"entries":[{"team_id":'
                b'"BL-001-v1","team_id":"BL-001-v1","active":true,'
                b'"metadata_sha256":"' + b"1" * 64 + b'"}]}'
            ),
            b'{"schema_version":1,"schema\\u005fversion":1}',
        )
        for raw in raws:
            with self.subTest(raw=raw[:20]):
                self.fixture.write_registry(raw=raw)
                self.assert_code(
                    "CANONICAL_REGISTRY_DUPLICATE_KEY",
                    self.fixture.load,
                )

    def test_registry_exact_fields(self):
        for mutation in ("unknown", "missing"):
            document = self.valid_document()
            if mutation == "unknown":
                document["extra"] = True
            else:
                del document["registry_version"]
            with self.subTest(mutation=mutation):
                self.assert_code(
                    "CANONICAL_REGISTRY_INVALID",
                    lambda document=document: _parse_registry_document(document),
                )
        for mutation in ("unknown", "missing"):
            document = self.valid_document()
            if mutation == "unknown":
                document["entries"][0]["extra"] = True
            else:
                del document["entries"][0]["active"]
            with self.subTest(mutation=mutation):
                self.assert_code(
                    "CANONICAL_REGISTRY_INVALID",
                    lambda document=document: _parse_registry_document(document),
                )

    def test_registry_schema_and_versions_are_exact(self):
        cases = (
            ("schema_version", 2, "CANONICAL_REGISTRY_SCHEMA_UNSUPPORTED"),
            ("schema_version", True, "CANONICAL_REGISTRY_SCHEMA_UNSUPPORTED"),
            ("registry_version", "01.0", "CANONICAL_REGISTRY_INVALID"),
            ("registry_version", "1.", "CANONICAL_REGISTRY_INVALID"),
            ("registry_version", " 1", "CANONICAL_REGISTRY_INVALID"),
            ("registry_version", 1, "CANONICAL_REGISTRY_INVALID"),
            ("format_id", "Gen9TUGS", "CANONICAL_REGISTRY_INVALID"),
            (
                "artifact_schema_version",
                2,
                "CANONICAL_REGISTRY_SCHEMA_UNSUPPORTED",
            ),
            (
                "metadata_schema_version",
                2,
                "CANONICAL_REGISTRY_SCHEMA_UNSUPPORTED",
            ),
        )
        for field, value, code in cases:
            document = self.valid_document()
            document[field] = value
            with self.subTest(field=field, value=value):
                self.assert_code(
                    code,
                    lambda document=document: _parse_registry_document(document),
                )

    def test_registry_version_uses_the_documented_grammar_without_normalizing(self):
        valid_versions = ("1", "1.0", "1.00", "12.003.4")
        for version in valid_versions:
            document = self.valid_document()
            document["registry_version"] = version
            with self.subTest(valid=version):
                parsed = _parse_registry_document(document)
                self.assertEqual(version, parsed[1])

        invalid_versions = (
            "",
            "0",
            "0.1",
            "01",
            "01.2",
            ".1",
            "1.",
            "1..0",
            "+1",
            "-1",
            " 1",
            "1 ",
            "1e3",
            "1.0e2",
            1,
            1.0,
        )
        for version in invalid_versions:
            document = self.valid_document()
            document["registry_version"] = version
            with self.subTest(invalid=version):
                self.assert_code(
                    "CANONICAL_REGISTRY_INVALID",
                    lambda document=document: _parse_registry_document(document),
                )

    def test_registry_entry_policy_is_exact(self):
        mutations = (
            ("team_id", "descriptive", "CANONICAL_REGISTRY_INVALID"),
            ("active", 1, "CANONICAL_REGISTRY_INVALID"),
            ("active", "true", "CANONICAL_REGISTRY_INVALID"),
            ("metadata_sha256", "A" * 64, "CANONICAL_REGISTRY_INVALID"),
            ("metadata_sha256", "1" * 63, "CANONICAL_REGISTRY_INVALID"),
        )
        for field, value, code in mutations:
            document = self.valid_document()
            document["entries"][0][field] = value
            with self.subTest(field=field):
                self.assert_code(
                    code,
                    lambda document=document: _parse_registry_document(document),
                )

    def test_registry_requires_unique_nonempty_entries_and_one_active(self):
        empty = self.valid_document()
        empty["entries"] = []
        self.assert_code(
            "CANONICAL_REGISTRY_INVALID",
            lambda: _parse_registry_document(empty),
        )
        inactive = self.valid_document()
        inactive["entries"][0]["active"] = False
        self.assert_code(
            "CANONICAL_REGISTRY_INVALID",
            lambda: _parse_registry_document(inactive),
        )
        duplicate = self.valid_document()
        duplicate["entries"].append(dict(duplicate["entries"][0]))
        self.assert_code(
            "CANONICAL_REGISTRY_INVALID",
            lambda: _parse_registry_document(duplicate),
        )

    def test_fingerprint_is_order_independent_and_path_independent(self):
        first = SyntheticCanonicalFixture()
        second = SyntheticCanonicalFixture()
        try:
            for fixture in (first, second):
                fixture.add_artifact("BL-002-v1")
                fixture.add_artifact("BL-001-v1")
            second.entries.reverse()
            second.write_registry()
            first_registry = first.load()
            second_registry = second.load()
            self.assertEqual(
                first_registry.registry_fingerprint,
                second_registry.registry_fingerprint,
            )
            self.assertEqual(
                first_registry.registry_fingerprint,
                compute_canonical_registry_fingerprint(first_registry),
            )
        finally:
            first.close()
            second.close()

    def test_fingerprint_changes_for_every_semantic_input(self):
        entries = (CanonicalRegistryEntry("BL-001-v1", True, DIGEST_A),)

        def fingerprint(**overrides):
            values = {
                "schema_version": 1,
                "registry_version": "1.0",
                "format_id": "gen9tugs",
                "artifact_schema_version": 1,
                "metadata_schema_version": 1,
                "entries": entries,
            }
            values.update(overrides)
            return _compute_fingerprint(**values)

        baseline = fingerprint()
        mutations = (
            {"schema_version": 2},
            {"registry_version": "1.1"},
            {"format_id": "syntheticformat"},
            {"artifact_schema_version": 2},
            {"metadata_schema_version": 2},
            {"entries": (CanonicalRegistryEntry("BL-001-v1", False, DIGEST_A),)},
            {"entries": (CanonicalRegistryEntry("BL-001-v1", True, DIGEST_B),)},
            {"entries": (CanonicalRegistryEntry("BL-002-v1", True, DIGEST_A),)},
        )
        for mutation in mutations:
            with self.subTest(mutation=tuple(mutation)):
                self.assertNotEqual(baseline, fingerprint(**mutation))
        with mock.patch.object(
            canonical_registry,
            "CANONICAL_REGISTRY_FINGERPRINT_PROFILE_VERSION",
            2,
        ):
            self.assertNotEqual(baseline, fingerprint())
        self.assertEqual(1, CANONICAL_REGISTRY_FINGERPRINT_PROFILE_VERSION)

    def test_metadata_digest_is_checked_before_json_parsing(self):
        directory, entry = self.fixture.add_artifact("BL-001-v1")
        invalid_metadata = b"not-json"
        (directory / "metadata.json").write_bytes(invalid_metadata)
        error = self.assert_code(
            "CANONICAL_METADATA_INTEGRITY_MISMATCH",
            self.fixture.load,
        )
        self.assertNotIn(entry["metadata_sha256"], error)
        self.assertNotIn(hashlib.sha256(invalid_metadata).hexdigest(), error)

        entry["metadata_sha256"] = hashlib.sha256(invalid_metadata).hexdigest()
        self.fixture.write_registry()
        self.assert_code("CANONICAL_METADATA_INVALID", self.fixture.load)

    def test_metadata_exact_schema_and_policy(self):
        mutations = (
            {"schema_version": 2},
            {"team_id": "BL-002-v1"},
            {"format_id": "gen9tugs "},
            {"validation_status": "pending"},
            {"team_size": 5},
            {"showdown_tree_clean": False},
            {"showdown_tree_clean": 1},
            {"showdown_commit": "a" * 39},
            {"showdown_package_version": " bad"},
            {"node_version": "22.1.0"},
            {"npm_version": "01.0.0"},
            {"provisioner_version": 2},
            {"packed_sha256": "A" * 64},
        )
        for mutation in mutations:
            fixture = SyntheticCanonicalFixture()
            try:
                fixture.add_artifact(
                    "BL-001-v1",
                    metadata_overrides=mutation,
                )
                with self.subTest(mutation=tuple(mutation)):
                    self.assert_code("CANONICAL_METADATA_INVALID", fixture.load)
            finally:
                fixture.close()

    def test_metadata_duplicate_unknown_and_missing_fields(self):
        packed = b"synthetic-wire"
        sidecar = compact_json(synthetic_sidecar("BL-001-v1"))
        metadata = synthetic_metadata("BL-001-v1", packed, sidecar)
        invalid_documents = []
        unknown = dict(metadata, unknown=True)
        invalid_documents.append(compact_json(unknown))
        missing = dict(metadata)
        del missing["source_sha256"]
        invalid_documents.append(compact_json(missing))
        text = compact_json(metadata).decode()
        invalid_documents.append(
            text.replace(
                '"schema_version":1',
                '"schema_version":1,"schema_version":1',
            ).encode()
        )
        invalid_documents.append(
            text.replace(
                '"schema_version":1',
                '"schema_version":1,"schema\\u005fversion":1',
            ).encode()
        )
        for raw in invalid_documents:
            fixture = SyntheticCanonicalFixture()
            try:
                fixture.add_artifact(
                    "BL-001-v1",
                    packed=packed,
                    metadata_raw=raw,
                )
                with self.subTest(length=len(raw)):
                    self.assert_code("CANONICAL_METADATA_INVALID", fixture.load)
            finally:
                fixture.close()

    def test_active_entries_require_each_common_provenance_field_to_match(self):
        fields_and_values = (
            ("format_fingerprint_sha256", "a" * 64),
            ("showdown_commit", "b" * 40),
            ("showdown_package_version", "0.12.0"),
            ("showdown_tree_clean", False),
            ("package_lock_sha256", "b" * 64),
            ("dist_tree_sha256", "c" * 64),
            ("node_version", "v23.0.0"),
            ("npm_version", "11.0.0"),
            ("provisioner_version", 2),
            ("provisioner_sha256", "d" * 64),
        )
        for field, value in fields_and_values:
            fixture = SyntheticCanonicalFixture()
            try:
                fixture.add_artifact("BL-001-v1")
                fixture.add_artifact(
                    "BL-002-v1",
                    metadata_overrides={field: value},
                )
                with self.subTest(field=field):
                    expected = (
                        "CANONICAL_METADATA_INVALID"
                        if field in ("showdown_tree_clean", "provisioner_version")
                        else "CANONICAL_ACTIVE_PROVENANCE_MISMATCH"
                    )
                    self.assert_code(expected, fixture.load)
            finally:
                fixture.close()

    def test_team_specific_hashes_may_differ(self):
        self.fixture.add_artifact("BL-001-v1", packed=b"wire-one")
        self.fixture.add_artifact(
            "BL-002-v1",
            packed=b"wire-two",
            metadata_overrides={
                "source_sha256": "a" * 64,
                "semantic_team_fingerprint_sha256": "b" * 64,
            },
        )
        self.assertEqual(2, len(self.fixture.load()))

    def test_inactive_historical_entry_may_have_other_supported_provenance(self):
        self.fixture.add_artifact("BL-001-v1")
        self.fixture.add_artifact(
            "BL-002-v1",
            active=False,
            metadata_overrides={
                "showdown_commit": "a" * 40,
                "showdown_package_version": "0.12.0",
                "node_version": "v23.0.0",
            },
        )
        registry = self.fixture.load()
        self.assertEqual(("BL-001-v1",), registry.active_ids)

    def test_registry_and_models_are_deeply_immutable(self):
        self.fixture.add_artifact("BL-001-v1")
        registry = self.fixture.load()
        with self.assertRaises((AttributeError, TypeError)):
            registry.entries += (registry.entries[0],)
        with self.assertRaises((AttributeError, TypeError)):
            registry.entries[0].active = False
        metadata = registry._metadata_for("BL-001-v1")
        with self.assertRaises((AttributeError, TypeError)):
            metadata.team_size = 1
        self.assertNotIn(DIGEST_A, repr(registry.entries[0]))
        self.assertNotIn("showdown_commit", repr(metadata))
        fingerprint = registry.registry_fingerprint
        active_ids = registry.active_ids
        with self.assertRaises(TypeError):
            registry._entry_by_id["BL-002-v1"] = registry.entries[0]
        with self.assertRaises(TypeError):
            registry._metadata_by_id["BL-002-v1"] = metadata
        active_ids += ("BL-002-v1",)
        self.assertEqual(("BL-001-v1",), registry.active_ids)
        self.assertEqual(fingerprint, registry.registry_fingerprint)

    def test_raw_registry_parent_fingerprint_vector_is_unchanged(self):
        registry = BlindPoolRegistry(
            schema_version=1,
            registry_version="1.0",
            format_id="gen9tugs",
            entries=(
                BlindPoolEntry(
                    "BL-010-v2",
                    False,
                    "teams/BL-010-v2.team",
                    Path("unused-b"),
                    "b" * 64,
                ),
                BlindPoolEntry(
                    "BL-001-v1",
                    True,
                    "teams/BL-001-v1.team",
                    Path("unused-a"),
                    "a" * 64,
                ),
            ),
        )
        self.assertEqual(
            "51c31fa750148962387e54d36d80908ce5e1777af56829613966ebeb254ab25b",
            compute_registry_fingerprint(registry),
        )

    def test_private_root_and_registry_must_be_explicit_external_paths(self):
        self.fixture.add_artifact("BL-001-v1")
        self.assert_code(
            "CANONICAL_ROOT_INVALID",
            lambda: load_canonical_runtime_registry(
                Path("relative"),
                self.fixture.registry_path,
            ),
        )
        self.assert_code(
            "CANONICAL_REGISTRY_INVALID",
            lambda: load_canonical_runtime_registry(
                self.fixture.private_root,
                Path("relative.json"),
            ),
        )
        self.assert_code(
            "CANONICAL_ROOT_INVALID",
            lambda: load_canonical_runtime_registry(
                Path.cwd(),
                self.fixture.registry_path,
            ),
        )


if __name__ == "__main__":
    unittest.main()
