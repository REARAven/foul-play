import copy
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from fp.battle.helpers import normalize_name
from fp.config import FoulPlayConfig
from fp.data import all_move_json, pokedex
from fp.data.team_pools import (
    PoolIdentity,
    TeamPoolRegistry,
    TeamPoolRegistryError,
    TeamPoolValidationError,
    TeamRecordId,
    canonical_roster_key,
    load_team_pool,
)


SPECIES = ("pikachu", "charizard", "blastoise", "venusaur", "snorlax", "gengar")
MOVES = ("tackle", "protect", "rest", "sleeptalk")
STATS_0 = {"hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}
IVS_31 = {"hp": 31, "atk": 31, "def": 31, "spa": 31, "spd": 31, "spe": 31}
FIXTURE_TEMP_ROOT = Path(__file__).resolve().parent


def _ability_for(species_id):
    return normalize_name(next(iter(pokedex[species_id]["abilities"].values())))


def _pokemon(species_id, slot_number):
    ability_id = _ability_for(species_id)
    return {
        "slot_id": f"slot{slot_number}",
        "species_id": species_id,
        "base_species_id": normalize_name(
            pokedex[species_id].get("baseSpecies", species_id)
        ),
        "item_id": "none",
        "base_ability_id": ability_id,
        "current_ability_id": ability_id,
        "move_ids": list(MOVES),
        "nature_id": "serious",
        "evs": dict(STATS_0),
        "ivs": dict(IVS_31),
        "level": 100,
        "public_fields": ["species_id", "level"],
        "metadata": {"fixture": {"tags": ["synthetic"]}},
    }


def _team(team_id="teamone", variant_id="default", species=SPECIES):
    pokemon = [
        _pokemon(species_id, index + 1) for index, species_id in enumerate(species)
    ]
    return {
        "team_id": team_id,
        "variant_id": variant_id,
        "display_name": "Synthetic team",
        "roster_key": sorted(species),
        "pokemon": pokemon,
        "metadata": {"fixture": True},
    }


def _document(
    teams=None,
    *,
    pool_id="syntheticpool",
    pool_version="1.0.0",
    format_id="gen9tugs",
):
    return {
        "schema_version": 1,
        "pool": {
            "pool_id": pool_id,
            "pool_version": pool_version,
            "format_id": format_id,
            "patch_version": "syntheticpatch1",
            "source_documents": ["syntheticsource1"],
            "display_name": "Synthetic pool",
            "default_public_fields": ["species_id"],
            "metadata": {"nested": {"values": [1, 2]}},
        },
        "teams": [_team()] if teams is None else teams,
    }


def _write_document(directory, document, filename="pool.json"):
    path = Path(directory, filename)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _load_document(document):
    with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
        return load_team_pool(_write_document(directory, document))


def _validation_error(document):
    with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
        path = _write_document(directory, document)
        try:
            load_team_pool(path)
        except TeamPoolValidationError as error:
            return error
    raise AssertionError("expected TeamPoolValidationError")


class TestTeamPoolLoader(unittest.TestCase):
    def test_01_valid_schema_version_one_pool_loads(self):
        pool = _load_document(_document())
        self.assertEqual(1, pool.schema_version)
        self.assertEqual(
            PoolIdentity("syntheticpool", "1.0.0", "gen9tugs"), pool.identity
        )
        self.assertEqual(1, len(pool.teams))
        self.assertEqual("pikachu", pool.teams[0].pokemon[0].species_id)

    def test_02_loaded_pool_and_nested_records_are_immutable(self):
        pool = _load_document(_document())
        with self.assertRaises(FrozenInstanceError):
            pool.patch_version = "changed"
        with self.assertRaises(FrozenInstanceError):
            pool.teams[0].record_id = TeamRecordId("other", "default")
        with self.assertRaises(FrozenInstanceError):
            pool.teams[0].pokemon[0].level = 50
        with self.assertRaises(TypeError):
            pool.metadata["new"] = True
        with self.assertRaises(TypeError):
            pool.teams[0].pokemon[0].metadata["fixture"]["new"] = True
        self.assertIsInstance(pool.metadata["nested"]["values"], tuple)

    def test_03_canonical_roster_key_is_order_independent(self):
        self.assertEqual(
            canonical_roster_key(SPECIES), canonical_roster_key(reversed(SPECIES))
        )

    def test_04_canonical_roster_key_preserves_exact_forms(self):
        ordinary = canonical_roster_key(
            ("raichu", "charizard", "blastoise", "venusaur", "snorlax", "gengar")
        )
        regional = canonical_roster_key(
            ("raichualola", "charizard", "blastoise", "venusaur", "snorlax", "gengar")
        )
        self.assertNotEqual(ordinary, regional)

    def test_05_shared_roster_indexes_multiple_team_ids(self):
        first = _team("teamone", "default")
        second = _team("teamtwo", "alternate")
        second["pokemon"].reverse()
        pool = _load_document(_document([second, first]))
        self.assertEqual(
            (TeamRecordId("teamone", "default"), TeamRecordId("teamtwo", "alternate")),
            pool.matching_team_ids(canonical_roster_key(SPECIES)),
        )

    def test_06_unique_roster_indexes_one_team_id(self):
        pool = _load_document(_document())
        self.assertEqual(
            (TeamRecordId("teamone", "default"),),
            pool.matching_team_ids(canonical_roster_key(SPECIES)),
        )

    def test_07_unknown_roster_returns_no_matches(self):
        pool = _load_document(_document())
        unknown = canonical_roster_key(
            ("raichu", "charizard", "blastoise", "venusaur", "snorlax", "gengar")
        )
        self.assertEqual((), pool.matching_team_ids(unknown))

    def test_08_team_lookup_by_stable_id_is_deterministic(self):
        pool = _load_document(_document([_team("zteam"), _team("ateam")]))
        self.assertEqual(
            (TeamRecordId("ateam", "default"), TeamRecordId("zteam", "default")),
            tuple(pool.team_lookup),
        )
        record_id = TeamRecordId("ateam", "default")
        self.assertIs(pool.team_lookup[record_id], pool.get_team(record_id))

    def test_09_multiple_pools_coexist_in_immutable_registry(self):
        second = _document(pool_id="anotherpool", pool_version="2")
        registry = TeamPoolRegistry(
            (_load_document(_document()), _load_document(second))
        )
        self.assertEqual(2, len(registry))
        with self.assertRaises(TypeError):
            registry.pool_lookup[PoolIdentity("x", "1", "gen9tugs")] = registry.pools[0]

    def test_10_duplicate_pool_identity_is_always_rejected(self):
        pool = _load_document(_document())
        with self.assertRaisesRegex(
            TeamPoolRegistryError, "duplicates are always rejected"
        ):
            TeamPoolRegistry((pool, pool))

    def test_11_unsupported_schema_version_is_rejected(self):
        document = _document()
        document["schema_version"] = 2
        error = _validation_error(document)
        self.assertEqual("$.schema_version", error.issues[-1].path)
        self.assertEqual(
            "unsupported schema version 2; only version 1 is supported",
            error.issues[-1].message,
        )

    def test_12_malformed_json_is_reported_clearly(self):
        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
            path = Path(directory, "pool.json")
            path.write_text('{"schema_version": 1,', encoding="utf-8")
            with self.assertRaisesRegex(
                TeamPoolValidationError, r"malformed JSON at line 1, column 22"
            ):
                load_team_pool(path)

    def test_13_missing_required_fields_are_reported_with_exact_path(self):
        document = _document()
        del document["teams"][0]["pokemon"][0]["item_id"]
        error = _validation_error(document)
        issue = next(issue for issue in error.issues if issue.path.endswith(".item_id"))
        self.assertEqual("$.teams[0].pokemon[0].item_id", issue.path)
        self.assertEqual("missing required field", issue.message)
        self.assertEqual("slot1", issue.slot_id)

    def test_14_wrong_field_types_are_reported(self):
        document = _document()
        document["pool"]["pool_id"] = 7
        document["teams"] = "not-an-array"
        error = _validation_error(document)
        self.assertEqual(
            {"$.pool.pool_id", "$.teams"},
            {issue.path for issue in error.issues},
        )

    def test_15_empty_pool_is_rejected(self):
        error = _validation_error(_document([]))
        self.assertEqual(
            "pool must contain at least one team", error.issues[-1].message
        )

    def test_16_duplicate_team_identity_is_rejected(self):
        error = _validation_error(_document([_team(), _team()]))
        self.assertTrue(
            any(
                "duplicate team-record identity" in issue.message
                for issue in error.issues
            )
        )

    def test_17_team_with_fewer_than_six_pokemon_is_rejected(self):
        team = _team()
        team["pokemon"].pop()
        team.pop("roster_key")
        error = _validation_error(_document([team]))
        self.assertTrue(any("received 5" in issue.message for issue in error.issues))

    def test_18_team_with_more_than_six_pokemon_is_rejected(self):
        team = _team()
        team["pokemon"].append(_pokemon("raichu", 7))
        team.pop("roster_key")
        error = _validation_error(_document([team]))
        self.assertTrue(any("received 7" in issue.message for issue in error.issues))

    def test_19_duplicate_slot_id_is_rejected(self):
        team = _team()
        team["pokemon"][1]["slot_id"] = "slot1"
        error = _validation_error(_document([team]))
        self.assertTrue(
            any("duplicate slot ID 'slot1'" == issue.message for issue in error.issues)
        )

    def test_20_species_clause_violation_across_forms_is_rejected(self):
        species = (
            "raichu",
            "raichualola",
            "blastoise",
            "venusaur",
            "snorlax",
            "gengar",
        )
        error = _validation_error(_document([_team(species=species)]))
        issue = next(
            issue for issue in error.issues if "Species Clause" in issue.message
        )
        self.assertEqual(
            "Species Clause violation for base species 'raichu'", issue.message
        )

    def test_21_unknown_exact_species_form_is_rejected(self):
        document = _document()
        pokemon = document["teams"][0]["pokemon"][0]
        pokemon["species_id"] = "madeupmon"
        pokemon["base_species_id"] = "madeupmon"
        document["teams"][0].pop("roster_key")
        error = _validation_error(document)
        self.assertTrue(
            any(
                "unknown exact species/form ID 'madeupmon'" == issue.message
                for issue in error.issues
            )
        )

    def test_22_noncanonical_species_id_is_rejected_not_rewritten(self):
        document = _document()
        document["teams"][0]["pokemon"][0]["species_id"] = "Porygon-2"
        document["teams"][0].pop("roster_key")
        error = _validation_error(document)
        issue = next(
            issue for issue in error.issues if issue.path.endswith(".species_id")
        )
        self.assertEqual(
            "expected canonical ID 'porygon2', received 'Porygon-2'", issue.message
        )

    def test_23_distinct_represented_dudunsparce_forms_have_distinct_keys(self):
        common = ("charizard", "blastoise", "venusaur", "snorlax", "gengar")
        self.assertNotEqual(
            canonical_roster_key(("dudunsparce", *common)),
            canonical_roster_key(("dudunsparcethreesegment", *common)),
        )

    def test_24_unknown_move_is_rejected(self):
        document = _document()
        document["teams"][0]["pokemon"][0]["move_ids"][0] = "madeupmove"
        error = _validation_error(document)
        self.assertTrue(
            any(
                "unknown move ID 'madeupmove'" == issue.message
                for issue in error.issues
            )
        )

    def test_25_noncanonical_move_id_is_rejected_not_rewritten(self):
        document = _document()
        document["teams"][0]["pokemon"][0]["move_ids"][0] = "Sleep Talk"
        error = _validation_error(document)
        self.assertTrue(
            any(
                issue.message
                == "expected canonical ID 'sleeptalk', received 'Sleep Talk'"
                for issue in error.issues
            )
        )

    def test_26_duplicate_move_ids_are_rejected(self):
        document = _document()
        document["teams"][0]["pokemon"][0]["move_ids"] = [
            "tackle",
            "tackle",
            "rest",
            "sleeptalk",
        ]
        error = _validation_error(document)
        self.assertTrue(
            any("duplicate move ID 'tackle'" == issue.message for issue in error.issues)
        )

    def test_27_record_without_exactly_four_moves_is_rejected(self):
        document = _document()
        document["teams"][0]["pokemon"][0]["move_ids"].pop()
        error = _validation_error(document)
        self.assertTrue(
            any(
                "expected exactly 4 entries, received 3" == issue.message
                for issue in error.issues
            )
        )

    def test_28_invalid_nature_is_rejected(self):
        document = _document()
        document["teams"][0]["pokemon"][0]["nature_id"] = "invented"
        error = _validation_error(document)
        self.assertTrue(
            any(
                "unknown nature ID 'invented'" == issue.message
                for issue in error.issues
            )
        )

    def test_29_ev_field_names_ranges_and_total_are_validated(self):
        cases = []
        wrong_field = _document()
        wrong_field["teams"][0]["pokemon"][0]["evs"]["speed"] = wrong_field["teams"][0][
            "pokemon"
        ][0]["evs"].pop("spe")
        cases.append((wrong_field, "unexpected stat field"))
        bad_range = _document()
        bad_range["teams"][0]["pokemon"][0]["evs"]["hp"] = 253
        cases.append((bad_range, "must be between 0 and 252"))
        bad_total = _document()
        bad_total["teams"][0]["pokemon"][0]["evs"].update(
            {"hp": 252, "atk": 252, "def": 252}
        )
        cases.append((bad_total, "total must not exceed 510"))
        for document, message in cases:
            with self.subTest(message=message):
                self.assertTrue(
                    any(
                        message in issue.message
                        for issue in _validation_error(document).issues
                    )
                )

    def test_30_iv_field_names_and_ranges_are_validated(self):
        wrong_field = _document()
        wrong_field["teams"][0]["pokemon"][0]["ivs"]["speed"] = wrong_field["teams"][0][
            "pokemon"
        ][0]["ivs"].pop("spe")
        self.assertTrue(
            any(
                "unexpected stat field" in issue.message
                for issue in _validation_error(wrong_field).issues
            )
        )
        bad_range = _document()
        bad_range["teams"][0]["pokemon"][0]["ivs"]["hp"] = 32
        self.assertTrue(
            any(
                "must be between 0 and 31" in issue.message
                for issue in _validation_error(bad_range).issues
            )
        )

    def test_31_invalid_level_is_rejected(self):
        for value in (0, 101, True, 1.5):
            with self.subTest(value=value):
                document = _document()
                document["teams"][0]["pokemon"][0]["level"] = value
                self.assertTrue(
                    any(
                        issue.path.endswith(".level")
                        for issue in _validation_error(document).issues
                    )
                )

    def test_32_base_current_ability_mismatch_is_rejected(self):
        document = _document()
        document["teams"][0]["pokemon"][0]["current_ability_id"] = "staticx"
        error = _validation_error(document)
        self.assertTrue(
            any("must equal base_ability_id" in issue.message for issue in error.issues)
        )

    def test_33_invalid_public_field_declaration_is_rejected(self):
        document = _document()
        document["teams"][0]["pokemon"][0]["public_fields"] = ["species_id", "nickname"]
        error = _validation_error(document)
        self.assertTrue(
            any(
                "unrecognized public field 'nickname'" == issue.message
                for issue in error.issues
            )
        )

    def test_34_invalid_variant_of_reference_is_rejected(self):
        team = _team("teamtwo", "alternate")
        team["variant_of"] = {"team_id": "missing", "variant_id": "default"}
        error = _validation_error(_document([team]))
        self.assertTrue(
            any(
                "variant_of references unknown team record" in issue.message
                for issue in error.issues
            )
        )

    def test_35_source_roster_key_mismatch_is_rejected(self):
        team = _team()
        team["roster_key"] = sorted(("raichu", *SPECIES[1:]))
        error = _validation_error(_document([team]))
        self.assertTrue(
            any(
                "source roster key does not match" in issue.message
                for issue in error.issues
            )
        )

    def test_36_validation_reports_multiple_independent_errors(self):
        document = _document()
        first = document["teams"][0]["pokemon"][0]
        second = document["teams"][0]["pokemon"][1]
        first["level"] = 0
        first["move_ids"][0] = "notamove"
        second["nature_id"] = "notanature"
        error = _validation_error(document)
        paths = {issue.path for issue in error.issues}
        self.assertIn("$.teams[0].pokemon[0].level", paths)
        self.assertIn("$.teams[0].pokemon[0].move_ids[0]", paths)
        self.assertIn("$.teams[0].pokemon[1].nature_id", paths)
        self.assertGreaterEqual(len(error.issues), 3)

    def test_37_loading_does_not_modify_source_file(self):
        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
            path = _write_document(directory, _document())
            before = path.read_bytes()
            load_team_pool(path)
            self.assertEqual(before, path.read_bytes())

    def test_38_loading_creates_no_adjacent_file_or_cache(self):
        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
            path = _write_document(directory, _document())
            before = tuple(sorted(item.name for item in Path(directory).iterdir()))
            load_team_pool(path)
            after = tuple(sorted(item.name for item in Path(directory).iterdir()))
            self.assertEqual(("pool.json",), before)
            self.assertEqual(before, after)

    def test_39_loading_does_not_mutate_global_move_data(self):
        before = copy.deepcopy(all_move_json)
        _load_document(_document())
        self.assertEqual(before, all_move_json)

    def test_40_loading_does_not_mutate_global_pokedex_data(self):
        before = copy.deepcopy(pokedex)
        _load_document(_document())
        self.assertEqual(before, pokedex)

    def test_41_loading_does_not_mutate_foul_play_config(self):
        before = dict(vars(FoulPlayConfig))
        _load_document(_document())
        self.assertEqual(before, vars(FoulPlayConfig))

    def test_42_ordinary_non_tugs_data_remains_unaffected(self):
        before_moves = copy.deepcopy(all_move_json)
        before_pokedex = copy.deepcopy(pokedex)
        pool = _load_document(_document(format_id="gen9ou"))
        self.assertEqual("gen9ou", pool.identity.format_id)
        self.assertEqual(before_moves, all_move_json)
        self.assertEqual(before_pokedex, pokedex)

    def test_43_every_lookup_result_is_immutable(self):
        pool = _load_document(_document())
        matches = pool.matching_team_ids(canonical_roster_key(SPECIES))
        self.assertIsInstance(matches, tuple)
        with self.assertRaises(TypeError):
            pool.team_lookup[TeamRecordId("new", "default")] = pool.teams[0]
        with self.assertRaises(TypeError):
            pool.roster_index[canonical_roster_key(SPECIES)] = ()

    def test_44_registry_iteration_order_is_deterministic(self):
        pools = (
            _load_document(_document(pool_id="zpool", pool_version="1")),
            _load_document(_document(pool_id="apool", pool_version="2")),
            _load_document(_document(pool_id="apool", pool_version="1")),
        )
        registry = TeamPoolRegistry(pools)
        self.assertEqual(
            (
                PoolIdentity("apool", "1", "gen9tugs"),
                PoolIdentity("apool", "2", "gen9tugs"),
                PoolIdentity("zpool", "1", "gen9tugs"),
            ),
            tuple(pool.identity for pool in registry),
        )

    def test_missing_file_is_reported(self):
        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
            missing = Path(directory, "missing.json")
            with self.assertRaisesRegex(TeamPoolValidationError, "does not exist"):
                load_team_pool(missing)

    def test_invalid_utf8_is_reported(self):
        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
            path = Path(directory, "pool.json")
            path.write_bytes(b"\xff\xfe")
            with self.assertRaisesRegex(TeamPoolValidationError, "not valid UTF-8"):
                load_team_pool(path)

    def test_wrong_top_level_type_is_rejected(self):
        error = _validation_error([])
        self.assertEqual("$", error.issues[0].path)
        self.assertEqual("expected an object", error.issues[0].message)

    def test_duplicate_json_field_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
            path = Path(directory, "pool.json")
            valid = json.dumps(_document())
            path.write_text(
                valid.replace(
                    '"schema_version": 1', '"schema_version": 1, "schema_version": 1', 1
                ),
                encoding="utf-8",
            )
            with self.assertRaises(TeamPoolValidationError) as caught:
                load_team_pool(path)
            self.assertTrue(
                any(
                    "duplicate JSON field" in issue.message
                    for issue in caught.exception.issues
                )
            )

    def test_explicit_base_species_must_match_authoritative_metadata(self):
        document = _document()
        pokemon = document["teams"][0]["pokemon"][0]
        pokemon["species_id"] = "raichualola"
        pokemon["base_species_id"] = "raichualola"
        pokemon["base_ability_id"] = _ability_for("raichualola")
        pokemon["current_ability_id"] = _ability_for("raichualola")
        document["teams"][0].pop("roster_key")
        error = _validation_error(document)
        self.assertTrue(
            any(
                "authoritative base species 'raichu'" in issue.message
                for issue in error.issues
            )
        )

    def test_species_specific_unknown_ability_is_rejected(self):
        document = _document()
        pokemon = document["teams"][0]["pokemon"][0]
        pokemon["base_ability_id"] = "levitate"
        pokemon["current_ability_id"] = "levitate"
        error = _validation_error(document)
        self.assertTrue(
            any(
                "is not listed for species 'pikachu'" in issue.message
                for issue in error.issues
            )
        )

    def test_no_item_requires_explicit_canonical_string(self):
        for value in (None, "", "No Item"):
            with self.subTest(value=value):
                document = _document()
                document["teams"][0]["pokemon"][0]["item_id"] = value
                self.assertTrue(
                    any(
                        issue.path.endswith(".item_id")
                        for issue in _validation_error(document).issues
                    )
                )

    def test_valid_variant_reference_is_retained(self):
        base = _team("teamone", "default")
        variant = _team("teamone", "alternate")
        variant["variant_of"] = {"team_id": "teamone", "variant_id": "default"}
        pool = _load_document(_document([variant, base]))
        record = pool.get_team(TeamRecordId("teamone", "alternate"))
        self.assertEqual(TeamRecordId("teamone", "default"), record.variant_of)

    def test_source_document_ids_must_already_be_canonical(self):
        document = _document()
        document["pool"]["source_documents"] = ["Document One.docx"]
        error = _validation_error(document)
        self.assertTrue(
            any(issue.path == "$.pool.source_documents[0]" for issue in error.issues)
        )

    def test_canonical_roster_key_rejects_partial_and_noncanonical_inputs(self):
        with self.assertRaisesRegex(ValueError, "exactly six"):
            canonical_roster_key(SPECIES[:5])
        with self.assertRaisesRegex(ValueError, "canonical normalized ID"):
            canonical_roster_key(("Pikachu", *SPECIES[1:]))

    def test_loading_does_not_mutate_battle_mode_or_set_singletons(self):
        from fp.modes import BATTLE_MODES

        mode_ids = {key: id(value) for key, value in BATTLE_MODES.items()}
        mode_states = {key: dict(vars(value)) for key, value in BATTLE_MODES.items()}
        standard_mode = next(
            mode for mode in BATTLE_MODES.values() if hasattr(mode, "team_datasets")
        )
        datasets_id = id(standard_mode.team_datasets)
        smogon_id = id(standard_mode.smogon_sets)
        datasets_state = dict(vars(standard_mode.team_datasets))
        smogon_state = dict(vars(standard_mode.smogon_sets))
        _load_document(_document())
        self.assertEqual(
            mode_ids, {key: id(value) for key, value in BATTLE_MODES.items()}
        )
        self.assertEqual(
            mode_states, {key: dict(vars(value)) for key, value in BATTLE_MODES.items()}
        )
        self.assertEqual(datasets_id, id(standard_mode.team_datasets))
        self.assertEqual(smogon_id, id(standard_mode.smogon_sets))
        self.assertEqual(datasets_state, vars(standard_mode.team_datasets))
        self.assertEqual(smogon_state, vars(standard_mode.smogon_sets))


if __name__ == "__main__":
    unittest.main()
