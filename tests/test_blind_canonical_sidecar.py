import copy
from dataclasses import FrozenInstanceError
import unittest
from unittest import mock

from fp.data.blind_pool.canonical_artifacts import load_canonical_team_artifact
from fp.data.blind_pool.canonical_models import CanonicalArtifactError
from fp.data.blind_pool.canonical_sidecar import parse_canonical_sidecar
from fp import constants
from fp.battle.state import Battler
from fp.config import FoulPlayConfig
from fp.data import pokedex

from tests.test_blind_canonical_registry import (
    STAT_KEYS,
    SyntheticCanonicalFixture,
    compact_json,
    synthetic_sidecar,
)


TEAM_ID = "BL-001-v1"


class CanonicalSidecarTestCase(unittest.TestCase):
    def assert_invalid(self, document=None, *, raw=None):
        if raw is None:
            raw = compact_json(document)
        with self.assertRaises(CanonicalArtifactError) as caught:
            parse_canonical_sidecar(raw, expected_team_id=TEAM_ID)
        self.assertEqual("CANONICAL_SIDECAR_INVALID", caught.exception.code)
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)

    def test_valid_exact_six_set_sidecar_is_minimally_retained(self):
        parsed = parse_canonical_sidecar(
            compact_json(synthetic_sidecar(TEAM_ID)),
            expected_team_id=TEAM_ID,
        )
        self.assertEqual(1, parsed.schema_version)
        self.assertEqual(TEAM_ID, parsed.team_id)
        self.assertEqual("gen9tugs", parsed.format_id)
        self.assertEqual(6, len(parsed.records))
        self.assertEqual("syntheticspecies1", parsed.records[0].species_id)
        self.assertNotIn("Synthetic Species", repr(parsed))

    def test_strict_json_encoding_and_syntax(self):
        self.assert_invalid(raw="{}")
        invalid_raw = (
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
        for raw in invalid_raw:
            with self.subTest(raw=raw[:3]):
                self.assert_invalid(raw=raw)

    def test_duplicate_keys_including_escaped_equivalent_are_rejected(self):
        raw = compact_json(synthetic_sidecar(TEAM_ID)).decode()
        replacements = (
            (
                '"schema_version":1',
                '"schema_version":1,"schema_version":1',
            ),
            (
                '"schema_version":1',
                '"schema_version":1,"schema\\u005fversion":1',
            ),
            (
                '"slot":1,"name"',
                '"slot":1,"slot":1,"name"',
            ),
            (
                '"slot":1,"name":"Synthetic Move 1"',
                '"slot":1,"slot":1,"name":"Synthetic Move 1"',
            ),
            (
                '"hp":0,"atk"',
                '"hp":0,"hp":0,"atk"',
            ),
        )
        for old, new in replacements:
            with self.subTest(old=old):
                self.assert_invalid(raw=raw.replace(old, new, 1).encode())

    def test_top_level_schema_binding_and_exact_fields(self):
        mutations = (
            ("schema_version", 2),
            ("schema_version", True),
            ("team_id", "BL-002-v1"),
            ("format_id", "Gen9TUGS"),
            ("format_id", "gen9tugs "),
        )
        for field, value in mutations:
            document = synthetic_sidecar(TEAM_ID)
            document[field] = value
            with self.subTest(field=field, value=value):
                self.assert_invalid(document)
        document = synthetic_sidecar(TEAM_ID)
        document["unexpected"] = True
        self.assert_invalid(document)
        document = synthetic_sidecar(TEAM_ID)
        del document["sets"]
        self.assert_invalid(document)

    def test_exactly_six_ordered_contiguous_set_slots_are_required(self):
        for count in (5, 7):
            document = synthetic_sidecar(TEAM_ID)
            if count == 5:
                document["sets"].pop()
            else:
                document["sets"].append(copy.deepcopy(document["sets"][-1]))
                document["sets"][-1]["slot"] = 7
            with self.subTest(count=count):
                self.assert_invalid(document)
        for slots in ((1, 1, 3, 4, 5, 6), (1, 2, 4, 4, 5, 6)):
            document = synthetic_sidecar(TEAM_ID)
            for canonical_set, slot in zip(document["sets"], slots):
                canonical_set["slot"] = slot
            with self.subTest(slots=slots):
                self.assert_invalid(document)

    def test_set_exact_fields_and_primitive_types(self):
        document = synthetic_sidecar(TEAM_ID)
        document["sets"][0]["unknown"] = True
        self.assert_invalid(document)
        document = synthetic_sidecar(TEAM_ID)
        del document["sets"][0]["ability"]
        self.assert_invalid(document)
        primitive_mutations = (
            ("slot", True),
            ("name", 1),
            ("species", None),
            ("ability", False),
            ("moves", "move"),
            ("nature", []),
            ("gender", None),
            ("level", "100"),
            ("happiness", 255.0),
            ("shiny", 0),
        )
        for field, value in primitive_mutations:
            document = synthetic_sidecar(TEAM_ID)
            document["sets"][0][field] = value
            with self.subTest(field=field):
                self.assert_invalid(document)

    def test_display_and_canonical_id_contracts_are_structural(self):
        for field in ("species_id", "ability_id", "nature_id"):
            for value in ("", "Upper", "has-hyphen", 1):
                document = synthetic_sidecar(TEAM_ID)
                document["sets"][0][field] = value
                with self.subTest(field=field, value=value):
                    self.assert_invalid(document)
        for field in ("species", "ability", "nature"):
            document = synthetic_sidecar(TEAM_ID)
            document["sets"][0][field] = ""
            with self.subTest(field=field):
                self.assert_invalid(document)
        document = synthetic_sidecar(TEAM_ID)
        document["sets"][0]["name"] = ""
        parse_canonical_sidecar(compact_json(document), expected_team_id=TEAM_ID)

    def test_nullable_display_id_pairs_must_match(self):
        pairs = (
            ("item", "item_id"),
            ("hidden_power_type", "hidden_power_type_id"),
            ("pokeball", "pokeball_id"),
        )
        for display, canonical_id in pairs:
            for left, right in (("Synthetic", None), (None, "synthetic")):
                document = synthetic_sidecar(TEAM_ID)
                document["sets"][0][display] = left
                document["sets"][0][canonical_id] = right
                with self.subTest(pair=display, left=left):
                    self.assert_invalid(document)
            document = synthetic_sidecar(TEAM_ID)
            document["sets"][0][display] = "Synthetic"
            document["sets"][0][canonical_id] = "synthetic"
            parse_canonical_sidecar(compact_json(document), expected_team_id=TEAM_ID)

    def test_one_through_four_moves_are_accepted(self):
        for move_count in range(1, 5):
            document = synthetic_sidecar(TEAM_ID, move_count=move_count)
            with self.subTest(move_count=move_count):
                parsed = parse_canonical_sidecar(
                    compact_json(document),
                    expected_team_id=TEAM_ID,
                )
                self.assertEqual(6, len(parsed.records))

    def test_zero_and_five_moves_are_rejected(self):
        for move_count in (0, 5):
            document = synthetic_sidecar(TEAM_ID, move_count=max(move_count, 1))
            moves = document["sets"][0]["moves"]
            if move_count == 0:
                moves.clear()
            else:
                moves.append({"slot": 5, "name": "Synthetic Five", "id": "fifth"})
            with self.subTest(move_count=move_count):
                self.assert_invalid(document)

    def test_move_slots_fields_text_and_ids_are_exact(self):
        mutations = (
            ("slot", 2),
            ("slot", True),
            ("name", ""),
            ("name", None),
            ("id", "Upper"),
            ("id", ""),
        )
        for field, value in mutations:
            document = synthetic_sidecar(TEAM_ID, move_count=2)
            document["sets"][0]["moves"][0][field] = value
            with self.subTest(field=field, value=value):
                self.assert_invalid(document)
        document = synthetic_sidecar(TEAM_ID)
        document["sets"][0]["moves"][0]["unknown"] = True
        self.assert_invalid(document)
        document = synthetic_sidecar(TEAM_ID)
        del document["sets"][0]["moves"][0]["name"]
        self.assert_invalid(document)

    def test_ev_and_iv_objects_are_exact_and_bounded(self):
        for kind in ("evs", "ivs"):
            document = synthetic_sidecar(TEAM_ID)
            document["sets"][0][kind]["extra"] = 0
            with self.subTest(kind=kind, mutation="extra"):
                self.assert_invalid(document)
            document = synthetic_sidecar(TEAM_ID)
            del document["sets"][0][kind]["hp"]
            with self.subTest(kind=kind, mutation="missing"):
                self.assert_invalid(document)
        cases = (
            ("evs", "hp", -1),
            ("evs", "hp", 253),
            ("evs", "hp", True),
            ("evs", "hp", "0"),
            ("ivs", "hp", -1),
            ("ivs", "hp", 32),
            ("ivs", "hp", 31.0),
        )
        for kind, stat_name, value in cases:
            document = synthetic_sidecar(TEAM_ID)
            document["sets"][0][kind][stat_name] = value
            with self.subTest(kind=kind, value=value):
                self.assert_invalid(document)
        document = synthetic_sidecar(TEAM_ID)
        document["sets"][0]["evs"].update({key: 252 for key in STAT_KEYS})
        self.assert_invalid(document)

    def test_gender_level_happiness_and_shiny_bounds_are_exact(self):
        for gender in ("", "M", "F", "N"):
            document = synthetic_sidecar(TEAM_ID)
            document["sets"][0]["gender"] = gender
            parse_canonical_sidecar(compact_json(document), expected_team_id=TEAM_ID)
        for gender in ("m", "Male", "X"):
            document = synthetic_sidecar(TEAM_ID)
            document["sets"][0]["gender"] = gender
            with self.subTest(gender=gender):
                self.assert_invalid(document)
        for field, values in (("level", (0, 101)), ("happiness", (-1, 256))):
            for value in values:
                document = synthetic_sidecar(TEAM_ID)
                document["sets"][0][field] = value
                with self.subTest(field=field, value=value):
                    self.assert_invalid(document)

    def test_hidden_power_and_pokeball_ids_are_not_derived(self):
        document = synthetic_sidecar(TEAM_ID)
        canonical_set = document["sets"][0]
        canonical_set["hidden_power_type"] = "Display Does Not Match"
        canonical_set["hidden_power_type_id"] = "synthetictype"
        canonical_set["pokeball"] = "Unrelated Display"
        canonical_set["pokeball_id"] = "syntheticball"
        parsed = parse_canonical_sidecar(
            compact_json(document),
            expected_team_id=TEAM_ID,
        )
        self.assertEqual(6, len(parsed.records))

    def test_gigantamax_dynamax_and_tera_profile_state_is_exact(self):
        mutations = (
            ("gigantamax", True),
            ("gigantamax", 0),
            ("dynamax_level", 9),
            ("dynamax_level", True),
            ("tera_type", "Synthetic"),
            ("tera_type_id", "synthetic"),
        )
        for field, value in mutations:
            document = synthetic_sidecar(TEAM_ID)
            document["sets"][0][field] = value
            with self.subTest(field=field, value=value):
                self.assert_invalid(document)

    def test_schema_valid_non_pokemon_ids_are_accepted_without_semantics(self):
        document = synthetic_sidecar(TEAM_ID)
        canonical_set = document["sets"][0]
        canonical_set["species"] = "Definitely Not A Pokemon"
        canonical_set["species_id"] = "notapokemonrecord"
        canonical_set["ability"] = "Not A Real Ability"
        canonical_set["ability_id"] = "notarealability"
        canonical_set["nature"] = "Not A Real Nature"
        canonical_set["nature_id"] = "notarealnature"
        canonical_set["moves"][0] = {
            "slot": 1,
            "name": "Not A Real Move",
            "id": "notarealmove",
        }
        with (
            mock.patch(
                "fp.teams.team_converter.json_to_packed",
                side_effect=AssertionError("converter called"),
            ),
            mock.patch(
                "fp.battle.helpers.normalize_name",
                side_effect=AssertionError("normalizer called"),
            ),
        ):
            parsed = parse_canonical_sidecar(
                compact_json(document),
                expected_team_id=TEAM_ID,
            )
        self.assertEqual("notapokemonrecord", parsed.records[0].species_id)

    def test_parsed_records_are_deeply_immutable(self):
        parsed = parse_canonical_sidecar(
            compact_json(synthetic_sidecar(TEAM_ID)),
            expected_team_id=TEAM_ID,
        )
        with self.assertRaises(FrozenInstanceError):
            parsed.records[0].species_id = "changed"
        with self.assertRaises(TypeError):
            parsed.records[0].evs[0] = 1
        with self.assertRaises(FrozenInstanceError):
            parsed.records = ()

    def test_minimal_battle_projection_has_exact_legacy_shape(self):
        fixture = SyntheticCanonicalFixture()
        try:
            document = synthetic_sidecar(TEAM_ID)
            first = document["sets"][0]
            first["species_id"] = "syntheticformalpha"
            first["nature_id"] = "syntheticnaturealpha"
            first["evs"] = {
                "hp": 1,
                "atk": 2,
                "def": 3,
                "spa": 4,
                "spd": 5,
                "spe": 6,
            }
            first["ivs"] = {
                "hp": 31,
                "atk": 30,
                "def": 29,
                "spa": 28,
                "spd": 27,
                "spe": 26,
            }
            fixture.add_artifact(TEAM_ID, sidecar_document=document)
            registry = fixture.load()
            artifact = load_canonical_team_artifact(registry, TEAM_ID)
            projection = artifact.new_battle_team_projection()
            self.assertEqual(6, len(projection))
            self.assertEqual(
                {"species", "nature", "evs", "ivs"},
                set(projection[0]),
            )
            self.assertEqual("syntheticformalpha", projection[0]["species"])
            self.assertEqual("syntheticnaturealpha", projection[0]["nature"])
            self.assertEqual(
                {key: str(value) for key, value in first["evs"].items()},
                projection[0]["evs"],
            )
            self.assertEqual(
                {key: str(value) for key, value in first["ivs"].items()},
                projection[0]["ivs"],
            )
            forbidden = {
                "packed",
                "metadata",
                "provenance",
                "path",
                "item",
                "ability",
                "moves",
            }
            self.assertTrue(forbidden.isdisjoint(projection[0]))
        finally:
            fixture.close()

    def test_each_projection_is_a_fresh_independent_mutable_copy(self):
        fixture = SyntheticCanonicalFixture()
        try:
            fixture.add_artifact(TEAM_ID)
            artifact = load_canonical_team_artifact(fixture.load(), TEAM_ID)
            first = artifact.new_battle_team_projection()
            second = artifact.new_battle_team_projection()
            first[0]["species"] = "changed"
            first[0]["evs"]["hp"] = "252"
            first[0]["ivs"]["atk"] = "0"
            first.append({})
            third = artifact.new_battle_team_projection()
            self.assertNotEqual(first, second)
            self.assertEqual(second, third)
            self.assertIsNot(first, second)
            self.assertIsNot(first[0]["evs"], second[0]["evs"])
        finally:
            fixture.close()

    def test_minimal_projection_feeds_existing_battler_initialization_seam(self):
        fixture = SyntheticCanonicalFixture()
        original_entries = {
            key: pokedex.get(key)
            for key in ("syntheticspecies{}".format(index) for index in range(1, 7))
        }
        original_format = FoulPlayConfig.pokemon_format
        try:
            FoulPlayConfig.pokemon_format = "gen9tugs"
            fixture.add_artifact(TEAM_ID)
            artifact = load_canonical_team_artifact(fixture.load(), TEAM_ID)
            projection = artifact.new_battle_team_projection()
            request_pokemon = []
            for slot in range(1, 7):
                species_id = "syntheticspecies{}".format(slot)
                pokedex[species_id] = {
                    "name": "SyntheticSpecies{}".format(slot),
                    constants.BASESTATS: {
                        constants.HITPOINTS: 80,
                        constants.ATTACK: 80,
                        constants.DEFENSE: 80,
                        constants.SPECIAL_ATTACK: 80,
                        constants.SPECIAL_DEFENSE: 80,
                        constants.SPEED: 80,
                    },
                    constants.TYPES: ["normal"],
                }
                request_pokemon.append(
                    {
                        constants.IDENT: "p1: Synthetic{}".format(slot),
                        constants.DETAILS: "SyntheticSpecies{}, L100".format(slot),
                        constants.CONDITION: "300/300",
                        constants.ACTIVE: slot == 1,
                        constants.STATS: {
                            "atk": 200,
                            "def": 200,
                            "spa": 200,
                            "spd": 200,
                            "spe": 200,
                        },
                        constants.MOVES: [],
                        "baseAbility": "syntheticability",
                        "ability": "syntheticability",
                        constants.ITEM: "",
                    }
                )
            battler = Battler()
            battler.team_dict = projection
            battler.initialize_first_turn_user_from_json(
                {
                    constants.SIDE: {
                        constants.ID: "p1",
                        constants.POKEMON: request_pokemon,
                    }
                }
            )
            for initialized in [battler.active] + battler.reserve:
                expected = next(
                    item for item in projection if item["species"] == initialized.name
                )
                self.assertEqual(expected["nature"], initialized.nature)
                self.assertEqual(
                    tuple(int(expected["evs"][key]) for key in STAT_KEYS),
                    initialized.evs,
                )
                self.assertEqual(
                    tuple(int(expected["ivs"][key]) for key in STAT_KEYS),
                    initialized.ivs,
                )
        finally:
            FoulPlayConfig.pokemon_format = original_format
            for species_id, value in original_entries.items():
                if value is None:
                    pokedex.pop(species_id, None)
                else:
                    pokedex[species_id] = value
            fixture.close()

    def test_sidecar_does_not_retain_unneeded_private_fields(self):
        parsed = parse_canonical_sidecar(
            compact_json(synthetic_sidecar(TEAM_ID)),
            expected_team_id=TEAM_ID,
        )
        record = parsed.records[0]
        for field in ("item", "ability", "moves", "level", "happiness", "name"):
            self.assertFalse(hasattr(record, field), field)


if __name__ == "__main__":
    unittest.main()
