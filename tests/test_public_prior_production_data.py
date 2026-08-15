import copy
import hashlib
import json
import math
import socket
import sys
import unittest
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
from types import MappingProxyType
from unittest import mock

from fp import constants
from fp.battle.helpers import normalize_name
from fp.battle.public_prior_context import PublicPriorFallback
from fp.battle.state import Battle, LastUsedMove, Pokemon
from fp.battle.team_inference import (
    CandidateFilterState,
    PublicObservationSource,
    TeamPoolMatchState,
)
from fp.config import FoulPlayConfig
from fp.data import all_move_json, pokedex
from fp.data.mods.apply_mods import apply_mods
from fp.data.public_priors import (
    PublicPriorIdentity,
    PublicPriorRegistry,
    PublicSourceKind,
    load_public_prior,
)
from fp.data.public_priors.runtime import (
    PublicPriorConfigurationError,
    PublicPriorStartupOptions,
    load_public_prior_runtime_configuration,
)
from fp.format_spec import FormatSpec
from fp.search.public_prior_sampling import (
    PublicPriorSelectionStatus,
    select_public_prior_variant,
)
from fp.search.standard_battles import prepare_battles


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_PATH = (
    ROOT / "fp" / "data" / "public_priors" / "pools" / "tugspublicarchetypes-1.0.0.json"
)
CANONICAL_SHA256 = "ddb04702321aace7d7368938016a18d54841935d2bf5df194cfaf7af8d4a55ae"
IDENTITY = PublicPriorIdentity("tugspublicarchetypes", "1.0.0", "gen9tugs")
SOURCE_IDS = ("publicformatpatch12", "publicmanualv1")
STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")
AUTHORIZED_SUBMISSION_DENIAL_PATH = "$.metadata.authorization"
AUTHORIZED_SUBMISSION_DENIAL_VALUE = (
    "Generic public design prior authored without closed-submission data."
)

ORIGINAL_MOVES = copy.deepcopy(all_move_json)
ORIGINAL_POKEDEX = copy.deepcopy(pokedex)
ORIGINAL_FORMAT = FoulPlayConfig.pokemon_format
ORIGINAL_SMOGON_STATS = FoulPlayConfig.smogon_stats


def setUpModule():
    FoulPlayConfig.pokemon_format = "gen9tugs"
    FoulPlayConfig.smogon_stats = None
    apply_mods(FormatSpec.from_format_string("gen9tugs"))


def tearDownModule():
    all_move_json.clear()
    all_move_json.update(ORIGINAL_MOVES)
    pokedex.clear()
    pokedex.update(ORIGINAL_POKEDEX)
    FoulPlayConfig.pokemon_format = ORIGINAL_FORMAT
    FoulPlayConfig.smogon_stats = ORIGINAL_SMOGON_STATS


@dataclass(frozen=True)
class ApprovedVariant:
    weight: int
    item: str
    ability: str
    moves: tuple[str, str, str, str]
    nature: str
    evs: tuple[int, int, int, int, int, int]
    ivs: tuple[int, int, int, int, int, int] = (31, 31, 31, 31, 31, 31)
    level: int = 100
    sources: tuple[str, str] = SOURCE_IDS


def _approved(weight, item, ability, moves, nature, evs, ivs=(31,) * 6):
    return ApprovedVariant(weight, item, ability, tuple(moves), nature, evs, ivs)


EXPECTED_VARIANTS = MappingProxyType(
    {
        ("bastiodon", "defensivepivot"): _approved(
            4,
            "leftovers",
            "soundproof",
            ("stealthrock", "partingshot", "bodypress", "toxic"),
            "careful",
            (252, 0, 4, 0, 252, 0),
        ),
        ("bastiodon", "irondefense"): _approved(
            2,
            "chestoberry",
            "soundproof",
            ("irondefense", "bodypress", "rest", "rockblast"),
            "impish",
            (252, 0, 252, 0, 4, 0),
        ),
        ("bastiodon", "sturdymetalburst"): _approved(
            1,
            "custapberry",
            "sturdy",
            ("metalburst", "stealthrock", "partingshot", "bodypress"),
            "sassy",
            (252, 0, 4, 0, 252, 0),
            (31, 31, 31, 31, 31, 0),
        ),
        ("claydol", "levitateutility"): _approved(
            3,
            "leftovers",
            "levitate",
            ("earthpower", "psychic", "rapidspin", "toxic"),
            "calm",
            (252, 0, 4, 0, 252, 0),
            (31, 0, 31, 31, 31, 31),
        ),
        ("claydol", "regeneratorpivot"): _approved(
            4,
            "heavydutyboots",
            "regenerator",
            ("earthpower", "rapidspin", "stealthrock", "teleport"),
            "bold",
            (252, 0, 252, 0, 4, 0),
            (31, 0, 31, 31, 31, 31),
        ),
        ("claydol", "trickroomsetter"): _approved(
            2,
            "mentalherb",
            "levitate",
            ("trickroom", "teleport", "earthpower", "icebeam"),
            "sassy",
            (252, 0, 4, 0, 252, 0),
            (31, 0, 31, 31, 31, 0),
        ),
        ("druddigon", "roughskinutility"): _approved(
            4,
            "rockyhelmet",
            "roughskin",
            ("cragmend", "glare", "dragontail", "stealthrock"),
            "impish",
            (252, 0, 252, 0, 4, 0),
        ),
        ("druddigon", "sheerforcebreaker"): _approved(
            3,
            "lifeorb",
            "sheerforce",
            ("dragonclaw", "gunkshot", "firepunch", "rockslide"),
            "adamant",
            (4, 252, 0, 0, 0, 252),
        ),
        ("druddigon", "slowbandbreaker"): _approved(
            2,
            "choiceband",
            "moldbreaker",
            ("outrage", "earthquake", "gunkshot", "suckerpunch"),
            "brave",
            (252, 252, 0, 0, 4, 0),
            (31, 31, 31, 31, 31, 0),
        ),
        ("dustox", "levitateutility"): _approved(
            4,
            "heavydutyboots",
            "levitate",
            ("roost", "defog", "uturn", "corrosivegas"),
            "calm",
            (252, 0, 4, 0, 252, 0),
            (31, 0, 31, 31, 31, 31),
        ),
        ("dustox", "shielddustphazer"): _approved(
            3,
            "blacksludge",
            "shielddust",
            ("roost", "toxic", "whirlwind", "bugbuzz"),
            "bold",
            (252, 0, 252, 0, 4, 0),
            (31, 0, 31, 31, 31, 31),
        ),
        ("forretress", "hazardpivot"): _approved(
            4,
            "heavydutyboots",
            "overcoat",
            ("spikes", "rapidspin", "voltswitch", "gyroball"),
            "relaxed",
            (252, 0, 252, 0, 4, 0),
            (31, 31, 31, 31, 31, 0),
        ),
        ("forretress", "sturdyrocks"): _approved(
            3,
            "leftovers",
            "sturdy",
            ("stealthrock", "rapidspin", "bodypress", "voltswitch"),
            "impish",
            (252, 0, 252, 0, 4, 0),
        ),
        ("forretress", "toxicspikesboom"): _approved(
            1,
            "custapberry",
            "sturdy",
            ("toxicspikes", "explosion", "rapidspin", "gyroball"),
            "brave",
            (252, 252, 4, 0, 0, 0),
            (31, 31, 31, 31, 31, 0),
        ),
        ("lapras", "ancientshellperishtrap"): _approved(
            2,
            "leftovers",
            "ancientshell",
            ("whirlpool", "perishsong", "protect", "freezedry"),
            "bold",
            (252, 0, 252, 0, 4, 0),
            (31, 0, 31, 31, 31, 31),
        ),
        ("lapras", "ancientshellpivot"): _approved(
            4,
            "heavydutyboots",
            "ancientshell",
            ("freezedry", "surf", "chillyreception", "healbell"),
            "calm",
            (252, 0, 4, 0, 252, 0),
            (31, 0, 31, 31, 31, 31),
        ),
        ("lapras", "waterabsorbtank"): _approved(
            3,
            "leftovers",
            "waterabsorb",
            ("freezedry", "surf", "rest", "sleeptalk"),
            "calm",
            (252, 0, 4, 0, 252, 0),
            (31, 0, 31, 31, 31, 31),
        ),
        ("mawile", "choicebandinterceptor"): _approved(
            4,
            "choiceband",
            "closingjaws",
            ("playrough", "ironhead", "knockoff", "suckerpunch"),
            "adamant",
            (252, 252, 0, 0, 4, 0),
        ),
        ("mawile", "mixedcoverage"): _approved(
            2,
            "expertbelt",
            "closingjaws",
            ("playrough", "knockoff", "flamethrower", "suckerpunch"),
            "brave",
            (4, 252, 0, 252, 0, 0),
            (31, 31, 31, 31, 31, 0),
        ),
        ("mawile", "swordsdancebreaker"): _approved(
            3,
            "lifeorb",
            "closingjaws",
            ("swordsdance", "playrough", "knockoff", "suckerpunch"),
            "adamant",
            (252, 252, 0, 0, 4, 0),
        ),
        ("porygon2", "downloadattacker"): _approved(
            3,
            "eviolite",
            "download",
            ("triattack", "icebeam", "thunderbolt", "recover"),
            "modest",
            (252, 0, 0, 252, 4, 0),
            (31, 0, 31, 31, 31, 31),
        ),
        ("porygon2", "traceutility"): _approved(
            4,
            "eviolite",
            "trace",
            ("foulplay", "discharge", "toxic", "recover"),
            "calm",
            (252, 0, 4, 0, 252, 0),
            (31, 0, 31, 31, 31, 31),
        ),
        ("porygon2", "trickroomsetter"): _approved(
            2,
            "eviolite",
            "download",
            ("trickroom", "recover", "triattack", "icebeam"),
            "quiet",
            (252, 0, 0, 252, 4, 0),
            (31, 0, 31, 31, 31, 0),
        ),
    }
)

EXPECTED_VARIANT_IDS = MappingProxyType(
    {
        "bastiodon": ("defensivepivot", "irondefense", "sturdymetalburst"),
        "claydol": ("levitateutility", "regeneratorpivot", "trickroomsetter"),
        "druddigon": ("roughskinutility", "sheerforcebreaker", "slowbandbreaker"),
        "dustox": ("levitateutility", "shielddustphazer"),
        "forretress": ("hazardpivot", "sturdyrocks", "toxicspikesboom"),
        "lapras": ("ancientshellperishtrap", "ancientshellpivot", "waterabsorbtank"),
        "mawile": ("choicebandinterceptor", "mixedcoverage", "swordsdancebreaker"),
        "porygon2": ("downloadattacker", "traceutility", "trickroomsetter"),
    }
)
EXPECTED_SPECIES = tuple(EXPECTED_VARIANT_IDS)


class DuplicateJsonKeyError(ValueError):
    pass


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise DuplicateJsonKeyError("duplicate JSON key: {!r}".format(key))
        value[key] = item
    return value


def _reject_nonstandard_constant(value):
    raise ValueError("nonstandard JSON constant: {}".format(value))


def _parse_production_bytes(source_bytes=None):
    source_bytes = (
        PRODUCTION_PATH.read_bytes() if source_bytes is None else source_bytes
    )
    text = source_bytes.decode("utf-8", errors="strict")
    return json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_nonstandard_constant,
    )


def _canonical_bytes(document):
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    ).encode("utf-8")


FORBIDDEN_STRING_PATTERNS = (
    "swiss",
    "submission",
    "participant",
    "trainer",
    "teamrecord",
    "teampoolcandidate",
    "closedteamsheet",
    "privateteamsheet",
    "private_team_sheet",
    "referencepool",
    "reference_pool",
)
FORBIDDEN_STRUCTURAL_KEYS = frozenset(
    {
        "roster_key",
        "team_id",
        "trainer_name",
        "participant_id",
        "candidate_id",
        "pool_identity",
        "variant_of",
    }
)


def _assert_public_data_firewall(document):
    def check_string(value, path, *, is_key=False):
        folded = value.casefold()
        for pattern in FORBIDDEN_STRING_PATTERNS:
            if pattern not in folded:
                continue
            if (
                not is_key
                and path == AUTHORIZED_SUBMISSION_DENIAL_PATH
                and value == AUTHORIZED_SUBMISSION_DENIAL_VALUE
                and pattern == "submission"
            ):
                continue
            raise AssertionError(
                "forbidden public-data pattern {!r} at {}: {!r}".format(
                    pattern, path, value
                )
            )

    def walk(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.casefold() in FORBIDDEN_STRUCTURAL_KEYS:
                    raise AssertionError(
                        "forbidden structural key {!r} at {}".format(key, path)
                    )
                check_string(key, "{}.<key:{}>".format(path, key), is_key=True)
                walk(item, "{}.{}".format(path, key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, "{}[{}]".format(path, index))
        elif isinstance(value, str):
            check_string(value, path)

    walk(document, "$")


class _Mode:
    smogon_sets = None

    def __deepcopy__(self, memo):
        return self

    def sample_mega_evolution(self, battle, index, smogon_sets):
        return None

    def check_zoroark_from_move(
        self, battle, side, pokemon, move_name, split_msg, zoroark_from_reserves
    ):
        return pokemon

    def assume_spread_for_speed_check(self, battle, battle_copy):
        return None


def _battle(context, species_id):
    battle = Battle("production-public-prior", public_prior_context=context)
    battle.pokemon_format = "gen9tugs"
    battle.generation = FormatSpec.from_format_string("gen9tugs").generation
    battle.mode = _Mode()
    battle.user.name = "p1"
    battle.opponent.name = "p2"
    battle.user.active = Pokemon("weedle", 100)
    battle.user.last_selected_move = LastUsedMove("weedle", "tackle", 0)
    battle.opponent.active = Pokemon(species_id, 100)
    battle.team_inference.record_public_member(species_id, 100)
    return battle


def _record_complete_public_evidence(battle, species_id, expected):
    for move_id in expected.moves:
        battle.team_inference.record_selected_move(species_id, move_id)
    battle.team_inference.record_initial_item(
        species_id,
        expected.item,
        PublicObservationSource.DIRECT_ITEM_REVEAL,
    )
    battle.team_inference.record_base_ability(species_id, expected.ability)


def _startup(fallback):
    options = PublicPriorStartupOptions((str(PRODUCTION_PATH),), fallback)
    return load_public_prior_runtime_configuration(options, "gen9tugs")


class _FixedRng:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


class TestProductionDocumentIntegrity(unittest.TestCase):
    def test_01_source_is_strict_utf8_with_trailing_newline(self):
        source = PRODUCTION_PATH.read_bytes()
        self.assertTrue(source.endswith(b"\n"))
        self.assertIsInstance(source.decode("utf-8", errors="strict"), str)

    def test_02_parser_rejects_duplicate_keys_and_nonstandard_constants(self):
        self.assertIsInstance(_parse_production_bytes(), dict)
        with self.assertRaises(DuplicateJsonKeyError):
            _parse_production_bytes(b'{"same":1,"same":2}\n')
        with self.assertRaises(ValueError):
            _parse_production_bytes(b'{"value":NaN}\n')

    def test_03_canonical_semantic_hash_is_exact(self):
        digest = hashlib.sha256(_canonical_bytes(_parse_production_bytes())).hexdigest()
        self.assertEqual(CANONICAL_SHA256, digest)

    def test_04_phase_four_loader_accepts_exact_identity_and_inventory(self):
        dataset = load_public_prior(PRODUCTION_PATH)
        self.assertEqual(IDENTITY, dataset.identity)
        self.assertEqual("1.2", dataset.patch_version)
        self.assertEqual("public", dataset.visibility)
        self.assertEqual(2, len(dataset.sources))
        self.assertEqual(8, len(dataset.species))
        self.assertEqual(23, sum(len(record.variants) for record in dataset.species))
        self.assertEqual(
            EXPECTED_SPECIES, tuple(record.species_id for record in dataset.species)
        )

    def test_05_source_declarations_are_exact_and_public_only(self):
        dataset = load_public_prior(PRODUCTION_PATH)
        self.assertEqual(
            SOURCE_IDS, tuple(source.source_id for source in dataset.sources)
        )
        self.assertEqual(
            (PublicSourceKind.PUBLIC_FORMAT, PublicSourceKind.PUBLIC_MANUAL),
            tuple(source.kind for source in dataset.sources),
        )
        self.assertNotIn(
            PublicSourceKind.PUBLIC_REPLAY,
            tuple(source.kind for source in dataset.sources),
        )

    def test_06_exact_species_and_variant_ids(self):
        dataset = load_public_prior(PRODUCTION_PATH)
        self.assertEqual(EXPECTED_SPECIES, tuple(dataset.species_lookup))
        for species_id, expected_ids in EXPECTED_VARIANT_IDS.items():
            with self.subTest(species=species_id):
                record = dataset.get_species(species_id)
                self.assertIsNotNone(record)
                self.assertEqual(expected_ids, tuple(record.variant_lookup))
                self.assertEqual(
                    expected_ids, tuple(v.variant_id for v in record.variants)
                )

    def test_07_every_approved_set_field_is_exact(self):
        dataset = load_public_prior(PRODUCTION_PATH)
        for key, expected in EXPECTED_VARIANTS.items():
            species_id, variant_id = key
            with self.subTest(species=species_id, variant=variant_id):
                variant = dataset.get_species(species_id).get_variant(variant_id)
                actual = ApprovedVariant(
                    weight=variant.weight,
                    item=variant.item_id,
                    ability=variant.base_ability_id,
                    moves=variant.move_ids,
                    nature=variant.nature_id,
                    evs=variant.evs.as_tuple(),
                    ivs=variant.ivs.as_tuple(),
                    level=variant.level,
                    sources=variant.source_ids,
                )
                self.assertEqual(
                    expected, actual, "changed approved set: {}/{}".format(*key)
                )

    def test_08_every_variant_is_structurally_valid_after_tugs_overlay(self):
        dataset = load_public_prior(PRODUCTION_PATH)
        for species in dataset.species:
            known_abilities = {
                normalize_name(ability)
                for ability in pokedex[species.species_id]["abilities"].values()
            }
            for variant in species.variants:
                with self.subTest(
                    species=species.species_id, variant=variant.variant_id
                ):
                    self.assertNotIsInstance(variant.weight, bool)
                    self.assertTrue(
                        math.isfinite(variant.weight) and variant.weight > 0
                    )
                    self.assertEqual(4, len(variant.move_ids))
                    self.assertEqual(4, len(set(variant.move_ids)))
                    self.assertTrue(
                        all(normalize_name(move) == move for move in variant.move_ids)
                    )
                    self.assertTrue(
                        all(move in all_move_json for move in variant.move_ids)
                    )
                    self.assertIn(variant.base_ability_id, known_abilities)
                    self.assertEqual(6, len(variant.evs.as_tuple()))
                    self.assertTrue(
                        all(0 <= value <= 252 for value in variant.evs.as_tuple())
                    )
                    self.assertLessEqual(sum(variant.evs.as_tuple()), 510)
                    self.assertEqual(6, len(variant.ivs.as_tuple()))
                    self.assertTrue(
                        all(0 <= value <= 31 for value in variant.ivs.as_tuple())
                    )
                    self.assertIs(type(variant.level), int)
                    self.assertEqual(100, variant.level)
                    self.assertEqual(SOURCE_IDS, variant.source_ids)
                    self.assertTrue(
                        all(
                            source in dataset.source_lookup
                            for source in variant.source_ids
                        )
                    )

    def test_09_records_lookups_and_metadata_are_immutable(self):
        dataset = load_public_prior(PRODUCTION_PATH)
        variant = dataset.species[0].variants[0]
        with self.assertRaises(FrozenInstanceError):
            dataset.visibility = "private"
        with self.assertRaises(FrozenInstanceError):
            variant.weight = 99
        with self.assertRaises(TypeError):
            dataset.species_lookup["new"] = dataset.species[0]
        with self.assertRaises(TypeError):
            dataset.source_lookup["new"] = dataset.sources[0]
        with self.assertRaises(TypeError):
            variant.metadata["new"] = True

    def test_10_lookup_order_and_normalized_probabilities_are_deterministic(self):
        dataset = load_public_prior(PRODUCTION_PATH)
        registry = PublicPriorRegistry((dataset,))
        self.assertIs(dataset, registry.get(IDENTITY))
        self.assertIsNone(dataset.get_species("Bastiodon"))
        self.assertIsNone(dataset.get_species("aerodactyl"))
        for species_id in EXPECTED_SPECIES:
            with self.subTest(species=species_id):
                first = dataset.normalized_probabilities(species_id)
                second = dataset.normalized_probabilities(species_id)
                self.assertEqual(first, second)
                variants = dataset.get_species(species_id).variants
                denominator = sum(variant.weight for variant in variants)
                self.assertEqual(
                    tuple(variant.variant_id for variant in variants),
                    tuple(entry.reference.variant_id for entry in first),
                )
                self.assertEqual(
                    tuple(variant.weight / denominator for variant in variants),
                    tuple(entry.probability for entry in first),
                )
                self.assertTrue(
                    math.isclose(1.0, sum(entry.probability for entry in first))
                )

    def test_11_validation_does_not_modify_source_bytes(self):
        before = PRODUCTION_PATH.read_bytes()
        load_public_prior(PRODUCTION_PATH)
        _parse_production_bytes(before)
        after = PRODUCTION_PATH.read_bytes()
        self.assertEqual(before, after)


class TestProductionPublicDataFirewall(unittest.TestCase):
    def test_12_approved_document_passes_recursive_firewall(self):
        document = _parse_production_bytes()
        self.assertEqual(
            AUTHORIZED_SUBMISSION_DENIAL_VALUE,
            document["metadata"]["authorization"],
        )
        _assert_public_data_firewall(document)

    def test_13_submission_exemption_is_path_and_value_specific(self):
        document = _parse_production_bytes()
        copied = copy.deepcopy(document)
        copied["metadata"]["copied_authorization"] = AUTHORIZED_SUBMISSION_DENIAL_VALUE
        with self.assertRaisesRegex(AssertionError, "submission"):
            _assert_public_data_firewall(copied)

        altered = copy.deepcopy(document)
        altered["metadata"]["authorization"] += " Altered."
        self.assertNotEqual(
            CANONICAL_SHA256,
            hashlib.sha256(_canonical_bytes(altered)).hexdigest(),
        )
        with self.assertRaisesRegex(AssertionError, "submission"):
            _assert_public_data_firewall(altered)

    def test_14_every_other_private_pattern_is_rejected_in_keys_and_values(self):
        document = _parse_production_bytes()
        for pattern in FORBIDDEN_STRING_PATTERNS:
            with self.subTest(pattern=pattern, location="value"):
                changed = copy.deepcopy(document)
                changed["metadata"]["probe"] = pattern
                with self.assertRaises(AssertionError):
                    _assert_public_data_firewall(changed)
            with self.subTest(pattern=pattern, location="key"):
                changed = copy.deepcopy(document)
                changed["metadata"][pattern] = "probe"
                with self.assertRaises(AssertionError):
                    _assert_public_data_firewall(changed)

    def test_15_prohibited_structural_keys_are_rejected_recursively(self):
        document = _parse_production_bytes()
        for key in sorted(FORBIDDEN_STRUCTURAL_KEYS):
            with self.subTest(key=key):
                changed = copy.deepcopy(document)
                changed["species"][0]["variants"][0]["metadata"][key] = "opaque"
                with self.assertRaisesRegex(AssertionError, "structural key"):
                    _assert_public_data_firewall(changed)


class TestProductionStartupConfiguration(unittest.TestCase):
    def test_16_explicit_generic_startup_loads_once_and_creates_fresh_contexts(self):
        with mock.patch(
            "fp.data.public_priors.runtime.load_public_prior",
            wraps=load_public_prior,
        ) as loader:
            configuration = _startup(PublicPriorFallback.GENERIC)
        loader.assert_called_once_with(PRODUCTION_PATH)
        self.assertEqual((IDENTITY,), configuration.selected_identities)
        self.assertEqual("gen9tugs", configuration.format_id)
        self.assertIs(PublicPriorFallback.GENERIC, configuration.fallback_policy)
        self.assertIs(
            configuration.registry.get(IDENTITY), configuration.registry.datasets[0]
        )
        first = configuration.create_battle_context("gen9tugs")
        second = configuration.create_battle_context("gen9tugs")
        self.assertIsNot(first, second)
        self.assertIs(first.registry, second.registry)
        self.assertEqual((IDENTITY,), first.selected_identities)

    def test_17_explicit_none_startup_loads_once_and_creates_fresh_contexts(self):
        with mock.patch(
            "fp.data.public_priors.runtime.load_public_prior",
            wraps=load_public_prior,
        ) as loader:
            configuration = _startup(PublicPriorFallback.NONE)
        loader.assert_called_once_with(PRODUCTION_PATH)
        self.assertIs(PublicPriorFallback.NONE, configuration.fallback_policy)
        self.assertIsNot(
            configuration.create_battle_context("gen9tugs"),
            configuration.create_battle_context("gen9tugs"),
        )

    def test_18_format_isolation_rejects_non_tugs_formats(self):
        for format_id in ("gen9", "gen9ou", "gen9nationaldex"):
            with self.subTest(format_id=format_id):
                options = PublicPriorStartupOptions(
                    (str(PRODUCTION_PATH),), PublicPriorFallback.GENERIC
                )
                with self.assertRaises(PublicPriorConfigurationError):
                    load_public_prior_runtime_configuration(options, format_id)

    def test_19_no_configuration_preserves_ordinary_formats_and_no_default(self):
        with mock.patch("fp.data.public_priors.runtime.load_public_prior") as loader:
            for format_id in ("gen9", "gen9ou", "gen9nationaldex", "gen9tugs"):
                self.assertIsNone(
                    load_public_prior_runtime_configuration(None, format_id)
                )
        loader.assert_not_called()
        production_sources = (
            ROOT / "fp" / "config.py",
            ROOT / "fp" / "main.py",
            ROOT / "fp" / "run_battle.py",
            ROOT / "fp" / "modes" / "standard_battle.py",
        )
        for path in production_sources:
            self.assertNotIn(PRODUCTION_PATH.name, path.read_text(encoding="utf-8"))

    def test_20_startup_reads_only_explicit_public_file_and_no_private_models(self):
        runtime_source = (
            ROOT / "fp" / "data" / "public_priors" / "runtime.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("fp.data.team_pools", runtime_source)
        self.assertNotIn("TeamRecord", runtime_source)
        self.assertNotIn("TeamPoolCandidate", runtime_source)

        read_paths = []
        original_read_text = Path.read_text

        def tracked_read_text(path, *args, **kwargs):
            read_paths.append(path.resolve())
            return original_read_text(path, *args, **kwargs)

        modules_before = set(sys.modules)
        with mock.patch.object(Path, "read_text", tracked_read_text):
            _startup(PublicPriorFallback.GENERIC)
        self.assertEqual([PRODUCTION_PATH.resolve()], read_paths)
        newly_imported_private = {
            name
            for name in set(sys.modules) - modules_before
            if name.startswith("fp.data.team_pools")
        }
        self.assertEqual(set(), newly_imported_private)


class TestProductionSamplingIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generic_configuration = _startup(PublicPriorFallback.GENERIC)
        cls.none_configuration = _startup(PublicPriorFallback.NONE)
        cls.dataset = cls.none_configuration.registry.get(IDENTITY)

    def test_21_all_23_variants_are_selected_and_populated_coherently(self):
        context = self.none_configuration.create_battle_context("gen9tugs")
        seen_species = set()
        seen_variants = set()
        for key, expected in EXPECTED_VARIANTS.items():
            species_id, variant_id = key
            with self.subTest(species=species_id, variant=variant_id):
                battle = _battle(context, species_id)
                _record_complete_public_evidence(battle, species_id, expected)

                # Opaque closed-pool counts and match state must not affect public sampling.
                battle.team_inference._baseline_candidate_ids = ("opaque-a", "opaque-b")
                battle.team_inference._active_candidate_ids = ("opaque-b",)
                battle.team_inference._match_state = TeamPoolMatchState.MATCHED
                battle.team_inference._filter_state = CandidateFilterState.REDUCED

                canonical_before = (
                    battle.opponent.active.item,
                    battle.opponent.active.ability,
                    battle.opponent.active.original_ability,
                    tuple(battle.opponent.active.moves),
                    battle.opponent.active.nature,
                    tuple(battle.opponent.active.evs),
                    tuple(battle.opponent.active.ivs),
                )
                with mock.patch("fp.search.standard_battles.sample_pokemon") as generic:
                    sampled = prepare_battles(battle, 1)[0][0]
                generic.assert_not_called()
                pokemon = sampled.opponent.active
                self.assertEqual(expected.item, pokemon.item)
                self.assertEqual(
                    (expected.ability, expected.ability),
                    (pokemon.ability, pokemon.original_ability),
                )
                self.assertEqual(
                    expected.moves, tuple(move.name for move in pokemon.moves)
                )
                self.assertEqual(expected.nature, pokemon.nature)
                self.assertEqual(expected.evs, tuple(pokemon.evs))
                self.assertEqual(expected.ivs, tuple(pokemon.ivs))
                self.assertEqual(100, pokemon.level)
                canonical_after = (
                    battle.opponent.active.item,
                    battle.opponent.active.ability,
                    battle.opponent.active.original_ability,
                    tuple(battle.opponent.active.moves),
                    battle.opponent.active.nature,
                    tuple(battle.opponent.active.evs),
                    tuple(battle.opponent.active.ivs),
                )
                self.assertEqual(canonical_before, canonical_after)
                seen_species.add(species_id)
                seen_variants.add(key)
        self.assertEqual(set(EXPECTED_SPECIES), seen_species)
        self.assertEqual(set(EXPECTED_VARIANTS), seen_variants)

    def test_22_public_evidence_filters_moves_items_and_abilities_exactly(self):
        context = self.none_configuration.create_battle_context("gen9tugs")
        for key, expected in EXPECTED_VARIANTS.items():
            species_id, variant_id = key
            record = self.dataset.get_species(species_id)
            evidence_battle = _battle(context, species_id)
            _record_complete_public_evidence(evidence_battle, species_id, expected)
            evidence = evidence_battle.team_inference.observation_ledger.member(
                species_id
            )
            result = select_public_prior_variant(
                context,
                battle_format="gen9tugs",
                species_id=species_id,
                level=100,
                evidence=evidence,
                rng=_FixedRng(0.5),
            )
            with self.subTest(species=species_id, variant=variant_id):
                self.assertIs(PublicPriorSelectionStatus.SELECTED, result.status)
                self.assertIs(record.get_variant(variant_id), result.variant)

    def test_23_weights_are_scoped_to_selected_species_and_dataset(self):
        context = self.none_configuration.create_battle_context("gen9tugs")
        for species_id, variant_ids in EXPECTED_VARIANT_IDS.items():
            with self.subTest(species=species_id, edge="first"):
                first = select_public_prior_variant(
                    context,
                    battle_format="gen9tugs",
                    species_id=species_id,
                    level=100,
                    evidence=None,
                    rng=_FixedRng(0.0),
                )
                self.assertEqual(variant_ids[0], first.variant.variant_id)
                self.assertIn((species_id, first.variant.variant_id), EXPECTED_VARIANTS)
            with self.subTest(species=species_id, edge="last"):
                last = select_public_prior_variant(
                    context,
                    battle_format="gen9tugs",
                    species_id=species_id,
                    level=100,
                    evidence=None,
                    rng=_FixedRng(0.999999999),
                )
                self.assertEqual(variant_ids[-1], last.variant.variant_id)
                self.assertIn((species_id, last.variant.variant_id), EXPECTED_VARIANTS)

    def test_24_uncovered_species_uses_generic_only_under_generic_fallback(self):
        context = self.generic_configuration.create_battle_context("gen9tugs")
        battle = _battle(context, "aerodactyl")
        with mock.patch("fp.search.standard_battles.sample_pokemon") as generic:
            sampled = prepare_battles(battle, 1)[0][0]
        generic.assert_called_once()
        self.assertEqual("aerodactyl", sampled.opponent.active.name)

    def test_25_uncovered_species_remains_unknown_under_none_without_error(self):
        context = self.none_configuration.create_battle_context("gen9tugs")
        battle = _battle(context, "aerodactyl")
        before = (
            battle.opponent.active.item,
            battle.opponent.active.ability,
            tuple(battle.opponent.active.moves),
            battle.opponent.active.nature,
            tuple(battle.opponent.active.evs),
            tuple(battle.opponent.active.ivs),
        )
        with mock.patch("fp.search.standard_battles.sample_pokemon") as generic:
            sampled = prepare_battles(battle, 1)[0][0]
        generic.assert_not_called()
        after = (
            sampled.opponent.active.item,
            sampled.opponent.active.ability,
            tuple(sampled.opponent.active.moves),
            sampled.opponent.active.nature,
            tuple(sampled.opponent.active.evs),
            tuple(sampled.opponent.active.ivs),
        )
        self.assertEqual(before, after)
        self.assertEqual(constants.UNKNOWN_ITEM, sampled.opponent.active.item)

    def test_26_loading_and_sampling_are_read_only_offline_and_overlay_stable(self):
        source_before = PRODUCTION_PATH.read_bytes()
        moves_before = copy.deepcopy(all_move_json)
        pokedex_before = copy.deepcopy(pokedex)
        directory_before = tuple(
            sorted(path.name for path in PRODUCTION_PATH.parent.iterdir())
        )
        writes = ("write_text", "write_bytes", "touch", "mkdir")
        patches = [
            mock.patch.object(
                Path, method, side_effect=AssertionError("write attempted")
            )
            for method in writes
        ]
        for patcher in patches:
            patcher.start()
        try:
            with mock.patch.object(
                socket,
                "create_connection",
                side_effect=AssertionError("network attempted"),
            ):
                configuration = _startup(PublicPriorFallback.NONE)
                context = configuration.create_battle_context("gen9tugs")
                battle = _battle(context, "druddigon")
                expected = EXPECTED_VARIANTS[("druddigon", "roughskinutility")]
                _record_complete_public_evidence(battle, "druddigon", expected)
                prepare_battles(battle, 1)
        finally:
            for patcher in reversed(patches):
                patcher.stop()
        self.assertEqual(source_before, PRODUCTION_PATH.read_bytes())
        self.assertEqual(moves_before, all_move_json)
        self.assertEqual(pokedex_before, pokedex)
        self.assertEqual(
            directory_before,
            tuple(sorted(path.name for path in PRODUCTION_PATH.parent.iterdir())),
        )


if __name__ == "__main__":
    unittest.main()
