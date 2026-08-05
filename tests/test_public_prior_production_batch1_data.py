import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from fp import constants
from fp.battle.public_prior_context import PublicPriorFallback
from fp.battle.state import Battle, LastUsedMove, Pokemon
from fp.battle.team_inference import PublicObservationSource
from fp.config import FoulPlayConfig
from fp.data import all_move_json, pokedex
from fp.data.mods.apply_mods import apply_mods
from fp.data.public_priors import PublicPriorIdentity, load_public_prior
from fp.data.public_priors.runtime import (
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
POOLS = ROOT / "fp" / "data" / "public_priors" / "pools"
VERSION_1_0_PATH = POOLS / "tugspublicarchetypes-1.0.0.json"
VERSION_1_1_PATH = POOLS / "tugspublicarchetypes-1.1.0.json"
VERSION_1_0_RAW_SHA256 = (
    "690c943f0ae554f68423f988f6a61d43f4310f25a2db8e65f98238d8c92c2a91"
)
VERSION_1_1_RAW_SHA256 = (
    "1a5a4d9bb3ad82e29eae907fcc2a41698aeaf1ec55fb8375ad18197e874662e4"
)
VERSION_1_1_CANONICAL_SHA256 = (
    "59425bcd4371870ec6489d384ba9a7d6a3c3c5375371fbfc48c11fab17740d8d"
)
IDENTITY = PublicPriorIdentity("tugspublicarchetypes", "1.1.0", "gen9tugs")
SOURCE_IDS = ("publicformatpatch12", "publicmanualv1")
AUTHORIZATION_PATH = "$.metadata.authorization"
AUTHORIZATION_VALUE = (
    "Generic public design prior authored without closed-submission data."
)
NEW_VARIANT_IDS = {
    "aerodactyl": ("choiceband", "dragondance", "sashlead"),
    "jellicent": ("choicespecs", "cursedbodyutility", "defensivespinblocker"),
    "jynx": ("choicespecs", "nastyplot", "sashdisruption"),
    "obstagoon": ("bulkysetup", "choicescarf", "flameorbguts"),
    "starmie": ("analyticbreaker", "rapidspinutility"),
    "swampert": ("choiceband", "cursechesto", "rockypivot"),
    "sylveon": ("calmmind", "choicespecs", "wishsupport"),
    "tinkaton": ("assaultvest", "rocksutility", "swordsdance"),
    "torkoal": ("droughtspinner", "overheatmomentum", "sunbreaker"),
    "vikavolt": ("bootspivot", "specsbreaker", "stickyweb"),
}
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


def _document(path):
    return json.loads(path.read_text(encoding="utf-8", errors="strict"))


def _canonical_bytes(document):
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    ).encode("utf-8")


def _assert_public_data_firewall(document):
    def check_string(value, path, *, is_key=False):
        folded = value.casefold()
        for pattern in FORBIDDEN_STRING_PATTERNS:
            if pattern not in folded:
                continue
            if (
                not is_key
                and path == AUTHORIZATION_PATH
                and value == AUTHORIZATION_VALUE
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


class _FixedRng:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


def _startup(fallback):
    options = PublicPriorStartupOptions((str(VERSION_1_1_PATH),), fallback)
    return load_public_prior_runtime_configuration(options, "gen9tugs")


def _battle(context, species_id):
    battle = Battle("batch-one-public-prior", public_prior_context=context)
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


def _record_complete_public_evidence(battle, species_id, variant):
    for move_id in variant.move_ids:
        battle.team_inference.record_selected_move(species_id, move_id)
    battle.team_inference.record_initial_item(
        species_id,
        variant.item_id,
        source=PublicObservationSource.DIRECT_ITEM_REVEAL,
    )
    battle.team_inference.record_base_ability(species_id, variant.base_ability_id)


def _pokemon_snapshot(pokemon):
    return (
        pokemon.item,
        pokemon.ability,
        pokemon.original_ability,
        tuple(move.name for move in pokemon.moves),
        pokemon.nature,
        tuple(pokemon.evs),
        tuple(pokemon.ivs),
        pokemon.level,
    )


class TestBatchOneDocument(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document_1_0 = _document(VERSION_1_0_PATH)
        cls.document_1_1 = _document(VERSION_1_1_PATH)
        cls.dataset_1_0 = load_public_prior(VERSION_1_0_PATH)
        cls.dataset_1_1 = load_public_prior(VERSION_1_1_PATH)

    def test_01_encoding_hash_identity_and_inventory_are_exact(self):
        raw = VERSION_1_1_PATH.read_bytes()
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\r", raw)
        self.assertIsInstance(raw.decode("utf-8", errors="strict"), str)
        self.assertEqual(69667, len(raw))
        self.assertEqual(VERSION_1_1_RAW_SHA256, hashlib.sha256(raw).hexdigest())
        self.assertEqual(
            VERSION_1_1_CANONICAL_SHA256,
            hashlib.sha256(_canonical_bytes(self.document_1_1)).hexdigest(),
        )
        self.assertEqual(IDENTITY, self.dataset_1_1.identity)
        self.assertEqual("1.2", self.dataset_1_1.patch_version)
        self.assertEqual("public", self.dataset_1_1.visibility)
        self.assertEqual(18, len(self.dataset_1_1.species))
        self.assertEqual(
            52,
            sum(len(record.variants) for record in self.dataset_1_1.species),
        )
        self.assertEqual(29, sum(map(len, NEW_VARIANT_IDS.values())))

    def test_02_handcrafted_aerodactyl_variants_are_exact(self):
        record = self.dataset_1_1.get_species("aerodactyl")
        actual = {
            variant.variant_id: (
                variant.item_id,
                variant.base_ability_id,
                variant.nature_id,
                variant.evs.as_tuple(),
                variant.move_ids,
            )
            for variant in record.variants
        }
        self.assertEqual(
            {
                "choiceband": (
                    "choiceband",
                    "pressure",
                    "jolly",
                    (4, 252, 0, 0, 0, 252),
                    ("earthquake", "stoneedge", "dualwingbeat", "firefang"),
                ),
                "dragondance": (
                    "expertbelt",
                    "unnerve",
                    "adamant",
                    (4, 252, 0, 0, 0, 252),
                    ("dragondance", "earthquake", "stoneedge", "dualwingbeat"),
                ),
                "sashlead": (
                    "focussash",
                    "pressure",
                    "jolly",
                    (4, 252, 0, 0, 0, 252),
                    ("earthquake", "stoneedge", "taunt", "stealthrock"),
                ),
            },
            actual,
        )

    def test_03_version_1_0_is_unchanged_and_semantically_preserved(self):
        self.assertEqual(
            VERSION_1_0_RAW_SHA256,
            hashlib.sha256(VERSION_1_0_PATH.read_bytes()).hexdigest(),
        )
        for key in (
            "schema_version",
            "visibility",
            "dataset_id",
            "format_id",
            "patch_version",
            "metadata",
            "sources",
        ):
            self.assertEqual(self.document_1_0[key], self.document_1_1[key])
        records_1_0 = {
            record["species_id"]: record for record in self.document_1_0["species"]
        }
        records_1_1 = {
            record["species_id"]: record for record in self.document_1_1["species"]
        }
        self.assertEqual(
            records_1_0,
            {species_id: records_1_1[species_id] for species_id in records_1_0},
        )
        self.assertEqual(8, len(self.dataset_1_0.species))
        self.assertEqual(
            23,
            sum(len(record.variants) for record in self.dataset_1_0.species),
        )

    def test_04_new_records_are_complete_coherent_public_schema_values(self):
        setup_moves = {"bulkup", "calmmind", "curse", "dragondance", "nastyplot", "swordsdance"}
        choice_items = {"choiceband", "choicescarf", "choicespecs"}
        for species_id, expected_ids in NEW_VARIANT_IDS.items():
            record = self.dataset_1_1.get_species(species_id)
            with self.subTest(species=species_id):
                self.assertIsNotNone(record)
                self.assertEqual(expected_ids, tuple(v.variant_id for v in record.variants))
                self.assertIn(len(record.variants), (2, 3))
            for variant in record.variants:
                with self.subTest(species=species_id, variant=variant.variant_id):
                    self.assertTrue(math.isfinite(variant.weight))
                    self.assertGreater(variant.weight, 0)
                    self.assertEqual(4, len(variant.move_ids))
                    self.assertEqual(4, len(set(variant.move_ids)))
                    self.assertLessEqual(sum(variant.evs.as_tuple()), 510)
                    self.assertEqual(6, len(variant.ivs.as_tuple()))
                    self.assertTrue(all(0 <= value <= 31 for value in variant.ivs.as_tuple()))
                    self.assertEqual(100, variant.level)
                    self.assertEqual(SOURCE_IDS, variant.source_ids)
                    self.assertEqual(
                        {"role", "rationale", "distinguishing_evidence", "weight_reason"},
                        set(variant.metadata),
                    )
                    self.assertTrue(all(variant.metadata.values()))
                    if variant.item_id in choice_items:
                        self.assertTrue(setup_moves.isdisjoint(variant.move_ids))
                    if variant.item_id == "assaultvest":
                        self.assertTrue(
                            all(
                                all_move_json[move_id]["category"] != "Status"
                                for move_id in variant.move_ids
                            )
                        )
        source = VERSION_1_1_PATH.read_text(encoding="utf-8").casefold()
        for tera_token in ('"tera_type"', '"tera"', "terablast", "terastallization"):
            self.assertNotIn(tera_token, source)

    def test_05_public_data_firewall_is_path_and_value_specific(self):
        _assert_public_data_firewall(self.document_1_1)
        self.assertEqual(
            AUTHORIZATION_VALUE,
            self.document_1_1["metadata"]["authorization"],
        )
        wrong_path = copy.deepcopy(self.document_1_1)
        wrong_path["metadata"]["note"] = AUTHORIZATION_VALUE
        with self.assertRaises(AssertionError):
            _assert_public_data_firewall(wrong_path)
        altered = copy.deepcopy(self.document_1_1)
        altered["metadata"]["authorization"] += " Altered."
        with self.assertRaises(AssertionError):
            _assert_public_data_firewall(altered)
        forbidden_key = copy.deepcopy(self.document_1_1)
        forbidden_key["metadata"]["candidate_id"] = "x"
        with self.assertRaises(AssertionError):
            _assert_public_data_firewall(forbidden_key)

    def test_06_current_tugs_teamvalidator_accepts_both_full_snapshots(self):
        server_root = Path(
            os.environ.get("TUGS_SHOWDOWN_ROOT", ROOT.parent / "TUGS-showdown")
        ).resolve()
        node = shutil.which("node")
        self.assertIsNotNone(node, "Node.js is required for the TUGS TeamValidator test")
        self.assertTrue((server_root / "dist" / "sim" / "team-validator.js").is_file())
        validator_script = r"""
const fs = require('fs');
const {TeamValidator} = require('./dist/sim/team-validator');
const documents = JSON.parse(fs.readFileSync(0, 'utf8'));
const validator = new TeamValidator('gen9tugs');
let invalid = 0;
for (const document of documents) {
  let valid = 0;
  for (const species of document.species) {
    for (const variant of species.variants) {
      const set = {
        name: species.species_id,
        species: species.species_id,
        item: variant.item_id,
        ability: variant.base_ability_id,
        moves: variant.move_ids,
        nature: variant.nature_id,
        evs: variant.evs,
        ivs: variant.ivs,
        level: variant.level,
      };
      const problems = validator.validateTeam([set]);
      if (problems) {
        invalid++;
        console.error(
          `${document.dataset_version} ${species.species_id}/${variant.variant_id}: ` +
          problems.join(' | ')
        );
      } else {
        valid++;
      }
    }
  }
  console.log(`${document.dataset_version}:${valid}`);
}
process.exitCode = invalid ? 1 : 0;
"""
        completed = subprocess.run(
            [node, "-e", validator_script],
            input=json.dumps([self.document_1_0, self.document_1_1]),
            cwd=server_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertEqual(["1.0.0:23", "1.1.0:52"], completed.stdout.splitlines())


class TestBatchOneSelectionAndPopulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generic_configuration = _startup(PublicPriorFallback.GENERIC)
        cls.none_configuration = _startup(PublicPriorFallback.NONE)
        cls.dataset = cls.none_configuration.registry.get(IDENTITY)

    def test_06_controlled_rng_selects_every_new_variant(self):
        context = self.none_configuration.create_battle_context("gen9tugs")
        selected = set()
        for species_id in NEW_VARIANT_IDS:
            record = self.dataset.get_species(species_id)
            total = sum(variant.weight for variant in record.variants)
            cumulative = 0.0
            for variant in record.variants:
                rng_value = (cumulative + variant.weight / 2) / total
                result = select_public_prior_variant(
                    context,
                    battle_format="gen9tugs",
                    species_id=species_id,
                    level=100,
                    evidence=None,
                    rng=_FixedRng(rng_value),
                )
                with self.subTest(species=species_id, variant=variant.variant_id):
                    self.assertIs(PublicPriorSelectionStatus.SELECTED, result.status)
                    self.assertIs(variant, result.variant)
                selected.add((species_id, variant.variant_id))
                cumulative += variant.weight
        self.assertEqual(29, len(selected))

    def test_07_complete_public_evidence_narrows_to_every_new_variant(self):
        context = self.none_configuration.create_battle_context("gen9tugs")
        for species_id in NEW_VARIANT_IDS:
            record = self.dataset.get_species(species_id)
            for variant in record.variants:
                battle = _battle(context, species_id)
                _record_complete_public_evidence(battle, species_id, variant)
                evidence = battle.team_inference.observation_ledger.member(species_id)
                result = select_public_prior_variant(
                    context,
                    battle_format="gen9tugs",
                    species_id=species_id,
                    level=100,
                    evidence=evidence,
                    rng=_FixedRng(0.5),
                )
                with self.subTest(species=species_id, variant=variant.variant_id):
                    self.assertIs(PublicPriorSelectionStatus.SELECTED, result.status)
                    self.assertIs(variant, result.variant)

    def test_08_population_preserves_reveals_and_canonical_state(self):
        context = self.none_configuration.create_battle_context("gen9tugs")
        modules_before = set(sys.modules)
        populated = set()
        for species_id in NEW_VARIANT_IDS:
            record = self.dataset.get_species(species_id)
            for variant in record.variants:
                battle = _battle(context, species_id)
                pokemon = battle.opponent.active
                pokemon.item = variant.item_id
                pokemon.ability = variant.base_ability_id
                pokemon.original_ability = variant.base_ability_id
                pokemon.add_move(variant.move_ids[0])
                _record_complete_public_evidence(battle, species_id, variant)
                canonical_before = _pokemon_snapshot(pokemon)
                with mock.patch("fp.search.standard_battles.sample_pokemon") as generic:
                    sampled = prepare_battles(battle, 1)[0][0]
                generic.assert_not_called()
                sampled_pokemon = sampled.opponent.active
                with self.subTest(species=species_id, variant=variant.variant_id):
                    self.assertEqual(variant.item_id, sampled_pokemon.item)
                    self.assertEqual(variant.base_ability_id, sampled_pokemon.ability)
                    self.assertEqual(
                        variant.base_ability_id, sampled_pokemon.original_ability
                    )
                    self.assertEqual(
                        variant.move_ids,
                        tuple(move.name for move in sampled_pokemon.moves),
                    )
                    self.assertEqual(variant.nature_id, sampled_pokemon.nature)
                    self.assertEqual(variant.evs.as_tuple(), tuple(sampled_pokemon.evs))
                    self.assertEqual(variant.ivs.as_tuple(), tuple(sampled_pokemon.ivs))
                    self.assertEqual(canonical_before, _pokemon_snapshot(pokemon))
                populated.add((species_id, variant.variant_id))
        newly_imported_private = {
            name
            for name in set(sys.modules) - modules_before
            if name.startswith("fp.data.team_pools")
        }
        self.assertEqual(set(), newly_imported_private)
        self.assertEqual(29, len(populated))

    def test_09_public_sampling_sources_do_not_reference_private_pool_models(self):
        paths = (
            ROOT / "fp" / "battle" / "public_prior_context.py",
            ROOT / "fp" / "data" / "public_priors" / "runtime.py",
            ROOT / "fp" / "search" / "public_prior_sampling.py",
            ROOT / "fp" / "search" / "standard_battles.py",
        )
        forbidden = (
            "fp.data.team_pools",
            "TeamPool",
            "TeamRecord",
            "PokemonRecord",
            "baseline_candidate_ids",
            "active_candidate_ids",
        )
        for path in paths:
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)

    def test_10_uncovered_species_uses_only_explicit_generic_fallback(self):
        context = self.generic_configuration.create_battle_context("gen9tugs")
        battle = _battle(context, "xatu")
        with mock.patch("fp.search.standard_battles.sample_pokemon") as generic:
            sampled = prepare_battles(battle, 1)[0][0]
        generic.assert_called_once()
        self.assertEqual("xatu", sampled.opponent.active.name)

    def test_11_uncovered_species_remains_unknown_under_none_fallback(self):
        context = self.none_configuration.create_battle_context("gen9tugs")
        battle = _battle(context, "xatu")
        before = _pokemon_snapshot(battle.opponent.active)
        with mock.patch("fp.search.standard_battles.sample_pokemon") as generic:
            sampled = prepare_battles(battle, 1)[0][0]
        generic.assert_not_called()
        self.assertEqual(before, _pokemon_snapshot(sampled.opponent.active))
        self.assertEqual(constants.UNKNOWN_ITEM, sampled.opponent.active.item)


if __name__ == "__main__":
    unittest.main()
