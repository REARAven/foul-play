import copy
import hashlib
import inspect
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
from fp.data.public_priors import (
    PublicPriorIdentity,
    PublicPriorValidationError,
    load_public_prior,
    validate_public_prior_document,
)
from fp.data.public_priors.runtime import (
    PublicPriorStartupOptions,
    load_public_prior_runtime_configuration,
)
from fp.format_spec import FormatSpec
from fp.search.public_prior_sampling import (
    PublicPriorSelectionStatus,
    public_variant_is_compatible,
    select_public_prior_variant,
)
from fp.search.standard_battles import prepare_battles


ROOT = Path(__file__).resolve().parents[1]
POOLS = ROOT / "fp" / "data" / "public_priors" / "pools"
VERSION_PATHS = {
    version: POOLS / f"tugspublicarchetypes-{version}.json"
    for version in ("1.0.0", "1.1.0", "1.2.0")
}
RAW_SHA256 = {
    "1.0.0": "690c943f0ae554f68423f988f6a61d43f4310f25a2db8e65f98238d8c92c2a91",
    "1.1.0": "1a5a4d9bb3ad82e29eae907fcc2a41698aeaf1ec55fb8375ad18197e874662e4",
    "1.2.0": "1ebe4e9ad85ba30ab0c166f22cd8a8088a59d77e20f937162b9e528f0bb04b24",
}
CANONICAL_SHA256 = {
    "1.0.0": "ddb04702321aace7d7368938016a18d54841935d2bf5df194cfaf7af8d4a55ae",
    "1.1.0": "59425bcd4371870ec6489d384ba9a7d6a3c3c5375371fbfc48c11fab17740d8d",
    "1.2.0": "bc0456562feaccf6fc4a64bf0d189deec8b5c6331c5c64d8bee606674d946376",
}
IDENTITY = PublicPriorIdentity("tugspublicarchetypes", "1.2.0", "gen9tugs")
SOURCE_IDS = ("publicformatpatch12", "publicmanualv1")
AUTHORIZATION_PATH = "$.metadata.authorization"
AUTHORIZATION_VALUE = (
    "Generic public design prior authored without closed-submission data."
)
NEW_VARIANT_IDS = {
    "altaria": ("choicespecs", "perishtrap", "physicalutility"),
    "arcaninehisui": ("bootsrocks", "choiceband", "intimidateutility"),
    "bombirdier": ("bootsutility", "choiceband", "choicescarf"),
    "chesnaught": ("helmetspikes", "leftoversutility"),
    "cursola": ("perishbodyspecs", "specialutility", "weakarmorspecs"),
    "dhelmise": ("choiceband", "swordsdance"),
    "drapion": ("swordsdance", "toxicspikes"),
    "dudunsparce": ("calmmind", "coil", "glareutility"),
    "dudunsparcethreesegment": ("calmmind", "coil", "glareutility"),
    "dugtrioalola": ("choiceband", "choicescarf", "sashlead", "sashsetup"),
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
IVS_31 = (31, 31, 31, 31, 31, 31)
IVS_0_ATK = (31, 0, 31, 31, 31, 31)


def _expected(weight, item, ability, nature, evs, ivs, moves):
    return {
        "weight": weight,
        "item": item,
        "ability": ability,
        "nature": nature,
        "evs": evs,
        "ivs": ivs,
        "moves": moves,
    }


EXPECTED_NEW_SETS = {
    ("altaria", "perishtrap"): _expected(
        2,
        "heavydutyboots",
        "naturalcure",
        "calm",
        (252, 0, 0, 4, 252, 0),
        IVS_0_ATK,
        ("perishsong", "firespin", "roost", "defog"),
    ),
    ("altaria", "physicalutility"): _expected(
        4,
        "heavydutyboots",
        "naturalcure",
        "impish",
        (252, 0, 240, 0, 0, 16),
        IVS_31,
        ("defog", "roost", "bravebird", "willowisp"),
    ),
    ("altaria", "choicespecs"): _expected(
        3,
        "choicespecs",
        "naturalcure",
        "timid",
        (0, 0, 0, 252, 4, 252),
        IVS_0_ATK,
        ("dracometeor", "flamethrower", "hurricane", "moonblast"),
    ),
    ("arcaninehisui", "bootsrocks"): _expected(
        3,
        "heavydutyboots",
        "rockhead",
        "jolly",
        (0, 252, 0, 0, 4, 252),
        IVS_31,
        ("headsmash", "flareblitz", "extremespeed", "stealthrock"),
    ),
    ("arcaninehisui", "choiceband"): _expected(
        4,
        "choiceband",
        "rockhead",
        "jolly",
        (0, 252, 0, 0, 4, 252),
        IVS_31,
        ("headsmash", "flareblitz", "extremespeed", "closecombat"),
    ),
    ("arcaninehisui", "intimidateutility"): _expected(
        3,
        "heavydutyboots",
        "intimidate",
        "jolly",
        (0, 252, 0, 0, 4, 252),
        IVS_31,
        ("stealthrock", "morningsun", "flareblitz", "extremespeed"),
    ),
    ("bombirdier", "bootsutility"): _expected(
        4,
        "heavydutyboots",
        "bigpecks",
        "jolly",
        (4, 252, 0, 0, 0, 252),
        IVS_31,
        ("uturn", "stealthrock", "knockoff", "memento"),
    ),
    ("bombirdier", "choicescarf"): _expected(
        3,
        "choicescarf",
        "rockypayload",
        "jolly",
        (4, 252, 0, 0, 0, 252),
        IVS_31,
        ("uturn", "knockoff", "stoneedge", "bravebird"),
    ),
    ("bombirdier", "choiceband"): _expected(
        3,
        "choiceband",
        "bigpecks",
        "jolly",
        (4, 252, 0, 0, 0, 252),
        IVS_31,
        ("suckerpunch", "uturn", "knockoff", "bravebird"),
    ),
    ("chesnaught", "helmetspikes"): _expected(
        4,
        "rockyhelmet",
        "bulletproof",
        "impish",
        (252, 4, 252, 0, 0, 0),
        IVS_31,
        ("bodypress", "knockoff", "spikes", "synthesis"),
    ),
    ("chesnaught", "leftoversutility"): _expected(
        3,
        "leftovers",
        "bulletproof",
        "impish",
        (252, 4, 252, 0, 0, 0),
        IVS_31,
        ("spikyshield", "toxic", "knockoff", "drainpunch"),
    ),
    ("cursola", "specialutility"): _expected(
        4,
        "leftovers",
        "perishbody",
        "calm",
        (252, 0, 0, 4, 252, 0),
        IVS_0_ATK,
        ("hex", "scald", "stealthrock", "willowisp"),
    ),
    ("cursola", "perishbodyspecs"): _expected(
        3,
        "choicespecs",
        "perishbody",
        "modest",
        (252, 0, 0, 252, 4, 0),
        IVS_0_ATK,
        ("shadowball", "surf", "earthpower", "icebeam"),
    ),
    ("cursola", "weakarmorspecs"): _expected(
        2,
        "choicespecs",
        "weakarmor",
        "modest",
        (0, 0, 4, 252, 0, 252),
        IVS_0_ATK,
        ("shadowball", "surf", "earthpower", "icebeam"),
    ),
    ("dhelmise", "swordsdance"): _expected(
        4,
        "heavydutyboots",
        "steelworker",
        "adamant",
        (248, 252, 0, 0, 8, 0),
        IVS_31,
        ("anchorshot", "swordsdance", "powerwhip", "knockoff"),
    ),
    ("dhelmise", "choiceband"): _expected(
        3,
        "choiceband",
        "steelworker",
        "adamant",
        (248, 252, 0, 0, 8, 0),
        IVS_31,
        ("powerwhip", "knockoff", "anchorshot", "poltergeist"),
    ),
    ("drapion", "swordsdance"): _expected(
        4,
        "heavydutyboots",
        "battlearmor",
        "jolly",
        (4, 252, 0, 0, 0, 252),
        IVS_31,
        ("knockoff", "poisonjab", "earthquake", "swordsdance"),
    ),
    ("drapion", "toxicspikes"): _expected(
        3,
        "heavydutyboots",
        "battlearmor",
        "jolly",
        (4, 252, 0, 0, 0, 252),
        IVS_31,
        ("earthquake", "knockoff", "toxicspikes", "taunt"),
    ),
    ("dudunsparce", "calmmind"): _expected(
        4,
        "heavydutyboots",
        "thickfat",
        "bold",
        (252, 0, 252, 4, 0, 0),
        IVS_0_ATK,
        ("calmmind", "boomburst", "shadowball", "roost"),
    ),
    ("dudunsparce", "glareutility"): _expected(
        2,
        "chopleberry",
        "thickfat",
        "modest",
        (116, 0, 0, 252, 0, 140),
        IVS_0_ATK,
        ("glare", "boomburst", "hex", "stealthrock"),
    ),
    ("dudunsparce", "coil"): _expected(
        3,
        "leftovers",
        "serenegrace",
        "careful",
        (252, 4, 0, 0, 252, 0),
        IVS_31,
        ("coil", "bodyslam", "dragontail", "roost"),
    ),
    ("dudunsparcethreesegment", "calmmind"): _expected(
        4,
        "heavydutyboots",
        "thickfat",
        "bold",
        (252, 0, 252, 4, 0, 0),
        IVS_0_ATK,
        ("calmmind", "boomburst", "shadowball", "roost"),
    ),
    ("dudunsparcethreesegment", "glareutility"): _expected(
        2,
        "chopleberry",
        "thickfat",
        "modest",
        (116, 0, 0, 252, 0, 140),
        IVS_0_ATK,
        ("glare", "boomburst", "hex", "stealthrock"),
    ),
    ("dudunsparcethreesegment", "coil"): _expected(
        3,
        "leftovers",
        "serenegrace",
        "careful",
        (252, 4, 0, 0, 252, 0),
        IVS_31,
        ("coil", "bodyslam", "dragontail", "roost"),
    ),
    ("dugtrioalola", "sashsetup"): _expected(
        3,
        "focussash",
        "tanglinghair",
        "jolly",
        (4, 252, 0, 0, 0, 252),
        IVS_31,
        ("suckerpunch", "swordsdance", "earthquake", "ironhead"),
    ),
    ("dugtrioalola", "sashlead"): _expected(
        4,
        "focussash",
        "tanglinghair",
        "jolly",
        (4, 252, 0, 0, 0, 252),
        IVS_31,
        ("stealthrock", "earthquake", "ironhead", "memento"),
    ),
    ("dugtrioalola", "choicescarf"): _expected(
        2,
        "choicescarf",
        "tanglinghair",
        "jolly",
        (4, 252, 0, 0, 0, 252),
        IVS_31,
        ("earthquake", "ironhead", "stoneedge", "pursuit"),
    ),
    ("dugtrioalola", "choiceband"): _expected(
        3,
        "choiceband",
        "tanglinghair",
        "jolly",
        (4, 252, 0, 0, 0, 252),
        IVS_31,
        ("earthquake", "ironhead", "stoneedge", "suckerpunch"),
    ),
}

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


def _document(version):
    return json.loads(
        VERSION_PATHS[version].read_text(encoding="utf-8", errors="strict")
    )


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
                f"forbidden public-data pattern {pattern!r} at {path}: {value!r}"
            )

    def walk(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.casefold() in FORBIDDEN_STRUCTURAL_KEYS:
                    raise AssertionError(f"forbidden structural key {key!r} at {path}")
                check_string(key, f"{path}.<key:{key}>", is_key=True)
                walk(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
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
    options = PublicPriorStartupOptions((str(VERSION_PATHS["1.2.0"]),), fallback)
    return load_public_prior_runtime_configuration(options, "gen9tugs")


def _battle(context, species_id):
    battle = Battle("batch-two-public-prior", public_prior_context=context)
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


def _evidence(context, species_id, *, moves=(), item=None, ability=None):
    battle = _battle(context, species_id)
    for move_id in moves:
        battle.team_inference.record_selected_move(species_id, move_id)
    if item is not None:
        battle.team_inference.record_initial_item(
            species_id,
            item,
            source=PublicObservationSource.DIRECT_ITEM_REVEAL,
        )
    if ability is not None:
        battle.team_inference.record_base_ability(species_id, ability)
    return battle.team_inference.observation_ledger.member(species_id)


def _compatible_ids(dataset, species_id, evidence):
    return {
        variant.variant_id
        for variant in dataset.get_species(species_id).variants
        if public_variant_is_compatible(variant, species_id, 100, evidence)
    }


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


def _run_teamvalidator(documents):
    server_root = Path(
        os.environ.get("TUGS_SHOWDOWN_ROOT", ROOT.parent / "TUGS-showdown")
    ).resolve()
    node = shutil.which("node")
    if node is None:
        raise AssertionError("Node.js is required for the TUGS TeamValidator test")
    validator_script = r"""
const fs = require('fs');
const {TeamValidator} = require('./dist/sim/team-validator');
const payload = JSON.parse(fs.readFileSync(0, 'utf8'));
const validator = new TeamValidator('gen9tugs');
let invalid = 0;
for (const group of payload) {
  let valid = 0;
  for (const entry of group.entries) {
    const set = {
      species: entry.species_id,
      item: entry.variant.item_id,
      ability: entry.variant.base_ability_id,
      moves: entry.variant.move_ids,
      nature: entry.variant.nature_id,
      evs: entry.variant.evs,
      ivs: entry.variant.ivs,
      level: entry.variant.level,
    };
    const problems = validator.validateTeam([set]);
    if (problems) {
      invalid++;
      console.error(
        group.label + ' ' + entry.species_id + '/' +
        entry.variant.variant_id + ': ' + problems.join(' | ')
      );
    } else {
      valid++;
    }
  }
  console.log(group.label + ':' + valid + '/' + group.entries.length);
}
process.exitCode = invalid ? 1 : 0;
"""
    return subprocess.run(
        [node, "-e", validator_script],
        input=json.dumps(documents),
        cwd=server_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _entries(document, predicate=lambda species, variant: True):
    return [
        {"species_id": species["species_id"], "variant": variant}
        for species in document["species"]
        for variant in species["variants"]
        if predicate(species, variant)
    ]


class TestBatchTwoDocument(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents = {version: _document(version) for version in VERSION_PATHS}
        cls.datasets = {
            version: load_public_prior(path) for version, path in VERSION_PATHS.items()
        }

    def test_01_encoding_hash_identity_and_inventories_are_exact(self):
        for version, expected_species, expected_variants in (
            ("1.0.0", 8, 23),
            ("1.1.0", 18, 52),
            ("1.2.0", 28, 80),
        ):
            raw = VERSION_PATHS[version].read_bytes()
            with self.subTest(version=version):
                self.assertIsInstance(raw.decode("utf-8", errors="strict"), str)
                self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
                self.assertNotIn(b"\r", raw)
                self.assertTrue(raw.endswith(b"\n"))
                self.assertEqual(RAW_SHA256[version], hashlib.sha256(raw).hexdigest())
                self.assertEqual(
                    CANONICAL_SHA256[version],
                    hashlib.sha256(
                        _canonical_bytes(self.documents[version])
                    ).hexdigest(),
                )
                dataset = self.datasets[version]
                self.assertEqual("public", dataset.visibility)
                self.assertEqual("gen9tugs", dataset.identity.format_id)
                self.assertEqual(expected_species, len(dataset.species))
                self.assertEqual(
                    expected_variants,
                    sum(len(record.variants) for record in dataset.species),
                )
        self.assertEqual(107311, len(VERSION_PATHS["1.2.0"].read_bytes()))
        self.assertEqual(IDENTITY, self.datasets["1.2.0"].identity)
        self.assertEqual("1.2", self.datasets["1.2.0"].patch_version)
        self.assertEqual(10, len(NEW_VARIANT_IDS))
        self.assertEqual(28, sum(map(len, NEW_VARIANT_IDS.values())))

    def test_02_version_1_1_semantics_change_only_as_authorized(self):
        old = self.documents["1.1.0"]
        new = self.documents["1.2.0"]
        for key in (
            "schema_version",
            "visibility",
            "dataset_id",
            "format_id",
            "patch_version",
            "metadata",
            "sources",
        ):
            self.assertEqual(old[key], new[key], key)
        old_records = {record["species_id"]: record for record in old["species"]}
        new_records = {record["species_id"]: record for record in new["species"]}
        self.assertEqual(10, len(set(new_records) - set(old_records)))
        for species_id, old_record in old_records.items():
            new_record = new_records[species_id]
            if species_id != "aerodactyl":
                self.assertEqual(old_record, new_record, species_id)
                continue
            old_variants = {
                variant["variant_id"]: variant for variant in old_record["variants"]
            }
            new_variants = {
                variant["variant_id"]: variant for variant in new_record["variants"]
            }
            self.assertEqual(old_variants.keys(), new_variants.keys())
            for variant_id in old_variants:
                if variant_id != "dragondance":
                    self.assertEqual(old_variants[variant_id], new_variants[variant_id])
            old_dd = copy.deepcopy(old_variants["dragondance"])
            new_dd = copy.deepcopy(new_variants["dragondance"])
            self.assertEqual("expertbelt", old_dd["item_id"])
            self.assertEqual("lifeorb", new_dd["item_id"])
            self.assertEqual(3, new_dd["weight"])
            self.assertEqual("unnerve", new_dd["base_ability_id"])
            self.assertEqual("adamant", new_dd["nature_id"])
            self.assertEqual(
                ["dragondance", "earthquake", "stoneedge", "dualwingbeat"],
                new_dd["move_ids"],
            )
            old_dd["item_id"] = new_dd["item_id"]
            for field in ("rationale", "distinguishing_evidence"):
                old_dd["metadata"][field] = new_dd["metadata"][field]
            self.assertEqual(old_dd, new_dd)
            self.assertNotIn("expertbelt", json.dumps(new_dd).casefold())

    def test_03_all_28_human_authored_sets_are_exact_after_normalization(self):
        dataset = self.datasets["1.2.0"]
        actual_keys = {
            (species_id, variant_id)
            for species_id, variant_ids in NEW_VARIANT_IDS.items()
            for variant_id in variant_ids
        }
        self.assertEqual(set(EXPECTED_NEW_SETS), actual_keys)
        for key, expected in EXPECTED_NEW_SETS.items():
            species_id, variant_id = key
            variant = dataset.get_species(species_id).get_variant(variant_id)
            with self.subTest(species=species_id, variant=variant_id):
                self.assertIsNotNone(variant)
                self.assertEqual(float(expected["weight"]), variant.weight)
                self.assertTrue(math.isfinite(variant.weight))
                self.assertGreater(variant.weight, 0)
                self.assertEqual(expected["item"], variant.item_id)
                self.assertEqual(expected["ability"], variant.base_ability_id)
                self.assertEqual(expected["nature"], variant.nature_id)
                self.assertEqual(expected["evs"], variant.evs.as_tuple())
                self.assertEqual(expected["ivs"], variant.ivs.as_tuple())
                self.assertEqual(expected["moves"], variant.move_ids)
                self.assertEqual(100, variant.level)
                self.assertEqual(SOURCE_IDS, variant.source_ids)
                self.assertEqual(
                    {"role", "rationale", "distinguishing_evidence", "weight_reason"},
                    set(variant.metadata),
                )
                self.assertTrue(all(variant.metadata.values()))
                self.assertLessEqual(sum(variant.evs.as_tuple()), 510)
        source = VERSION_PATHS["1.2.0"].read_text(encoding="utf-8").casefold()
        for tera_token in ('"tera_type"', '"tera"', "terablast", "terastallization"):
            self.assertNotIn(tera_token, source)

    def test_04_real_loader_accepts_all_three_full_snapshots(self):
        expected = {
            "1.0.0": (8, 23),
            "1.1.0": (18, 52),
            "1.2.0": (28, 80),
        }
        for version, (species_count, variant_count) in expected.items():
            dataset = load_public_prior(VERSION_PATHS[version])
            with self.subTest(version=version):
                self.assertEqual("public", dataset.visibility)
                self.assertEqual("gen9tugs", dataset.identity.format_id)
                self.assertEqual(species_count, len(dataset.species))
                self.assertEqual(
                    variant_count,
                    sum(len(record.variants) for record in dataset.species),
                )

    def test_05_real_teamvalidator_accepts_all_required_groups(self):
        docs = self.documents
        old_species = {species["species_id"] for species in docs["1.1.0"]["species"]}
        groups = [
            {"label": "1.0.0", "entries": _entries(docs["1.0.0"])},
            {"label": "1.1.0", "entries": _entries(docs["1.1.0"])},
            {"label": "1.2.0", "entries": _entries(docs["1.2.0"])},
            {
                "label": "batch2-new",
                "entries": _entries(
                    docs["1.2.0"],
                    lambda species, variant: species["species_id"] not in old_species,
                ),
            },
            {
                "label": "aerodactyl-dragondance",
                "entries": _entries(
                    docs["1.2.0"],
                    lambda species, variant: (
                        species["species_id"] == "aerodactyl"
                        and variant["variant_id"] == "dragondance"
                    ),
                ),
            },
        ]
        completed = _run_teamvalidator(groups)
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertEqual(
            [
                "1.0.0:23/23",
                "1.1.0:52/52",
                "1.2.0:80/80",
                "batch2-new:28/28",
                "aerodactyl-dragondance:1/1",
            ],
            completed.stdout.splitlines(),
        )

    def test_06_authorization_firewall_and_negative_validators_are_exact(self):
        document = self.documents["1.2.0"]
        _assert_public_data_firewall(document)
        self.assertEqual(AUTHORIZATION_VALUE, document["metadata"]["authorization"])

        wrong_path = copy.deepcopy(document)
        wrong_path["metadata"]["note"] = AUTHORIZATION_VALUE
        altered = copy.deepcopy(document)
        altered["metadata"]["authorization"] += " Altered."
        private_phrase = copy.deepcopy(document)
        private_phrase["metadata"]["note"] = "Swiss participant material"
        forbidden_key = copy.deepcopy(document)
        forbidden_key["metadata"]["candidate_id"] = "x"
        trainer_value = copy.deepcopy(document)
        trainer_value["metadata"]["note"] = "trainer Alpha"
        team_key = copy.deepcopy(document)
        team_key["metadata"]["team_id"] = "alpha"
        for label, invalid in (
            ("wrong_path", wrong_path),
            ("altered", altered),
            ("private_phrase", private_phrase),
            ("forbidden_key", forbidden_key),
            ("trainer_value", trainer_value),
            ("team_key", team_key),
        ):
            with self.subTest(firewall=label), self.assertRaises(AssertionError):
                _assert_public_data_firewall(invalid)

        nonpublic = copy.deepcopy(document)
        nonpublic["visibility"] = "private"
        unsupported_source = copy.deepcopy(document)
        unsupported_source["sources"][0]["kind"] = "private"
        for label, invalid in (
            ("visibility", nonpublic),
            ("source_kind", unsupported_source),
        ):
            with self.subTest(schema=label), self.assertRaises(
                PublicPriorValidationError
            ):
                validate_public_prior_document(invalid)

        illegal = copy.deepcopy(document)
        altaria = next(
            species
            for species in illegal["species"]
            if species["species_id"] == "altaria"
        )
        altaria["variants"][0]["move_ids"][0] = "spectralthief"
        completed = _run_teamvalidator(
            [{"label": "illegal", "entries": _entries(illegal)}]
        )
        self.assertNotEqual(0, completed.returncode)
        self.assertIn("altaria/", completed.stderr)


class TestBatchTwoSelectionPopulationAndNarrowing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generic_configuration = _startup(PublicPriorFallback.GENERIC)
        cls.none_configuration = _startup(PublicPriorFallback.NONE)
        cls.dataset = cls.none_configuration.registry.get(IDENTITY)

    def test_07_every_new_species_and_variant_and_revision_are_selectable(self):
        context = self.none_configuration.create_battle_context("gen9tugs")
        for species_id in NEW_VARIANT_IDS:
            result = select_public_prior_variant(
                context,
                battle_format="gen9tugs",
                species_id=species_id,
                level=100,
                evidence=None,
                rng=_FixedRng(0.5),
            )
            self.assertIs(PublicPriorSelectionStatus.SELECTED, result.status)

        selected = set()
        targets = list(NEW_VARIANT_IDS)
        for species_id in targets:
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
        self.assertEqual(set(EXPECTED_NEW_SETS), selected)

        aero = self.dataset.get_species("aerodactyl")
        total = sum(variant.weight for variant in aero.variants)
        cumulative = 0.0
        for variant in aero.variants:
            if variant.variant_id == "dragondance":
                result = select_public_prior_variant(
                    context,
                    battle_format="gen9tugs",
                    species_id="aerodactyl",
                    level=100,
                    evidence=None,
                    rng=_FixedRng((cumulative + variant.weight / 2) / total),
                )
                self.assertIs(variant, result.variant)
                self.assertEqual("lifeorb", result.variant.item_id)
                break
            cumulative += variant.weight
        else:
            self.fail("missing revised Aerodactyl variant")

    def test_08_every_new_variant_and_revision_populates_only_copied_state(self):
        context = self.none_configuration.create_battle_context("gen9tugs")
        modules_before = set(sys.modules)
        populated = set()
        targets = list(EXPECTED_NEW_SETS) + [("aerodactyl", "dragondance")]
        for species_id, variant_id in targets:
            variant = self.dataset.get_species(species_id).get_variant(variant_id)
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
            with self.subTest(species=species_id, variant=variant_id):
                self.assertIsNot(sampled_pokemon, pokemon)
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
                if species_id == "aerodactyl":
                    self.assertEqual("lifeorb", sampled_pokemon.item)
            populated.add((species_id, variant_id))
        self.assertEqual(set(targets), populated)
        self.assertEqual(
            set(),
            {
                name
                for name in set(sys.modules) - modules_before
                if name.startswith("fp.data.team_pools")
            },
        )

    def test_09_observation_narrowing_covers_all_ten_species(self):
        context = self.none_configuration.create_battle_context("gen9tugs")

        def compatible(species_id, *, moves=(), item=None, ability=None):
            return _compatible_ids(
                self.dataset,
                species_id,
                _evidence(context, species_id, moves=moves, item=item, ability=ability),
            )

        checks = (
            ("altaria", {"perishtrap"}, {"moves": ("perishsong",)}),
            ("altaria", {"physicalutility"}, {"moves": ("bravebird",)}),
            ("altaria", {"choicespecs"}, {"moves": ("dracometeor",)}),
            ("arcaninehisui", {"intimidateutility"}, {"ability": "intimidate"}),
            (
                "arcaninehisui",
                {"bootsrocks", "intimidateutility"},
                {"moves": ("stealthrock",)},
            ),
            ("arcaninehisui", {"choiceband"}, {"moves": ("closecombat",)}),
            ("arcaninehisui", {"bootsrocks", "choiceband"}, {"ability": "rockhead"}),
            ("bombirdier", {"choicescarf"}, {"ability": "rockypayload"}),
            ("bombirdier", {"bootsutility"}, {"moves": ("memento",)}),
            ("bombirdier", {"choiceband"}, {"moves": ("suckerpunch",)}),
            ("bombirdier", {"choiceband"}, {"item": "choiceband"}),
            ("bombirdier", {"choicescarf"}, {"item": "choicescarf"}),
            ("chesnaught", {"helmetspikes"}, {"moves": ("spikes",)}),
            ("chesnaught", {"leftoversutility"}, {"moves": ("spikyshield",)}),
            ("cursola", {"weakarmorspecs"}, {"ability": "weakarmor"}),
            (
                "cursola",
                {"perishbodyspecs", "specialutility"},
                {"ability": "perishbody"},
            ),
            ("cursola", {"specialutility"}, {"moves": ("willowisp",)}),
            ("cursola", {"perishbodyspecs", "weakarmorspecs"}, {"item": "choicespecs"}),
            ("dhelmise", {"swordsdance"}, {"moves": ("swordsdance",)}),
            ("dhelmise", {"choiceband"}, {"moves": ("poltergeist",)}),
            ("drapion", {"swordsdance"}, {"moves": ("poisonjab",)}),
            ("drapion", {"toxicspikes"}, {"moves": ("toxicspikes",)}),
            ("dudunsparce", {"calmmind"}, {"moves": ("calmmind",)}),
            ("dudunsparce", {"glareutility"}, {"moves": ("glare",)}),
            ("dudunsparce", {"coil"}, {"ability": "serenegrace"}),
            ("dudunsparce", {"calmmind", "glareutility"}, {"ability": "thickfat"}),
            ("dudunsparcethreesegment", {"calmmind"}, {"moves": ("calmmind",)}),
            ("dudunsparcethreesegment", {"glareutility"}, {"moves": ("stealthrock",)}),
            ("dudunsparcethreesegment", {"coil"}, {"moves": ("coil",)}),
            (
                "dudunsparcethreesegment",
                {"calmmind", "glareutility"},
                {"ability": "thickfat"},
            ),
            ("dugtrioalola", {"sashsetup"}, {"moves": ("swordsdance",)}),
            ("dugtrioalola", {"sashlead"}, {"moves": ("memento",)}),
            ("dugtrioalola", {"choicescarf"}, {"moves": ("pursuit",)}),
            ("dugtrioalola", {"choiceband"}, {"item": "choiceband"}),
            ("dugtrioalola", {"choiceband", "sashsetup"}, {"moves": ("suckerpunch",)}),
            ("dugtrioalola", {"choicescarf"}, {"item": "choicescarf"}),
            ("dugtrioalola", {"sashlead", "sashsetup"}, {"item": "focussash"}),
        )
        for species_id, expected, kwargs in checks:
            with self.subTest(species=species_id, expected=expected, evidence=kwargs):
                self.assertEqual(expected, compatible(species_id, **kwargs))

        aero = compatible("aerodactyl", moves=("dragondance",))
        self.assertEqual({"dragondance"}, aero)
        self.assertEqual(
            "lifeorb",
            self.dataset.get_species("aerodactyl").get_variant("dragondance").item_id,
        )

    def test_10_exact_forms_conflicts_and_compatible_variants_are_isolated(self):
        context = self.none_configuration.create_battle_context("gen9tugs")
        base_evidence = _evidence(context, "dudunsparce", moves=("calmmind",))
        form_evidence = _evidence(
            context, "dudunsparcethreesegment", moves=("calmmind",)
        )
        self.assertEqual(
            set(),
            _compatible_ids(self.dataset, "dudunsparcethreesegment", base_evidence),
        )
        self.assertEqual(
            set(), _compatible_ids(self.dataset, "dudunsparce", form_evidence)
        )

        compatible_specs = _compatible_ids(
            self.dataset,
            "cursola",
            _evidence(context, "cursola", item="choicespecs"),
        )
        self.assertEqual({"perishbodyspecs", "weakarmorspecs"}, compatible_specs)

        conflict_battle = _battle(context, "dudunsparce")
        conflict_battle.team_inference.record_selected_move("dudunsparce", "calmmind")
        conflict_battle.team_inference.record_initial_item(
            "dudunsparce",
            "chopleberry",
            source=PublicObservationSource.DIRECT_ITEM_REVEAL,
        )
        result = select_public_prior_variant(
            context,
            battle_format="gen9tugs",
            species_id="dudunsparce",
            level=100,
            evidence=conflict_battle.team_inference.observation_ledger.member(
                "dudunsparce"
            ),
            rng=_FixedRng(0.5),
        )
        self.assertIs(PublicPriorSelectionStatus.NO_COMPATIBLE_VARIANT, result.status)

    def test_11_public_sampling_has_no_private_pool_or_candidate_dependency(self):
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
        parameters = inspect.signature(select_public_prior_variant).parameters
        self.assertNotIn("candidate_id", parameters)
        self.assertNotIn("candidate_ids", parameters)

    def test_12_still_uncovered_species_obeys_explicit_fallback_policy(self):
        generic_context = self.generic_configuration.create_battle_context("gen9tugs")
        generic_battle = _battle(generic_context, "xatu")
        with mock.patch("fp.search.standard_battles.sample_pokemon") as generic:
            sampled = prepare_battles(generic_battle, 1)[0][0]
        generic.assert_called_once()
        self.assertEqual("xatu", sampled.opponent.active.name)

        none_context = self.none_configuration.create_battle_context("gen9tugs")
        none_battle = _battle(none_context, "xatu")
        before = _pokemon_snapshot(none_battle.opponent.active)
        with mock.patch("fp.search.standard_battles.sample_pokemon") as generic:
            sampled = prepare_battles(none_battle, 1)[0][0]
        generic.assert_not_called()
        self.assertEqual(before, _pokemon_snapshot(sampled.opponent.active))
        self.assertEqual(constants.UNKNOWN_ITEM, sampled.opponent.active.item)


if __name__ == "__main__":
    unittest.main()
