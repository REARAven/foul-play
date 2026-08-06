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
    for version in ("1.0.0", "1.1.0", "1.2.0", "1.3.0")
}
RAW_SHA256 = {
    "1.0.0": "690c943f0ae554f68423f988f6a61d43f4310f25a2db8e65f98238d8c92c2a91",
    "1.1.0": "1a5a4d9bb3ad82e29eae907fcc2a41698aeaf1ec55fb8375ad18197e874662e4",
    "1.2.0": "1ebe4e9ad85ba30ab0c166f22cd8a8088a59d77e20f937162b9e528f0bb04b24",
    "1.3.0": "303597a07d62cfca763402b5bbaa3e300d134e93acdf0f08925482cb353d7497",
}
CANONICAL_SHA256 = {
    "1.0.0": "ddb04702321aace7d7368938016a18d54841935d2bf5df194cfaf7af8d4a55ae",
    "1.1.0": "59425bcd4371870ec6489d384ba9a7d6a3c3c5375371fbfc48c11fab17740d8d",
    "1.2.0": "bc0456562feaccf6fc4a64bf0d189deec8b5c6331c5c64d8bee606674d946376",
    "1.3.0": "53bcda29981d448b03151b47002a63e267cd180130241e445f8af123b810b226",
}
IDENTITY = PublicPriorIdentity("tugspublicarchetypes", "1.3.0", "gen9tugs")
SOURCE_IDS = ("publicformatpatch12", "publicmanualv1")
AUTHORIZATION_PATH = "$.metadata.authorization"
AUTHORIZATION_VALUE = (
    "Generic public design prior authored without closed-submission data."
)
NEW_SPECIES_VARIANTS = {
    "eelektross": ("assaultvest", "poweruppunch", "bulkyutility"),
    "flygon": (
        "choicescarf", "choiceband", "dragondance", "loadeddice",
        "mixedlifeorb",
    ),
    "froslass": ("sashlead", "bulkyspikes", "choicespecs"),
    "heracross": ("flameorbguts", "choicescarf", "bulkup"),
    "hitmontop": (
        "assaultvest", "fastboots", "technicianoffense", "technicianspdef",
    ),
    "houndoom": ("nastyplotlifeorb", "choicescarf", "nastyplotboots"),
}
REPLACEMENT_VARIANTS = {
    "claydol": ("regeneratorrocks", "regeneratortrickroom", "levitaterocks"),
    "druddigon": ("roughskinutility", "choiceband", "moldbreakerspdef"),
    "dustox": ("blacksludgeutility", "bootsutility"),
    "forretress": ("overcoathelmet", "sturdyboots"),
    "jellicent": (
        "bootsphysical", "colburphysical", "waterabsorbspdef",
        "cursedbodyspdef", "choicespecs",
    ),
}
TOUCHED_VARIANT_IDS = {**NEW_SPECIES_VARIANTS, **REPLACEMENT_VARIANTS}
REMOVED_VARIANT_IDS = {
    ("claydol", "levitateutility"),
    ("claydol", "regeneratorpivot"),
    ("claydol", "trickroomsetter"),
    ("druddigon", "sheerforcebreaker"),
    ("druddigon", "slowbandbreaker"),
    ("dustox", "levitateutility"),
    ("dustox", "shielddustphazer"),
    ("forretress", "hazardpivot"),
    ("forretress", "sturdyrocks"),
    ("forretress", "toxicspikesboom"),
    ("jellicent", "cursedbodyutility"),
    ("jellicent", "defensivespinblocker"),
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


def _expected(weight, confidence, item, ability, nature, evs, ivs, moves, role):
    return {
        "weight": weight,
        "confidence": confidence,
        "item": item,
        "ability": ability,
        "nature": nature,
        "evs": evs,
        "ivs": ivs,
        "moves": moves,
        "role": role,
    }


EXPECTED_TOUCHED_SETS = {
    ("eelektross", "assaultvest"): _expected(
        4, "established", "assaultvest", "levitate", "brave",
        (252, 252, 0, 4, 0, 0), IVS_31,
        ("voltswitch", "knockoff", "closecombat", "flamethrower"),
        "slow mixed Assault Vest pivot",
    ),
    ("eelektross", "poweruppunch"): _expected(
        2, "experimental", "leftovers", "levitate", "adamant",
        (252, 252, 0, 0, 4, 0), IVS_31,
        ("substitute", "poweruppunch", "knockoff", "supercellslam"),
        "Substitute-based physical setup attacker",
    ),
    ("eelektross", "bulkyutility"): _expected(
        3, "plausible", "heavydutyboots", "levitate", "careful",
        (252, 4, 0, 0, 252, 0), IVS_31,
        ("knockoff", "voltswitch", "toxic", "dragontail"),
        "specially defensive utility pivot",
    ),
    ("flygon", "choicescarf"): _expected(
        4, "established", "choicescarf", "levitate", "jolly",
        (0, 252, 0, 0, 4, 252), IVS_31,
        ("earthquake", "outrage", "uturn", "stoneedge"),
        "fast revenge killer and pivot",
    ),
    ("flygon", "choiceband"): _expected(
        3, "plausible", "choiceband", "levitate", "jolly",
        (0, 252, 0, 0, 4, 252), IVS_31,
        ("earthquake", "scaleshot", "firstimpression", "uturn"),
        "immediate physical wallbreaker with priority",
    ),
    ("flygon", "dragondance"): _expected(
        3, "plausible", "clearamulet", "levitate", "jolly",
        (0, 252, 0, 0, 4, 252), IVS_31,
        ("dragondance", "earthquake", "throatchop", "scaleshot"),
        "physical setup attacker resistant to stat reduction",
    ),
    ("flygon", "loadeddice"): _expected(
        2, "experimental", "loadeddice", "levitate", "jolly",
        (0, 252, 4, 0, 0, 252), IVS_31,
        ("stealthrock", "earthquake", "scaleshot", "uturn"),
        "Loaded Dice Scale Shot utility attacker",
    ),
    ("flygon", "mixedlifeorb"): _expected(
        2, "experimental", "lifeorb", "levitate", "naive",
        (0, 88, 0, 168, 0, 252), IVS_31,
        ("earthquake", "dracometeor", "fireblast", "stealthrock"),
        "mixed offensive hazard setter",
    ),
    ("froslass", "sashlead"): _expected(
        4, "established", "focussash", "cursedbody", "timid",
        (0, 0, 0, 252, 4, 252), IVS_0_ATK,
        ("spikes", "taunt", "destinybond", "icywind"),
        "fast Focus Sash hazard and disruption lead",
    ),
    ("froslass", "bulkyspikes"): _expected(
        3, "plausible", "heavydutyboots", "cursedbody", "timid",
        (252, 0, 200, 0, 0, 56), IVS_0_ATK,
        ("hex", "willowisp", "painsplit", "spikes"),
        "bulky status and hazard utility",
    ),
    ("froslass", "choicespecs"): _expected(
        2, "experimental", "choicespecs", "cursedbody", "timid",
        (4, 0, 0, 252, 0, 252), IVS_0_ATK,
        ("icebeam", "shadowball", "trick", "thunderbolt"),
        "fast special wallbreaker with Trick",
    ),
    ("heracross", "flameorbguts"): _expected(
        4, "established", "flameorb", "guts", "jolly",
        (0, 252, 0, 0, 4, 252), IVS_31,
        ("closecombat", "facade", "knockoff", "trailblaze"),
        "Guts physical wallbreaker with Speed boosting",
    ),
    ("heracross", "choicescarf"): _expected(
        3, "plausible", "choicescarf", "moxie", "jolly",
        (0, 252, 0, 0, 4, 252), IVS_31,
        ("closecombat", "megahorn", "knockoff", "stoneedge"),
        "Choice Scarf revenge killer and potential cleaner",
    ),
    ("heracross", "bulkup"): _expected(
        2, "plausible", "leftovers", "guts", "jolly",
        (0, 252, 4, 0, 0, 252), IVS_31,
        ("bulkup", "closecombat", "knockoff", "earthquake"),
        "Leftovers physical setup attacker",
    ),
    ("hitmontop", "assaultvest"): _expected(
        4, "established", "assaultvest", "intimidate", "adamant",
        (248, 252, 0, 0, 8, 0), IVS_31,
        ("closecombat", "machpunch", "rapidspin", "tripleaxel"),
        "bulky Assault Vest spinner with Intimidate",
    ),
    ("hitmontop", "fastboots"): _expected(
        3, "plausible", "heavydutyboots", "intimidate", "adamant",
        (0, 252, 0, 0, 4, 252), IVS_31,
        ("closecombat", "machpunch", "rapidspin", "tripleaxel"),
        "faster offensive spinner with Intimidate",
    ),
    ("hitmontop", "technicianoffense"): _expected(
        3, "plausible", "heavydutyboots", "technician", "adamant",
        (248, 252, 0, 0, 8, 0), IVS_31,
        ("closecombat", "bulletpunch", "rapidspin", "tripleaxel"),
        "bulky Technician attacker and spinner",
    ),
    ("hitmontop", "technicianspdef"): _expected(
        2, "experimental", "heavydutyboots", "technician", "careful",
        (248, 8, 0, 0, 252, 0), IVS_31,
        ("toxic", "machpunch", "rapidspin", "tripleaxel"),
        "specially defensive Technician utility spinner",
    ),
    ("houndoom", "nastyplotlifeorb"): _expected(
        4, "established", "lifeorb", "flashfire", "timid",
        (0, 0, 0, 252, 4, 252), IVS_0_ATK,
        ("nastyplot", "darkpulse", "flamethrower", "hiddenpowergrass60"),
        "Life Orb Nasty Plot wallbreaker",
    ),
    ("houndoom", "choicescarf"): _expected(
        2, "experimental", "choicescarf", "flashfire", "timid",
        (0, 0, 0, 252, 4, 252), IVS_0_ATK,
        ("overheat", "flamethrower", "darkpulse", "sludgebomb"),
        "Choice Scarf special revenge killer",
    ),
    ("houndoom", "nastyplotboots"): _expected(
        3, "plausible", "heavydutyboots", "flashfire", "timid",
        (0, 0, 0, 252, 4, 252), IVS_0_ATK,
        ("nastyplot", "fireblast", "darkpulse", "substitute"),
        "Heavy-Duty Boots Nasty Plot attacker",
    ),
    ("claydol", "regeneratorrocks"): _expected(
        4, "established", "heavydutyboots", "regenerator", "calm",
        (252, 0, 0, 4, 252, 0), IVS_31,
        ("stealthrock", "teleport", "rapidspin", "futuresight"),
        "specially defensive Regenerator hazard, removal, and pivot utility",
    ),
    ("claydol", "regeneratortrickroom"): _expected(
        3, "plausible", "heavydutyboots", "regenerator", "sassy",
        (252, 0, 0, 4, 252, 0), IVS_0_ATK,
        ("teleport", "toxic", "trickroom", "futuresight"),
        "slow Regenerator Trick Room and pivot support",
    ),
    ("claydol", "levitaterocks"): _expected(
        3, "plausible", "heavydutyboots", "levitate", "calm",
        (252, 0, 0, 4, 252, 0), IVS_31,
        ("stealthrock", "teleport", "rapidspin", "futuresight"),
        "specially defensive Levitate hazard, removal, and pivot utility",
    ),
    ("druddigon", "roughskinutility"): _expected(
        4, "established", "rockyhelmet", "roughskin", "impish",
        (252, 0, 252, 0, 4, 0), IVS_31,
        ("dragontail", "cragmend", "stealthrock", "glare"),
        "physical contact punishment, hazards, paralysis, and phazing",
    ),
    ("druddigon", "choiceband"): _expected(
        3, "plausible", "choiceband", "roughskin", "adamant",
        (252, 252, 0, 0, 4, 0), IVS_31,
        ("suckerpunch", "dragonclaw", "superpower", "gunkshot"),
        "bulky Choice Band physical wallbreaker",
    ),
    ("druddigon", "moldbreakerspdef"): _expected(
        2, "experimental", "rockyhelmet", "moldbreaker", "careful",
        (252, 4, 0, 0, 252, 0), IVS_31,
        ("earthquake", "stealthrock", "cragmend", "dragontail"),
        "specially defensive Mold Breaker utility and phazing",
    ),
    ("dustox", "blacksludgeutility"): _expected(
        4, "established", "blacksludge", "levitate", "careful",
        (252, 0, 4, 0, 252, 0), IVS_31,
        ("uturn", "defog", "roost", "corrosivegas"),
        "specially defensive item-removal and Defog utility",
    ),
    ("dustox", "bootsutility"): _expected(
        3, "plausible", "heavydutyboots", "levitate", "calm",
        (248, 0, 0, 8, 252, 0), IVS_31,
        ("defog", "sludgebomb", "uturn", "roost"),
        "specially defensive Boots removal and pivot utility",
    ),
    ("forretress", "overcoathelmet"): _expected(
        4, "established", "rockyhelmet", "overcoat", "bold",
        (252, 0, 252, 0, 4, 0), IVS_31,
        ("bodypress", "stealthrock", "voltswitch", "rapidspin"),
        "physical wall, hazard setter, spinner, and contact punishment",
    ),
    ("forretress", "sturdyboots"): _expected(
        3, "plausible", "heavydutyboots", "sturdy", "bold",
        (252, 0, 252, 0, 4, 0), IVS_31,
        ("spikes", "stealthrock", "voltswitch", "rapidspin"),
        "Sturdy dual-hazard and removal utility",
    ),
    ("jellicent", "bootsphysical"): _expected(
        4, "established", "heavydutyboots", "waterabsorb", "bold",
        (252, 0, 252, 4, 0, 0), IVS_0_ATK,
        ("scald", "shadowball", "taunt", "strengthsap"),
        "physically defensive Boots spinblocker and disruption utility",
    ),
    ("jellicent", "colburphysical"): _expected(
        3, "plausible", "colburberry", "waterabsorb", "bold",
        (252, 0, 252, 4, 0, 0), IVS_0_ATK,
        ("scald", "shadowball", "taunt", "strengthsap"),
        "physically defensive spinblocker using Colbur Berry for Dark attacks",
    ),
    ("jellicent", "waterabsorbspdef"): _expected(
        4, "established", "leftovers", "waterabsorb", "calm",
        (252, 0, 0, 4, 252, 0), IVS_0_ATK,
        ("hex", "willowisp", "recover", "surf"),
        "specially defensive Water Absorb status utility",
    ),
    ("jellicent", "cursedbodyspdef"): _expected(
        3, "plausible", "leftovers", "cursedbody", "calm",
        (252, 0, 0, 4, 252, 0), IVS_0_ATK,
        ("hex", "willowisp", "recover", "surf"),
        "specially defensive Cursed Body status utility",
    ),
    ("jellicent", "choicespecs"): _expected(
        2, "experimental", "choicespecs", "waterabsorb", "modest",
        (252, 0, 0, 252, 4, 0), IVS_0_ATK,
        ("trick", "icebeam", "hydropump", "shadowball"),
        "bulky Choice Specs special wallbreaker",
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
                    raise AssertionError(
                        f"forbidden structural key {key!r} at {path}"
                    )
                check_string(key, f"{path}.<key:{key}>", is_key=True)
                walk(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        elif isinstance(value, str):
            check_string(value, path)

    walk(document, "$")


class _ForbiddenGenericSource:
    def __getattr__(self, name):
        raise AssertionError(f"generic source queried through {name}")


class _Mode:
    smogon_sets = _ForbiddenGenericSource()
    team_datasets = _ForbiddenGenericSource()

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


def _startup(version, fallback):
    options = PublicPriorStartupOptions((str(VERSION_PATHS[version]),), fallback)
    return load_public_prior_runtime_configuration(options, "gen9tugs")


def _battle(context, species_id):
    battle = Battle("batch-three-public-prior", public_prior_context=context)
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
        pokemon.name,
        pokemon.item,
        pokemon.ability,
        pokemon.original_ability,
        tuple(move.name for move in pokemon.moves),
        pokemon.nature,
        tuple(pokemon.evs),
        tuple(pokemon.ivs),
        pokemon.level,
    )


def _run_teamvalidator(groups):
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
        input=json.dumps(groups),
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


class TestBatchThreeDocument(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents = {version: _document(version) for version in VERSION_PATHS}
        cls.datasets = {
            version: load_public_prior(path)
            for version, path in VERSION_PATHS.items()
        }

    def test_01_encoding_hash_identity_and_inventories_are_exact(self):
        expected = {
            "1.0.0": (8, 23),
            "1.1.0": (18, 52),
            "1.2.0": (28, 80),
            "1.3.0": (34, 102),
        }
        for version, (species_count, variant_count) in expected.items():
            raw = VERSION_PATHS[version].read_bytes()
            with self.subTest(version=version):
                self.assertIsInstance(raw.decode("utf-8", errors="strict"), str)
                self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
                self.assertNotIn(b"\r", raw)
                self.assertTrue(raw.endswith(b"\n"))
                self.assertFalse(
                    any(line.rstrip(b" \t") != line for line in raw.splitlines())
                )
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
                self.assertEqual(species_count, len(dataset.species))
                self.assertEqual(
                    variant_count,
                    sum(len(record.variants) for record in dataset.species),
                )
        self.assertEqual(137237, len(VERSION_PATHS["1.3.0"].read_bytes()))
        self.assertEqual(IDENTITY, self.datasets["1.3.0"].identity)
        self.assertEqual("1.2", self.datasets["1.3.0"].patch_version)
        self.assertEqual(6, len(NEW_SPECIES_VARIANTS))
        self.assertEqual(21, sum(map(len, NEW_SPECIES_VARIANTS.values())))
        self.assertEqual(5, len(REPLACEMENT_VARIANTS))
        self.assertEqual(15, sum(map(len, REPLACEMENT_VARIANTS.values())))

    def test_02_version_1_2_semantics_change_only_as_authorized(self):
        old = self.documents["1.2.0"]
        new = self.documents["1.3.0"]
        for key in (
            "schema_version", "visibility", "dataset_id", "format_id",
            "patch_version", "metadata", "sources",
        ):
            self.assertEqual(old[key], new[key], key)
        self.assertEqual("1.3.0", new["dataset_version"])
        self.assertEqual("Public TUGS archetypes 1.3.0", new["display_name"])

        old_records = {record["species_id"]: record for record in old["species"]}
        new_records = {record["species_id"]: record for record in new["species"]}
        old_ids = set(old_records)
        new_ids = set(new_records)
        self.assertEqual(set(NEW_SPECIES_VARIANTS), new_ids - old_ids)
        self.assertEqual(set(), old_ids - new_ids)
        changed = {
            species_id
            for species_id in old_ids
            if old_records[species_id] != new_records[species_id]
        }
        self.assertEqual(set(REPLACEMENT_VARIANTS), changed)
        unchanged = old_ids - changed
        self.assertEqual(23, len(unchanged))
        for species_id in unchanged:
            self.assertEqual(old_records[species_id], new_records[species_id])
        self.assertEqual(old_records["aerodactyl"], new_records["aerodactyl"])
        self.assertEqual(
            "lifeorb",
            next(
                variant for variant in new_records["aerodactyl"]["variants"]
                if variant["variant_id"] == "dragondance"
            )["item_id"],
        )

    def test_03_all_36_human_authored_sets_are_exact_after_normalization(self):
        dataset = self.datasets["1.3.0"]
        actual_keys = {
            (species_id, variant_id)
            for species_id, variant_ids in TOUCHED_VARIANT_IDS.items()
            for variant_id in variant_ids
        }
        self.assertEqual(36, len(actual_keys))
        self.assertEqual(set(EXPECTED_TOUCHED_SETS), actual_keys)
        for key, expected in EXPECTED_TOUCHED_SETS.items():
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
                    {
                        "role", "rationale", "distinguishing_evidence",
                        "weight_reason", "confidence",
                    },
                    set(variant.metadata),
                )
                self.assertEqual(expected["role"], variant.metadata["role"])
                self.assertEqual(
                    expected["confidence"], variant.metadata["confidence"]
                )
                self.assertIn(
                    variant.metadata["confidence"],
                    {"established", "plausible", "experimental"},
                )
                self.assertTrue(all(variant.metadata.values()))
                self.assertLessEqual(sum(variant.evs.as_tuple()), 510)
        source = VERSION_PATHS["1.3.0"].read_text(encoding="utf-8").casefold()
        for tera_token in ('"tera_type"', '"tera"', "terablast", "terastallization"):
            self.assertNotIn(tera_token, source)

    def test_04_real_loader_accepts_all_four_full_snapshots(self):
        expected = {
            "1.0.0": (8, 23),
            "1.1.0": (18, 52),
            "1.2.0": (28, 80),
            "1.3.0": (34, 102),
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
        unusual = {
            ("houndoom", "nastyplotlifeorb"),
            ("eelektross", "poweruppunch"),
            ("jellicent", "bootsphysical"),
            ("claydol", "regeneratorrocks"),
            ("flygon", "choiceband"),
            ("flygon", "loadeddice"),
            ("dustox", "blacksludgeutility"),
            ("druddigon", "roughskinutility"),
        }
        groups = [
            {"label": version, "entries": _entries(docs[version])}
            for version in VERSION_PATHS
        ]
        groups.extend(
            [
                {
                    "label": "batch3-new",
                    "entries": _entries(
                        docs["1.3.0"],
                        lambda species, variant: (
                            species["species_id"] in NEW_SPECIES_VARIANTS
                        ),
                    ),
                },
                {
                    "label": "batch3-replacements",
                    "entries": _entries(
                        docs["1.3.0"],
                        lambda species, variant: (
                            species["species_id"] in REPLACEMENT_VARIANTS
                        ),
                    ),
                },
                {
                    "label": "unusual-elements",
                    "entries": _entries(
                        docs["1.3.0"],
                        lambda species, variant: (
                            species["species_id"], variant["variant_id"]
                        ) in unusual,
                    ),
                },
            ]
        )
        completed = _run_teamvalidator(groups)
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertEqual(
            [
                "1.0.0:23/23",
                "1.1.0:52/52",
                "1.2.0:80/80",
                "1.3.0:102/102",
                "batch3-new:21/21",
                "batch3-replacements:15/15",
                "unusual-elements:8/8",
            ],
            completed.stdout.splitlines(),
        )

    def test_06_authorization_firewall_and_negative_validators_are_exact(self):
        document = self.documents["1.3.0"]
        _assert_public_data_firewall(document)
        self.assertEqual(
            AUTHORIZATION_VALUE, document["metadata"]["authorization"]
        )
        self.assertEqual(1, json.dumps(document).count(AUTHORIZATION_VALUE))

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
        eelektross = next(
            species for species in illegal["species"]
            if species["species_id"] == "eelektross"
        )
        eelektross["variants"][0]["move_ids"][0] = "spectralthief"
        completed = _run_teamvalidator(
            [{"label": "illegal", "entries": _entries(illegal)}]
        )
        self.assertNotEqual(0, completed.returncode)
        self.assertIn("eelektross/", completed.stderr)


class TestBatchThreeSelectionPopulationAndNarrowing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generic_configuration = _startup(
            "1.3.0", PublicPriorFallback.GENERIC
        )
        cls.none_configuration = _startup("1.3.0", PublicPriorFallback.NONE)
        cls.old_configuration = _startup("1.2.0", PublicPriorFallback.NONE)
        cls.dataset = cls.none_configuration.registry.get(IDENTITY)
        cls.old_dataset = cls.old_configuration.registry.get(
            PublicPriorIdentity("tugspublicarchetypes", "1.2.0", "gen9tugs")
        )

    def test_07_every_new_species_and_touched_variant_is_selectable(self):
        context = self.none_configuration.create_battle_context("gen9tugs")
        for species_id in NEW_SPECIES_VARIANTS:
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
        for species_id in TOUCHED_VARIANT_IDS:
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
        self.assertEqual(set(EXPECTED_TOUCHED_SETS), selected)

    def test_08_all_23_unchanged_records_select_identically(self):
        old_context = self.old_configuration.create_battle_context("gen9tugs")
        new_context = self.none_configuration.create_battle_context("gen9tugs")
        old_ids = {record.species_id for record in self.old_dataset.species}
        unchanged = old_ids - set(REPLACEMENT_VARIANTS)
        self.assertEqual(23, len(unchanged))
        for species_id in unchanged:
            for rng_value in (0.0, 0.25, 0.5, 0.75, 0.999999):
                old = select_public_prior_variant(
                    old_context,
                    battle_format="gen9tugs",
                    species_id=species_id,
                    level=100,
                    evidence=None,
                    rng=_FixedRng(rng_value),
                )
                new = select_public_prior_variant(
                    new_context,
                    battle_format="gen9tugs",
                    species_id=species_id,
                    level=100,
                    evidence=None,
                    rng=_FixedRng(rng_value),
                )
                with self.subTest(species=species_id, rng=rng_value):
                    self.assertIs(old.status, new.status)
                    self.assertEqual(old.variant, new.variant)

    def test_09_every_touched_variant_populates_only_copied_state(self):
        context = self.none_configuration.create_battle_context("gen9tugs")
        modules_before = set(sys.modules)
        populated = set()
        for species_id, variant_id in EXPECTED_TOUCHED_SETS:
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
                self.assertEqual(species_id, sampled_pokemon.name)
                self.assertEqual(variant.item_id, sampled_pokemon.item)
                self.assertEqual(variant.base_ability_id, sampled_pokemon.ability)
                self.assertEqual(
                    variant.base_ability_id, sampled_pokemon.original_ability
                )
                self.assertEqual(
                    variant.move_ids,
                    tuple(move.name for move in sampled_pokemon.moves),
                )
                self.assertIn(
                    variant.move_ids[0],
                    tuple(move.name for move in sampled_pokemon.moves),
                )
                self.assertEqual(variant.nature_id, sampled_pokemon.nature)
                self.assertEqual(variant.evs.as_tuple(), tuple(sampled_pokemon.evs))
                self.assertEqual(variant.ivs.as_tuple(), tuple(sampled_pokemon.ivs))
                self.assertEqual(canonical_before, _pokemon_snapshot(pokemon))
            populated.add((species_id, variant_id))
        self.assertEqual(set(EXPECTED_TOUCHED_SETS), populated)
        self.assertEqual(
            set(),
            {
                name
                for name in set(sys.modules) - modules_before
                if name.startswith("fp.data.team_pools")
            },
        )

    def test_10_observation_narrowing_covers_all_eleven_touched_species(self):
        context = self.none_configuration.create_battle_context("gen9tugs")

        def compatible(species_id, *, moves=(), item=None, ability=None):
            return _compatible_ids(
                self.dataset,
                species_id,
                _evidence(
                    context, species_id, moves=moves, item=item, ability=ability
                ),
            )

        checks = (
            ("eelektross", {"assaultvest"}, {"item": "assaultvest"}),
            ("eelektross", {"assaultvest"}, {"moves": ("flamethrower",)}),
            ("eelektross", {"poweruppunch"}, {"moves": ("poweruppunch",)}),
            ("eelektross", {"poweruppunch"}, {"moves": ("substitute",)}),
            ("eelektross", {"bulkyutility"}, {"moves": ("toxic",)}),
            ("eelektross", {"bulkyutility"}, {"moves": ("dragontail",)}),
            ("flygon", {"choicescarf"}, {"item": "choicescarf"}),
            ("flygon", {"choicescarf"}, {"moves": ("outrage",)}),
            ("flygon", {"choiceband"}, {"item": "choiceband"}),
            ("flygon", {"choiceband"}, {"moves": ("firstimpression",)}),
            ("flygon", {"dragondance"}, {"item": "clearamulet"}),
            ("flygon", {"dragondance"}, {"moves": ("dragondance",)}),
            ("flygon", {"loadeddice"}, {"item": "loadeddice"}),
            ("flygon", {"mixedlifeorb"}, {"moves": ("dracometeor",)}),
            ("flygon", {"mixedlifeorb"}, {"moves": ("fireblast",)}),
            ("flygon", {"loadeddice", "mixedlifeorb"}, {"moves": ("stealthrock",)}),
            ("froslass", {"sashlead"}, {"item": "focussash"}),
            ("froslass", {"sashlead"}, {"moves": ("destinybond",)}),
            ("froslass", {"bulkyspikes"}, {"moves": ("willowisp",)}),
            ("froslass", {"bulkyspikes"}, {"moves": ("painsplit",)}),
            ("froslass", {"choicespecs"}, {"item": "choicespecs"}),
            ("froslass", {"choicespecs"}, {"moves": ("thunderbolt",)}),
            ("heracross", {"flameorbguts"}, {"item": "flameorb"}),
            ("heracross", {"flameorbguts"}, {"moves": ("facade",)}),
            ("heracross", {"flameorbguts", "bulkup"}, {"ability": "guts"}),
            ("heracross", {"choicescarf"}, {"ability": "moxie"}),
            ("heracross", {"bulkup"}, {"moves": ("bulkup",)}),
            ("heracross", {"bulkup"}, {"item": "leftovers"}),
            ("hitmontop", {"assaultvest"}, {"item": "assaultvest"}),
            ("hitmontop", {"fastboots"}, {"item": "heavydutyboots", "ability": "intimidate"}),
            ("hitmontop", {"technicianoffense"}, {"moves": ("bulletpunch",), "ability": "technician"}),
            ("hitmontop", {"technicianspdef"}, {"moves": ("toxic",), "ability": "technician"}),
            ("hitmontop", {"assaultvest", "fastboots"}, {"ability": "intimidate"}),
            ("hitmontop", {"technicianoffense", "technicianspdef"}, {"ability": "technician"}),
            ("houndoom", {"nastyplotlifeorb"}, {"item": "lifeorb"}),
            ("houndoom", {"nastyplotlifeorb"}, {"moves": ("hiddenpowergrass60",)}),
            ("houndoom", {"choicescarf"}, {"item": "choicescarf"}),
            ("houndoom", {"choicescarf"}, {"moves": ("sludgebomb",)}),
            ("houndoom", {"nastyplotboots"}, {"item": "heavydutyboots"}),
            ("houndoom", {"nastyplotboots"}, {"moves": ("substitute",)}),
            ("houndoom", {"nastyplotlifeorb", "nastyplotboots"}, {"moves": ("nastyplot",)}),
            ("claydol", {"regeneratorrocks", "regeneratortrickroom"}, {"ability": "regenerator"}),
            ("claydol", {"levitaterocks"}, {"ability": "levitate"}),
            ("claydol", {"regeneratortrickroom"}, {"moves": ("trickroom",)}),
            ("claydol", {"regeneratortrickroom"}, {"moves": ("toxic",)}),
            ("claydol", {"regeneratorrocks", "levitaterocks"}, {"moves": ("rapidspin",)}),
            ("claydol", {"regeneratorrocks", "levitaterocks"}, {"moves": ("stealthrock",)}),
            ("druddigon", {"roughskinutility"}, {"moves": ("glare",), "ability": "roughskin"}),
            ("druddigon", {"choiceband"}, {"item": "choiceband"}),
            ("druddigon", {"choiceband"}, {"moves": ("gunkshot",)}),
            ("druddigon", {"moldbreakerspdef"}, {"ability": "moldbreaker"}),
            ("druddigon", {"roughskinutility", "moldbreakerspdef"}, {"moves": ("cragmend",)}),
            ("dustox", {"blacksludgeutility"}, {"item": "blacksludge"}),
            ("dustox", {"blacksludgeutility"}, {"moves": ("corrosivegas",)}),
            ("dustox", {"bootsutility"}, {"item": "heavydutyboots"}),
            ("dustox", {"bootsutility"}, {"moves": ("sludgebomb",)}),
            ("forretress", {"overcoathelmet"}, {"ability": "overcoat"}),
            ("forretress", {"overcoathelmet"}, {"item": "rockyhelmet"}),
            ("forretress", {"sturdyboots"}, {"ability": "sturdy"}),
            ("forretress", {"sturdyboots"}, {"moves": ("spikes",)}),
            ("jellicent", {"bootsphysical"}, {"item": "heavydutyboots"}),
            ("jellicent", {"colburphysical"}, {"item": "colburberry"}),
            ("jellicent", {"waterabsorbspdef"}, {"item": "leftovers", "ability": "waterabsorb"}),
            ("jellicent", {"cursedbodyspdef"}, {"ability": "cursedbody"}),
            ("jellicent", {"choicespecs"}, {"item": "choicespecs"}),
            ("jellicent", {"choicespecs"}, {"moves": ("icebeam",)}),
            ("jellicent", {"bootsphysical", "colburphysical"}, {"moves": ("strengthsap",)}),
            ("jellicent", {"waterabsorbspdef", "cursedbodyspdef"}, {"moves": ("hex", "willowisp", "recover", "surf")}),
        )
        for species_id, expected, kwargs in checks:
            with self.subTest(species=species_id, expected=expected, evidence=kwargs):
                self.assertEqual(expected, compatible(species_id, **kwargs))

    def test_11_conflicts_ambiguity_and_removed_variants_are_isolated(self):
        context = self.none_configuration.create_battle_context("gen9tugs")
        conflict = _evidence(
            context,
            "flygon",
            moves=("dragondance",),
            item="choiceband",
        )
        result = select_public_prior_variant(
            context,
            battle_format="gen9tugs",
            species_id="flygon",
            level=100,
            evidence=conflict,
            rng=_FixedRng(0.5),
        )
        self.assertIs(
            PublicPriorSelectionStatus.NO_COMPATIBLE_VARIANT, result.status
        )

        compatible = _compatible_ids(
            self.dataset,
            "jellicent",
            _evidence(
                context,
                "jellicent",
                moves=("hex", "willowisp", "recover", "surf"),
            ),
        )
        self.assertEqual(
            {"waterabsorbspdef", "cursedbodyspdef"}, compatible
        )

        for species_id, variant_id in REMOVED_VARIANT_IDS:
            record = self.dataset.get_species(species_id)
            with self.subTest(species=species_id, variant=variant_id):
                self.assertIsNone(record.get_variant(variant_id))
                self.assertNotIn(
                    variant_id,
                    TOUCHED_VARIANT_IDS[species_id],
                )
        self.assertEqual(
            {
                species_id: set(variant_ids)
                for species_id, variant_ids in REPLACEMENT_VARIANTS.items()
            },
            {
                species_id: {
                    variant.variant_id
                    for variant in self.dataset.get_species(species_id).variants
                }
                for species_id in REPLACEMENT_VARIANTS
            },
        )

    def test_12_public_sampling_has_no_private_pool_or_candidate_dependency(self):
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

    def test_13_still_uncovered_species_obeys_explicit_fallback_policy(self):
        generic_context = self.generic_configuration.create_battle_context(
            "gen9tugs"
        )
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
        json.dumps(sampled.request_json or {})


if __name__ == "__main__":
    unittest.main()
