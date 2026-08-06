import copy
import hashlib
import inspect
import json
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

from fp import constants
from fp.battle.public_prior_context import PublicPriorFallback
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
    select_public_prior_variant,
)
from fp.search.standard_battles import prepare_battles
from test_public_prior_production_batch3_data import (
    AUTHORIZATION_PATH,
    AUTHORIZATION_VALUE,
    _FixedRng,
    _assert_public_data_firewall,
    _battle,
    _compatible_ids,
    _entries,
    _evidence,
    _pokemon_snapshot,
    _record_complete_public_evidence,
    _run_teamvalidator,
)


ROOT = Path(__file__).resolve().parents[1]
POOLS = ROOT / "fp" / "data" / "public_priors" / "pools"
VERSIONS = ("1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.4.0")
VERSION_PATHS = {
    version: POOLS / f"tugspublicarchetypes-{version}.json"
    for version in VERSIONS
}
RAW_SHA256 = {
    "1.0.0": "690c943f0ae554f68423f988f6a61d43f4310f25a2db8e65f98238d8c92c2a91",
    "1.1.0": "1a5a4d9bb3ad82e29eae907fcc2a41698aeaf1ec55fb8375ad18197e874662e4",
    "1.2.0": "1ebe4e9ad85ba30ab0c166f22cd8a8088a59d77e20f937162b9e528f0bb04b24",
    "1.3.0": "303597a07d62cfca763402b5bbaa3e300d134e93acdf0f08925482cb353d7497",
    "1.4.0": "88532c06bd1fe64dd165e78730e34bedb2fb372f9129e29b1fadda2db4ceeb34",
}
CANONICAL_SHA256 = {
    "1.0.0": "ddb04702321aace7d7368938016a18d54841935d2bf5df194cfaf7af8d4a55ae",
    "1.1.0": "59425bcd4371870ec6489d384ba9a7d6a3c3c5375371fbfc48c11fab17740d8d",
    "1.2.0": "bc0456562feaccf6fc4a64bf0d189deec8b5c6331c5c64d8bee606674d946376",
    "1.3.0": "53bcda29981d448b03151b47002a63e267cd180130241e445f8af123b810b226",
    "1.4.0": "97dd627a9b256eb2abc61ce61a4d8b9a3972dc89e0591c26887d29ced4a91434",
}
FILE_SIZES = {
    "1.0.0": 25644,
    "1.1.0": 69667,
    "1.2.0": 107311,
    "1.3.0": 137237,
    "1.4.0": 177609,
}
COUNTS = {
    "1.0.0": (8, 23),
    "1.1.0": (18, 52),
    "1.2.0": (28, 80),
    "1.3.0": (34, 102),
    "1.4.0": (42, 132),
}
IDENTITY = PublicPriorIdentity("tugspublicarchetypes", "1.4.0", "gen9tugs")
SOURCE_IDS = ("publicformatpatch12", "publicmanualv1")

NEW_SPECIES_VARIANTS = {
    "kabutops": ("sashlead", "bootsspinner", "sashsetup"),
    "kangaskhan": (
        "choiceband", "bootsoffense", "earlybirdtrapper", "substitute",
    ),
    "leavanny": ("sashwebs", "ejectpackwebs", "swarmwebs", "sashsetup"),
    "lilligant": ("sashquiverdance", "lifeorbquiverdance"),
    "mismagius": (
        "substitutesetup", "lifeorbsetup", "choicescarf", "choicespecs",
    ),
    "mukalola": ("resttalk", "assaultvest", "choiceband"),
    "raichu": ("choiceband", "sashnastyplot", "bootsutility"),
    "reuniclus": (
        "magicguardspdef", "regeneratortrickroom", "magicguardbreaker",
        "magicguardcalmmind", "regeneratorphysicalwall",
    ),
}
REPLACEMENT_VARIANTS = {
    "jynx": ("choicespecs", "sashunburden", "throatspray", "bootsutility"),
    "lapras": (
        "chillyreception", "dragondance", "waterabsorboffense",
        "choicespecs", "perishtrap",
    ),
    "mawile": ("sashmixed", "choiceband"),
    "obstagoon": ("partingshot", "obstruct"),
    "porygon2": (
        "tracedefensive", "downloadoffense", "trickroom",
        "paralysisutility",
    ),
}
CLAYDOL_VARIANTS = (
    "regeneratorrocks", "regeneratortrickroom", "levitaterocks",
)
TOUCHED_VARIANT_IDS = {
    **NEW_SPECIES_VARIANTS,
    **REPLACEMENT_VARIANTS,
    "claydol": CLAYDOL_VARIANTS,
}
RETAINED_CHANGED_IDS = {("jynx", "choicespecs")}
REMOVED_VARIANT_IDS = {
    ("jynx", "nastyplot"),
    ("jynx", "sashdisruption"),
    ("lapras", "ancientshellperishtrap"),
    ("lapras", "ancientshellpivot"),
    ("lapras", "waterabsorbtank"),
    ("mawile", "choicebandinterceptor"),
    ("mawile", "mixedcoverage"),
    ("mawile", "swordsdancebreaker"),
    ("obstagoon", "bulkysetup"),
    ("obstagoon", "choicescarf"),
    ("obstagoon", "flameorbguts"),
    ("porygon2", "downloadattacker"),
    ("porygon2", "traceutility"),
    ("porygon2", "trickroomsetter"),
}
IVS_31 = (31, 31, 31, 31, 31, 31)
IVS_0_ATK = (31, 0, 31, 31, 31, 31)
IVS_0_ATK_SPEED = (31, 0, 31, 31, 31, 0)


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


EXPECTED_HUMAN_SETS = {
    ("kabutops", "sashlead"): _expected(4, "established", "focussash", "weakarmor", "jolly", (4,252,0,0,0,252), IVS_31, ("stealthrock","flipturn","knockoff","stoneedge"), "fast Focus Sash hazard lead and pivot"),
    ("kabutops", "bootsspinner"): _expected(3, "plausible", "heavydutyboots", "weakarmor", "jolly", (4,252,0,0,0,252), IVS_31, ("stealthrock","rapidspin","flipturn","stoneedge"), "fast hazard setter, remover, and pivot"),
    ("kabutops", "sashsetup"): _expected(2, "experimental", "focussash", "weakarmor", "jolly", (4,252,0,0,0,252), IVS_31, ("swordsdance","stoneedge","liquidation","knockoff"), "Focus Sash physical setup attacker"),
    ("kangaskhan", "choiceband"): _expected(4, "established", "choiceband", "scrappy", "jolly", (0,252,0,0,4,252), IVS_31, ("hammerarm","earthquake","suckerpunch","doubleedge"), "immediate Scrappy physical wallbreaker"),
    ("kangaskhan", "bootsoffense"): _expected(3, "plausible", "heavydutyboots", "scrappy", "jolly", (0,252,0,0,4,252), IVS_31, ("fakeout","earthquake","doubleedge","suckerpunch"), "fast Boots attacker with dual priority"),
    ("kangaskhan", "earlybirdtrapper"): _expected(2, "experimental", "heavydutyboots", "earlybird", "careful", (252,0,0,0,200,56), IVS_0_ATK, ("whirlpool","seismictoss","rest","toxic"), "specially defensive trapping and Rest utility"),
    ("kangaskhan", "substitute"): _expected(2, "experimental", "leftovers", "scrappy", "impish", (252,0,124,0,132,0), IVS_31, ("substitute","toxic","poweruppunch","return"), "bulky Substitute and Power-Up Punch attacker"),
    ("leavanny", "sashwebs"): _expected(4, "established", "focussash", "chlorophyll", "jolly", (0,252,0,0,4,252), IVS_31, ("stickyweb","knockoff","tripleaxel","leafblade"), "Focus Sash Sticky Web lead with physical coverage"),
    ("leavanny", "ejectpackwebs"): _expected(3, "plausible", "ejectpack", "chlorophyll", "hasty", (0,252,0,4,0,252), IVS_31, ("stickyweb","knockoff","tripleaxel","leafstorm"), "Sticky Web lead using Leaf Storm and Eject Pack for momentum"),
    ("leavanny", "swarmwebs"): _expected(3, "plausible", "focussash", "swarm", "jolly", (0,252,0,0,4,252), IVS_31, ("stickyweb","knockoff","tripleaxel","lunge"), "Focus Sash Sticky Web lead using Swarm pressure"),
    ("leavanny", "sashsetup"): _expected(2, "experimental", "focussash", "chlorophyll", "adamant", (0,252,0,0,4,252), IVS_31, ("swordsdance","knockoff","tripleaxel","leafblade"), "Focus Sash physical setup attacker"),
    ("lilligant", "sashquiverdance"): _expected(4, "established", "focussash", "chlorophyll", "modest", (4,0,0,252,0,252), IVS_0_ATK, ("quiverdance","sleeppowder","gigadrain","hiddenpowerfire60"), "Focus Sash Quiver Dance and Sleep Powder setup attacker"),
    ("lilligant", "lifeorbquiverdance"): _expected(3, "plausible", "lifeorb", "chlorophyll", "modest", (0,0,0,252,4,252), IVS_0_ATK, ("quiverdance","gigadrain","alluringvoice","encore"), "Life Orb Quiver Dance attacker with Encore utility"),
    ("mismagius", "substitutesetup"): _expected(3, "plausible", "leftovers", "levitate", "timid", (0,0,0,252,4,252), IVS_0_ATK, ("nastyplot","shadowball","drainingkiss","substitute"), "Leftovers Substitute and Nasty Plot attacker"),
    ("mismagius", "lifeorbsetup"): _expected(4, "established", "lifeorb", "levitate", "timid", (0,0,0,252,4,252), IVS_0_ATK, ("nastyplot","shadowball","dazzlinggleam","mysticalfire"), "Life Orb Nasty Plot wallbreaker"),
    ("mismagius", "choicescarf"): _expected(3, "plausible", "choicescarf", "levitate", "timid", (0,0,0,252,4,252), IVS_0_ATK, ("shadowball","dazzlinggleam","psychic","trick"), "Choice Scarf revenge killer and Trick utility"),
    ("mismagius", "choicespecs"): _expected(2, "experimental", "choicespecs", "levitate", "timid", (0,0,0,252,4,252), IVS_0_ATK, ("shadowball","dazzlinggleam","destinybond","trick"), "experimental Choice Specs attacker with Trick and Destiny Bond"),
    ("mukalola", "resttalk"): _expected(3, "plausible", "blacksludge", "poisontouch", "careful", (252,0,4,0,252,0), IVS_31, ("knockoff","poisonjab","rest","sleeptalk"), "specially defensive RestTalk utility"),
    ("mukalola", "assaultvest"): _expected(4, "established", "assaultvest", "poisontouch", "brave", (204,252,0,0,52,0), IVS_31, ("pursuit","poisonjab","knockoff","fireblast"), "bulky mixed-coverage Assault Vest attacker and Pursuit trapper"),
    ("mukalola", "choiceband"): _expected(3, "plausible", "choiceband", "poisontouch", "adamant", (252,252,0,0,4,0), IVS_31, ("knockoff","gunkshot","firepunch","shadowsneak"), "immediate Choice Band physical wallbreaker"),
    ("raichu", "choiceband"): _expected(3, "plausible", "choiceband", "lightningrod", "jolly", (4,252,0,0,0,252), IVS_31, ("voltswitch","knockoff","extremespeed","volttackle"), "fast physical Choice Band attacker with deliberate Volt Switch momentum"),
    ("raichu", "sashnastyplot"): _expected(4, "established", "focussash", "lightningrod", "timid", (4,0,0,252,0,252), IVS_0_ATK, ("nastyplot","thunderbolt","hiddenpowerice60","grassknot"), "Focus Sash Nasty Plot attacker"),
    ("raichu", "bootsutility"): _expected(3, "plausible", "heavydutyboots", "lightningrod", "jolly", (4,252,0,0,0,252), IVS_31, ("voltswitch","encore","knockoff","nuzzle"), "fast Boots pivot and disruption utility"),
    ("reuniclus", "magicguardspdef"): _expected(3, "plausible", "lifeorb", "magicguard", "sassy", (252,4,0,0,252,0), IVS_31, ("psyshock","knockoff","recover","shadowball"), "specially defensive Magic Guard Life Orb utility attacker"),
    ("reuniclus", "regeneratortrickroom"): _expected(3, "plausible", "heavydutyboots", "regenerator", "sassy", (248,0,0,8,252,0), IVS_0_ATK_SPEED, ("trickroom","recover","futuresight","encore"), "minimum-Speed Regenerator Trick Room support"),
    ("reuniclus", "magicguardbreaker"): _expected(4, "established", "lifeorb", "magicguard", "modest", (252,0,0,212,0,44), IVS_31, ("psychicnoise","knockoff","focusblast","recover"), "Magic Guard Life Orb wallbreaker with recovery"),
    ("reuniclus", "magicguardcalmmind"): _expected(3, "established", "leftovers", "magicguard", "relaxed", (252,0,252,0,4,0), IVS_31, ("calmmind","psychicnoise","knockoff","recover"), "physically defensive Calm Mind win condition"),
    ("reuniclus", "regeneratorphysicalwall"): _expected(3, "plausible", "heavydutyboots", "regenerator", "bold", (252,0,252,0,4,0), IVS_31, ("psychicnoise","knockoff","thunderwave","recover"), "physically defensive Regenerator utility"),
    ("jynx", "choicespecs"): _expected(4, "established", "choicespecs", "dryskin", "timid", (0,0,0,252,4,252), IVS_0_ATK, ("trick","psyshock","icebeam","shadowball"), "fast Choice Specs wallbreaker with Trick"),
    ("jynx", "sashunburden"): _expected(3, "plausible", "focussash", "unburden", "modest", (0,0,0,252,4,252), IVS_0_ATK, ("nastyplot","psyshock","icebeam","energyball"), "Focus Sash Nasty Plot attacker with Unburden potential"),
    ("jynx", "throatspray"): _expected(2, "experimental", "throatspray", "unburden", "modest", (0,0,0,252,4,252), IVS_0_ATK, ("hypervoice","icebeam","focusblast","psychic"), "Throat Spray special attacker with Unburden activation"),
    ("jynx", "bootsutility"): _expected(3, "plausible", "heavydutyboots", "dryskin", "timid", (0,0,0,252,4,252), IVS_0_ATK, ("encore","icebeam","psyshock","focusblast"), "fast Boots attacker and Encore utility"),
    ("lapras", "chillyreception"): _expected(4, "established", "heavydutyboots", "ancientshell", "modest", (248,0,0,252,8,0), IVS_0_ATK, ("chillyreception","freezedry","surf","healbell"), "bulky Ancient Shell pivot and cleric"),
    ("lapras", "dragondance"): _expected(3, "plausible", "loadeddice", "ancientshell", "adamant", (4,252,0,0,0,252), IVS_31, ("dragondance","iciclespear","liquidation","earthquake"), "Loaded Dice physical setup attacker"),
    ("lapras", "waterabsorboffense"): _expected(3, "plausible", "heavydutyboots", "waterabsorb", "modest", (0,0,196,252,0,60), IVS_0_ATK, ("freezedry","hydropump","alluringvoice","icebeam"), "bulky Water Absorb special attacker"),
    ("lapras", "choicespecs"): _expected(2, "plausible", "choicespecs", "ancientshell", "modest", (80,0,0,252,0,176), IVS_0_ATK, ("freezedry","hydropump","icebeam","hiddenpowerfire60"), "Ancient Shell Choice Specs special wallbreaker"),
    ("lapras", "perishtrap"): _expected(2, "experimental", "chestoberry", "ancientshell", "calm", (252,0,4,0,252,0), IVS_0_ATK, ("perishsong","whirlpool","rest","freezedry"), "specially defensive Perish Song and Whirlpool trapper"),
    ("mawile", "sashmixed"): _expected(3, "plausible", "focussash", "closingjaws", "naive", (0,252,0,4,0,252), IVS_31, ("knockoff","ironhead","flamethrower","playrough"), "fast mixed Focus Sash attacker using Closing Jaws"),
    ("mawile", "choiceband"): _expected(4, "established", "choiceband", "closingjaws", "adamant", (248,252,0,0,8,0), IVS_31, ("suckerpunch","ironhead","knockoff","playrough"), "bulky Choice Band Closing Jaws wallbreaker"),
    ("obstagoon", "partingshot"): _expected(4, "established", "flameorb", "guts", "jolly", (0,252,0,0,4,252), IVS_31, ("facade","knockoff","closecombat","partingshot"), "Guts wallbreaker and Parting Shot pivot"),
    ("obstagoon", "obstruct"): _expected(3, "plausible", "flameorb", "guts", "jolly", (0,252,0,0,4,252), IVS_31, ("facade","knockoff","closecombat","obstruct"), "Guts wallbreaker using Obstruct for activation and scouting"),
    ("porygon2", "tracedefensive"): _expected(4, "established", "eviolite", "trace", "calm", (252,0,0,4,252,0), IVS_0_ATK, ("toxic","recover","icebeam","thunderbolt"), "specially defensive Trace utility"),
    ("porygon2", "downloadoffense"): _expected(3, "plausible", "eviolite", "download", "modest", (252,0,0,252,4,0), IVS_0_ATK, ("triattack","recover","icebeam","thunderbolt"), "Download-boosted bulky special attacker"),
    ("porygon2", "trickroom"): _expected(3, "plausible", "eviolite", "trace", "sassy", (252,0,0,4,252,0), IVS_0_ATK_SPEED, ("teleport","trickroom","recover","icebeam"), "minimum-Speed Trace Trick Room support and pivot"),
    ("porygon2", "paralysisutility"): _expected(3, "plausible", "eviolite", "trace", "calm", (252,0,0,4,252,0), IVS_0_ATK, ("thunderwave","recover","shadowball","triattack"), "specially defensive paralysis utility"),
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
    return json.loads(VERSION_PATHS[version].read_text(encoding="utf-8", errors="strict"))


def _canonical_bytes(document):
    return json.dumps(document, ensure_ascii=False, allow_nan=False, sort_keys=True).encode("utf-8")


def _startup(version, fallback):
    options = PublicPriorStartupOptions((str(VERSION_PATHS[version]),), fallback)
    return load_public_prior_runtime_configuration(options, "gen9tugs")


class TestBatchFourDocument(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents = {version: _document(version) for version in VERSIONS}
        cls.datasets = {version: load_public_prior(VERSION_PATHS[version]) for version in VERSIONS}

    def test_01_encoding_hash_identity_and_inventories_are_exact(self):
        for version in VERSIONS:
            raw = VERSION_PATHS[version].read_bytes()
            species_count, variant_count = COUNTS[version]
            dataset = self.datasets[version]
            with self.subTest(version=version):
                self.assertIsInstance(raw.decode("utf-8", errors="strict"), str)
                self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
                self.assertNotIn(b"\r", raw)
                self.assertTrue(raw.endswith(b"\n"))
                self.assertFalse(any(line.rstrip(b" \t") != line for line in raw.splitlines()))
                self.assertEqual(FILE_SIZES[version], len(raw))
                self.assertEqual(RAW_SHA256[version], hashlib.sha256(raw).hexdigest())
                self.assertEqual(CANONICAL_SHA256[version], hashlib.sha256(_canonical_bytes(self.documents[version])).hexdigest())
                self.assertEqual("public", dataset.visibility)
                self.assertEqual("gen9tugs", dataset.identity.format_id)
                self.assertEqual(species_count, len(dataset.species))
                self.assertEqual(variant_count, sum(len(record.variants) for record in dataset.species))
        self.assertEqual(IDENTITY, self.datasets["1.4.0"].identity)
        self.assertEqual("1.2", self.datasets["1.4.0"].patch_version)
        self.assertEqual(8, len(NEW_SPECIES_VARIANTS))
        self.assertEqual(28, sum(map(len, NEW_SPECIES_VARIANTS.values())))
        self.assertEqual(5, len(REPLACEMENT_VARIANTS))
        self.assertEqual(17, sum(map(len, REPLACEMENT_VARIANTS.values())))
        self.assertEqual(1, len({"claydol"}))
        self.assertEqual(6, len(set(REPLACEMENT_VARIANTS) | {"claydol"}))
        self.assertEqual(20, sum(map(len, REPLACEMENT_VARIANTS.values())) + len(CLAYDOL_VARIANTS))

    def test_02_version_1_3_semantics_change_only_as_authorized(self):
        old = self.documents["1.3.0"]
        new = self.documents["1.4.0"]
        for key in ("schema_version", "visibility", "dataset_id", "format_id", "patch_version", "metadata", "sources"):
            self.assertEqual(old[key], new[key], key)
        self.assertEqual("1.4.0", new["dataset_version"])
        self.assertEqual("Public TUGS archetypes 1.4.0", new["display_name"])
        old_records = {record["species_id"]: record for record in old["species"]}
        new_records = {record["species_id"]: record for record in new["species"]}
        self.assertEqual(set(NEW_SPECIES_VARIANTS), set(new_records) - set(old_records))
        self.assertEqual(set(), set(old_records) - set(new_records))
        changed = {species_id for species_id in old_records if old_records[species_id] != new_records[species_id]}
        self.assertEqual(set(REPLACEMENT_VARIANTS) | {"claydol"}, changed)
        unchanged = set(old_records) - changed
        self.assertEqual(28, len(unchanged))
        for species_id in unchanged:
            self.assertEqual(old_records[species_id], new_records[species_id])
        old_claydol = copy.deepcopy(old_records["claydol"])
        old_tr = next(v for v in old_claydol["variants"] if v["variant_id"] == "regeneratortrickroom")
        self.assertEqual(31, old_tr["ivs"]["spe"])
        old_tr["ivs"]["spe"] = 0
        self.assertEqual(old_claydol, new_records["claydol"])
        self.assertEqual(CLAYDOL_VARIANTS, tuple(v["variant_id"] for v in new_records["claydol"]["variants"]))

    def test_03_all_45_human_authored_sets_are_exact_after_normalization(self):
        dataset = self.datasets["1.4.0"]
        expected_keys = {(species, variant) for species, variants in {**NEW_SPECIES_VARIANTS, **REPLACEMENT_VARIANTS}.items() for variant in variants}
        self.assertEqual(45, len(expected_keys))
        self.assertEqual(expected_keys, set(EXPECTED_HUMAN_SETS))
        for key, expected in EXPECTED_HUMAN_SETS.items():
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
                self.assertEqual({"role", "rationale", "distinguishing_evidence", "weight_reason", "confidence"}, set(variant.metadata))
                self.assertEqual(expected["role"], variant.metadata["role"])
                self.assertEqual(expected["confidence"], variant.metadata["confidence"])
                self.assertIn(variant.metadata["confidence"], {"established", "plausible", "experimental"})
                self.assertTrue(all(variant.metadata.values()))
                self.assertLessEqual(sum(variant.evs.as_tuple()), 510)
        source = VERSION_PATHS["1.4.0"].read_text(encoding="utf-8").casefold()
        for token in ('"tera_type"', '"tera"', "terablast", "terastallization", '"gender"'):
            self.assertNotIn(token, source)

    def test_04_real_loader_accepts_all_five_full_snapshots(self):
        for version, (species_count, variant_count) in COUNTS.items():
            dataset = load_public_prior(VERSION_PATHS[version])
            with self.subTest(version=version):
                self.assertEqual("public", dataset.visibility)
                self.assertEqual("gen9tugs", dataset.identity.format_id)
                self.assertEqual(species_count, len(dataset.species))
                self.assertEqual(variant_count, sum(len(record.variants) for record in dataset.species))

    def test_05_real_teamvalidator_accepts_all_required_groups(self):
        docs = self.documents
        unusual = {
            ("jynx", "throatspray"), ("kangaskhan", "earlybirdtrapper"),
            ("kangaskhan", "substitute"), ("lapras", "chillyreception"),
            ("lapras", "waterabsorboffense"), ("lapras", "choicespecs"),
            ("leavanny", "sashwebs"), ("leavanny", "ejectpackwebs"),
            ("lilligant", "sashquiverdance"), ("mawile", "sashmixed"),
            ("mismagius", "choicespecs"), ("mukalola", "assaultvest"),
            ("raichu", "choiceband"), ("raichu", "sashnastyplot"),
            ("reuniclus", "magicguardbreaker"),
            ("claydol", "regeneratortrickroom"), ("porygon2", "trickroom"),
            ("reuniclus", "regeneratortrickroom"),
        }
        groups = [{"label": version, "entries": _entries(docs[version])} for version in VERSIONS]
        groups.extend([
            {"label": "batch4-new", "entries": _entries(docs["1.4.0"], lambda species, variant: species["species_id"] in NEW_SPECIES_VARIANTS)},
            {"label": "batch4-replacements", "entries": _entries(docs["1.4.0"], lambda species, variant: species["species_id"] in REPLACEMENT_VARIANTS)},
            {"label": "batch4-claydol", "entries": _entries(docs["1.4.0"], lambda species, variant: species["species_id"] == "claydol")},
            {"label": "batch4-claydol-corrected", "entries": _entries(docs["1.4.0"], lambda species, variant: (species["species_id"], variant["variant_id"]) == ("claydol", "regeneratortrickroom"))},
            {"label": "unusual-elements", "entries": _entries(docs["1.4.0"], lambda species, variant: (species["species_id"], variant["variant_id"]) in unusual)},
        ])
        completed = _run_teamvalidator(groups)
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertEqual([
            "1.0.0:23/23", "1.1.0:52/52", "1.2.0:80/80",
            "1.3.0:102/102", "1.4.0:132/132", "batch4-new:28/28",
            "batch4-replacements:17/17", "batch4-claydol:3/3",
            "batch4-claydol-corrected:1/1", "unusual-elements:18/18",
        ], completed.stdout.splitlines())

    def test_06_trick_room_hidden_power_and_deliberate_mixed_sets_are_exact(self):
        dataset = self.datasets["1.4.0"]
        trick_room = {
            (record.species_id, variant.variant_id): variant
            for record in dataset.species
            for variant in record.variants
            if "trickroom" in variant.move_ids
        }
        self.assertEqual({
            ("claydol", "regeneratortrickroom"),
            ("porygon2", "trickroom"),
            ("reuniclus", "regeneratortrickroom"),
        }, set(trick_room))
        for key, variant in trick_room.items():
            with self.subTest(trick_room=key):
                self.assertEqual("sassy", variant.nature_id)
                self.assertEqual(0, variant.ivs.atk)
                self.assertEqual(0, variant.ivs.spe)
        non_trick_room_slow_natures = {
            (record.species_id, variant.variant_id): variant.ivs.spe
            for record in dataset.species
            for variant in record.variants
            if variant.nature_id in {"brave", "relaxed", "quiet", "sassy"}
            and "trickroom" not in variant.move_ids
        }
        self.assertEqual(
            {
                ("bastiodon", "sturdymetalburst"): 0,
                ("eelektross", "assaultvest"): 31,
                ("mukalola", "assaultvest"): 31,
                ("reuniclus", "magicguardspdef"): 31,
                ("reuniclus", "magicguardcalmmind"): 31,
                ("swampert", "rockypivot"): 31,
                ("torkoal", "sunbreaker"): 0,
            },
            non_trick_room_slow_natures,
        )
        raichu = dataset.get_species("raichu").get_variant("choiceband")
        self.assertEqual("choiceband", raichu.item_id)
        self.assertEqual(252, raichu.evs.atk)
        self.assertEqual(31, raichu.ivs.atk)
        self.assertIn("voltswitch", raichu.move_ids)
        self.assertIn("volttackle", raichu.move_ids)
        muk = dataset.get_species("mukalola").get_variant("assaultvest")
        self.assertEqual((204,252,0,0,52,0), muk.evs.as_tuple())
        self.assertEqual("brave", muk.nature_id)
        self.assertEqual(31, muk.ivs.atk)
        self.assertIn("pursuit", muk.move_ids)
        self.assertIn("fireblast", muk.move_ids)
        leavanny = dataset.get_species("leavanny").get_variant("ejectpackwebs")
        self.assertEqual("ejectpack", leavanny.item_id)
        self.assertEqual(31, leavanny.ivs.atk)
        self.assertIn("leafstorm", leavanny.move_ids)
        for variant_id in ("magicguardspdef", "magicguardbreaker", "magicguardcalmmind", "regeneratorphysicalwall"):
            variant = dataset.get_species("reuniclus").get_variant(variant_id)
            self.assertIn("knockoff", variant.move_ids)
            self.assertEqual(31, variant.ivs.atk)
        self.assertIn("destinybond", dataset.get_species("mismagius").get_variant("choicespecs").move_ids)
        self.assertIn("hiddenpowerfire60", dataset.get_species("lilligant").get_variant("sashquiverdance").move_ids)
        self.assertIn("hiddenpowerfire60", dataset.get_species("lapras").get_variant("choicespecs").move_ids)
        self.assertIn("hiddenpowerice60", dataset.get_species("raichu").get_variant("sashnastyplot").move_ids)

    def test_07_authorization_firewall_and_negative_validators_are_exact(self):
        document = self.documents["1.4.0"]
        _assert_public_data_firewall(document)
        self.assertEqual(AUTHORIZATION_VALUE, document["metadata"]["authorization"])
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
        for label, invalid in (("wrong_path", wrong_path), ("altered", altered), ("private_phrase", private_phrase), ("forbidden_key", forbidden_key), ("trainer_value", trainer_value), ("team_key", team_key)):
            with self.subTest(firewall=label), self.assertRaises(AssertionError):
                _assert_public_data_firewall(invalid)
        nonpublic = copy.deepcopy(document)
        nonpublic["visibility"] = "private"
        unsupported_source = copy.deepcopy(document)
        unsupported_source["sources"][0]["kind"] = "private"
        for label, invalid in (("visibility", nonpublic), ("source_kind", unsupported_source)):
            with self.subTest(schema=label), self.assertRaises(PublicPriorValidationError):
                validate_public_prior_document(invalid)
        illegal = copy.deepcopy(document)
        kabutops = next(species for species in illegal["species"] if species["species_id"] == "kabutops")
        kabutops["variants"][0]["move_ids"][0] = "spectralthief"
        completed = _run_teamvalidator([{"label": "illegal", "entries": _entries(illegal)}])
        self.assertNotEqual(0, completed.returncode)
        self.assertIn("kabutops/", completed.stderr)


class TestBatchFourSelectionPopulationAndNarrowing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generic_configuration = _startup("1.4.0", PublicPriorFallback.GENERIC)
        cls.none_configuration = _startup("1.4.0", PublicPriorFallback.NONE)
        cls.old_configuration = _startup("1.3.0", PublicPriorFallback.NONE)
        cls.dataset = cls.none_configuration.registry.get(IDENTITY)
        cls.old_dataset = cls.old_configuration.registry.get(PublicPriorIdentity("tugspublicarchetypes", "1.3.0", "gen9tugs"))

    def test_08_every_new_species_and_all_48_touched_variants_are_selectable(self):
        context = self.none_configuration.create_battle_context("gen9tugs")
        for species_id in NEW_SPECIES_VARIANTS:
            result = select_public_prior_variant(context, battle_format="gen9tugs", species_id=species_id, level=100, evidence=None, rng=_FixedRng(0.5))
            self.assertIs(PublicPriorSelectionStatus.SELECTED, result.status)
        selected = set()
        for species_id in TOUCHED_VARIANT_IDS:
            record = self.dataset.get_species(species_id)
            total = sum(variant.weight for variant in record.variants)
            cumulative = 0.0
            for variant in record.variants:
                rng_value = (cumulative + variant.weight / 2) / total
                result = select_public_prior_variant(context, battle_format="gen9tugs", species_id=species_id, level=100, evidence=None, rng=_FixedRng(rng_value))
                with self.subTest(species=species_id, variant=variant.variant_id):
                    self.assertIs(PublicPriorSelectionStatus.SELECTED, result.status)
                    self.assertIs(variant, result.variant)
                selected.add((species_id, variant.variant_id))
                cumulative += variant.weight
        expected = {(species, variant) for species, variants in TOUCHED_VARIANT_IDS.items() for variant in variants}
        self.assertEqual(48, len(expected))
        self.assertEqual(expected, selected)

    def test_09_all_28_unchanged_records_and_two_claydol_variants_are_stable(self):
        old_context = self.old_configuration.create_battle_context("gen9tugs")
        new_context = self.none_configuration.create_battle_context("gen9tugs")
        changed = set(REPLACEMENT_VARIANTS) | {"claydol"}
        unchanged = {record.species_id for record in self.old_dataset.species} - changed
        self.assertEqual(28, len(unchanged))
        for species_id in unchanged:
            for rng_value in (0.0, 0.25, 0.5, 0.75, 0.999999):
                old = select_public_prior_variant(old_context, battle_format="gen9tugs", species_id=species_id, level=100, evidence=None, rng=_FixedRng(rng_value))
                new = select_public_prior_variant(new_context, battle_format="gen9tugs", species_id=species_id, level=100, evidence=None, rng=_FixedRng(rng_value))
                with self.subTest(species=species_id, rng=rng_value):
                    self.assertIs(old.status, new.status)
                    self.assertEqual(old.variant, new.variant)
        for variant_id in ("regeneratorrocks", "levitaterocks"):
            self.assertEqual(self.old_dataset.get_species("claydol").get_variant(variant_id), self.dataset.get_species("claydol").get_variant(variant_id))

    def test_10_every_touched_variant_populates_only_copied_state(self):
        context = self.none_configuration.create_battle_context("gen9tugs")
        modules_before = set(sys.modules)
        populated = set()
        for species_id, variant_ids in TOUCHED_VARIANT_IDS.items():
            for variant_id in variant_ids:
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
                    self.assertEqual(variant.base_ability_id, sampled_pokemon.original_ability)
                    self.assertEqual(variant.move_ids, tuple(move.name for move in sampled_pokemon.moves))
                    self.assertIn(variant.move_ids[0], tuple(move.name for move in sampled_pokemon.moves))
                    self.assertEqual(variant.nature_id, sampled_pokemon.nature)
                    self.assertEqual(variant.evs.as_tuple(), tuple(sampled_pokemon.evs))
                    self.assertEqual(variant.ivs.as_tuple(), tuple(sampled_pokemon.ivs))
                    self.assertEqual(canonical_before, _pokemon_snapshot(pokemon))
                populated.add((species_id, variant_id))
        self.assertEqual(48, len(populated))
        self.assertEqual(set(), {name for name in set(sys.modules) - modules_before if name.startswith("fp.data.team_pools")})

    def test_11_observation_narrowing_covers_all_fourteen_touched_species(self):
        context = self.none_configuration.create_battle_context("gen9tugs")

        def compatible(species_id, *, moves=(), item=None, ability=None):
            return _compatible_ids(self.dataset, species_id, _evidence(context, species_id, moves=moves, item=item, ability=ability))

        checks = (
            ("kabutops", {"bootsspinner"}, {"moves": ("rapidspin",)}),
            ("kabutops", {"bootsspinner"}, {"item": "heavydutyboots"}),
            ("kabutops", {"sashsetup"}, {"moves": ("swordsdance",)}),
            ("kabutops", {"sashsetup"}, {"moves": ("liquidation",)}),
            ("kabutops", {"sashlead"}, {"moves": ("stealthrock","flipturn","knockoff"), "item": "focussash"}),
            ("kabutops", {"sashlead","sashsetup"}, {"moves": ("knockoff",)}),
            ("kangaskhan", {"choiceband"}, {"item": "choiceband"}),
            ("kangaskhan", {"choiceband"}, {"moves": ("hammerarm",)}),
            ("kangaskhan", {"bootsoffense"}, {"moves": ("fakeout",)}),
            ("kangaskhan", {"earlybirdtrapper"}, {"ability": "earlybird"}),
            ("kangaskhan", {"earlybirdtrapper"}, {"moves": ("whirlpool",)}),
            ("kangaskhan", {"substitute"}, {"moves": ("poweruppunch",)}),
            ("leavanny", {"ejectpackwebs"}, {"item": "ejectpack"}),
            ("leavanny", {"ejectpackwebs"}, {"moves": ("leafstorm",)}),
            ("leavanny", {"swarmwebs"}, {"ability": "swarm"}),
            ("leavanny", {"swarmwebs"}, {"moves": ("lunge",)}),
            ("leavanny", {"sashsetup"}, {"moves": ("swordsdance",)}),
            ("leavanny", {"sashwebs","swarmwebs"}, {"moves": ("stickyweb",), "item": "focussash"}),
            ("lilligant", {"sashquiverdance"}, {"moves": ("sleeppowder",)}),
            ("lilligant", {"sashquiverdance"}, {"moves": ("hiddenpowerfire60",)}),
            ("lilligant", {"lifeorbquiverdance"}, {"moves": ("alluringvoice",)}),
            ("mismagius", {"substitutesetup"}, {"item": "leftovers"}),
            ("mismagius", {"lifeorbsetup"}, {"moves": ("mysticalfire",)}),
            ("mismagius", {"choicescarf"}, {"moves": ("psychic",)}),
            ("mismagius", {"choicespecs"}, {"moves": ("destinybond",)}),
            ("mismagius", {"choicescarf","choicespecs"}, {"moves": ("trick",)}),
            ("mismagius", {"substitutesetup","lifeorbsetup"}, {"moves": ("nastyplot",)}),
            ("mukalola", {"resttalk"}, {"item": "blacksludge"}),
            ("mukalola", {"assaultvest"}, {"moves": ("fireblast",)}),
            ("mukalola", {"choiceband"}, {"moves": ("gunkshot",)}),
            ("mukalola", {"resttalk","assaultvest","choiceband"}, {"ability": "poisontouch"}),
            ("raichu", {"choiceband"}, {"moves": ("extremespeed",)}),
            ("raichu", {"sashnastyplot"}, {"moves": ("hiddenpowerice60",)}),
            ("raichu", {"bootsutility"}, {"moves": ("nuzzle",)}),
            ("raichu", {"choiceband","bootsutility"}, {"moves": ("voltswitch",)}),
            ("reuniclus", {"magicguardspdef","magicguardbreaker"}, {"item": "lifeorb", "ability": "magicguard"}),
            ("reuniclus", {"magicguardspdef"}, {"moves": ("psyshock",)}),
            ("reuniclus", {"regeneratortrickroom"}, {"moves": ("trickroom",)}),
            ("reuniclus", {"magicguardbreaker"}, {"moves": ("focusblast",)}),
            ("reuniclus", {"magicguardcalmmind"}, {"moves": ("calmmind",)}),
            ("reuniclus", {"regeneratorphysicalwall"}, {"moves": ("thunderwave",), "ability": "regenerator"}),
            ("jynx", {"choicespecs"}, {"moves": ("shadowball",)}),
            ("jynx", {"sashunburden"}, {"item": "focussash"}),
            ("jynx", {"throatspray"}, {"moves": ("hypervoice",)}),
            ("jynx", {"bootsutility"}, {"moves": ("encore",)}),
            ("jynx", {"sashunburden","throatspray"}, {"ability": "unburden"}),
            ("jynx", {"choicespecs","bootsutility"}, {"ability": "dryskin"}),
            ("lapras", {"chillyreception"}, {"moves": ("chillyreception",)}),
            ("lapras", {"dragondance"}, {"item": "loadeddice"}),
            ("lapras", {"waterabsorboffense"}, {"ability": "waterabsorb"}),
            ("lapras", {"choicespecs"}, {"moves": ("hiddenpowerfire60",)}),
            ("lapras", {"perishtrap"}, {"moves": ("whirlpool",)}),
            ("lapras", {"chillyreception","dragondance","choicespecs","perishtrap"}, {"ability": "ancientshell"}),
            ("mawile", {"sashmixed"}, {"moves": ("flamethrower",)}),
            ("mawile", {"choiceband"}, {"item": "choiceband"}),
            ("mawile", {"sashmixed","choiceband"}, {"ability": "closingjaws"}),
            ("obstagoon", {"partingshot"}, {"moves": ("partingshot",)}),
            ("obstagoon", {"obstruct"}, {"moves": ("obstruct",)}),
            ("obstagoon", {"partingshot","obstruct"}, {"moves": ("facade","knockoff","closecombat"), "item": "flameorb", "ability": "guts"}),
            ("porygon2", {"tracedefensive"}, {"moves": ("toxic",)}),
            ("porygon2", {"downloadoffense"}, {"ability": "download"}),
            ("porygon2", {"trickroom"}, {"moves": ("teleport",)}),
            ("porygon2", {"paralysisutility"}, {"moves": ("shadowball",)}),
            ("porygon2", {"tracedefensive","trickroom","paralysisutility"}, {"ability": "trace"}),
            ("porygon2", {"downloadoffense","paralysisutility"}, {"moves": ("triattack",)}),
            ("claydol", {"regeneratortrickroom"}, {"moves": ("trickroom",)}),
            ("claydol", {"regeneratorrocks","levitaterocks"}, {"moves": ("rapidspin",)}),
            ("claydol", {"regeneratorrocks","regeneratortrickroom"}, {"ability": "regenerator"}),
            ("claydol", {"levitaterocks"}, {"ability": "levitate"}),
        )
        self.assertEqual(set(TOUCHED_VARIANT_IDS), {species for species, _, _ in checks})
        for species_id, expected, kwargs in checks:
            with self.subTest(species=species_id, expected=expected, evidence=kwargs):
                self.assertEqual(expected, compatible(species_id, **kwargs))

    def test_12_conflicts_ambiguity_and_removed_variants_are_isolated(self):
        context = self.none_configuration.create_battle_context("gen9tugs")
        conflict = _evidence(context, "kabutops", moves=("swordsdance",), item="heavydutyboots")
        result = select_public_prior_variant(context, battle_format="gen9tugs", species_id="kabutops", level=100, evidence=conflict, rng=_FixedRng(0.5))
        self.assertIs(PublicPriorSelectionStatus.NO_COMPATIBLE_VARIANT, result.status)
        self.assertEqual({"choicescarf","choicespecs"}, _compatible_ids(self.dataset, "mismagius", _evidence(context, "mismagius", moves=("trick",))))
        old_ids = {
            (species_id, variant.variant_id)
            for species_id in REPLACEMENT_VARIANTS
            for variant in self.old_dataset.get_species(species_id).variants
        }
        new_ids = {
            (species_id, variant.variant_id)
            for species_id in REPLACEMENT_VARIANTS
            for variant in self.dataset.get_species(species_id).variants
        }
        self.assertEqual(RETAINED_CHANGED_IDS, old_ids & new_ids)
        self.assertEqual(REMOVED_VARIANT_IDS, old_ids - new_ids)
        for species_id, variant_id in REMOVED_VARIANT_IDS:
            with self.subTest(species=species_id, variant=variant_id):
                self.assertIsNone(self.dataset.get_species(species_id).get_variant(variant_id))
                self.assertNotIn(variant_id, REPLACEMENT_VARIANTS[species_id])
        self.assertEqual({species: set(variants) for species, variants in REPLACEMENT_VARIANTS.items()}, {
            species: {variant.variant_id for variant in self.dataset.get_species(species).variants}
            for species in REPLACEMENT_VARIANTS
        })

    def test_13_public_sampling_has_no_private_pool_or_candidate_dependency(self):
        paths = (
            ROOT / "fp" / "battle" / "public_prior_context.py",
            ROOT / "fp" / "data" / "public_priors" / "runtime.py",
            ROOT / "fp" / "search" / "public_prior_sampling.py",
            ROOT / "fp" / "search" / "standard_battles.py",
        )
        forbidden = ("fp.data.team_pools", "TeamPool", "TeamRecord", "PokemonRecord", "baseline_candidate_ids", "active_candidate_ids")
        for path in paths:
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)
        parameters = inspect.signature(select_public_prior_variant).parameters
        self.assertNotIn("candidate_id", parameters)
        self.assertNotIn("candidate_ids", parameters)

    def test_14_still_uncovered_species_obeys_explicit_fallback_policy(self):
        self.assertIsNone(self.dataset.get_species("xatu"))
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
        json.dumps(sampled.request_json or {})


if __name__ == "__main__":
    unittest.main()
