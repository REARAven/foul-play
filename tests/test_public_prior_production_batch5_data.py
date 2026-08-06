import copy
import hashlib
import inspect
import json
import math
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from fp import constants
from fp.battle.public_prior_context import PublicPriorFallback
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
    select_public_prior_variant,
)
from fp.search.standard_battles import prepare_battles
from test_public_prior_production_batch3_data import (
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
VERSIONS = ("1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.4.0", "1.5.0")
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
    "1.5.0": "a3bf5c0748603973968d85f2dfead1948d9bb619d1dc1be727b1770c1023ea93",
}
CANONICAL_SHA256 = {
    "1.0.0": "ddb04702321aace7d7368938016a18d54841935d2bf5df194cfaf7af8d4a55ae",
    "1.1.0": "59425bcd4371870ec6489d384ba9a7d6a3c3c5375371fbfc48c11fab17740d8d",
    "1.2.0": "bc0456562feaccf6fc4a64bf0d189deec8b5c6331c5c64d8bee606674d946376",
    "1.3.0": "53bcda29981d448b03151b47002a63e267cd180130241e445f8af123b810b226",
    "1.4.0": "97dd627a9b256eb2abc61ce61a4d8b9a3972dc89e0591c26887d29ced4a91434",
    "1.5.0": "92f77722ccdd21e9c44af451058540b2d1b70006a858c6cc287c2f360386e0d2",
}
FILE_SIZES = {
    "1.0.0": 25644,
    "1.1.0": 69667,
    "1.2.0": 107311,
    "1.3.0": 137237,
    "1.4.0": 177609,
    "1.5.0": 205650,
}
COUNTS = {
    "1.0.0": (8, 23),
    "1.1.0": (18, 52),
    "1.2.0": (28, 80),
    "1.3.0": (34, 102),
    "1.4.0": (42, 132),
    "1.5.0": (49, 153),
}
IDENTITY = PublicPriorIdentity("tugspublicarchetypes", "1.5.0", "gen9tugs")
SOURCE_IDS = ("publicformatpatch12", "publicmanualv1")
SERVER_HEAD = "9879c269e"

NEW_SPECIES_VARIANTS = {
    "slurpuff": ("sashwebs", "bellydrum", "redcardcalmmind"),
    "taurospaldeaaqua": ("choiceband", "lifeorb", "choicescarf"),
    "toxtricity": (
        "scarfsludgebomb", "scarfpsychicnoise", "choicespecs", "throatspray",
    ),
    "toxtricitylowkey": (
        "scarfsludgebomb", "scarfpsychicnoise", "choicespecs", "throatspray",
    ),
    "vileplume": ("effectsporephysical", "chlorophyllspdef"),
    "weezing": ("levitatephysical", "neutralizinggasphysical"),
    "xatu": ("helmetphysical", "leftoversutility", "bootsutility"),
}
REPLACEMENT_VARIANTS = {
    "starmie": ("lifeorbanalytic", "bootsnaturalcure", "choicespecs"),
    "swampert": ("spdefrocks", "offensiveyawn", "chestorest"),
    "sylveon": (
        "spdefwish", "defensivecalmmind", "toxicwish", "bootscleric",
    ),
    "tinkaton": ("defensiveutility", "sashsetup"),
    "torkoal": ("heatrockwillowisp", "heatrockrest", "bootsutility"),
    "vikavolt": ("sashwebs", "bulkyboots"),
}
TOUCHED_VARIANT_IDS = {**NEW_SPECIES_VARIANTS, **REPLACEMENT_VARIANTS}
UNCHANGED_SPECIES = {
    "aerodactyl", "altaria", "arcaninehisui", "bastiodon", "bombirdier",
    "chesnaught", "claydol", "cursola", "dhelmise", "drapion", "druddigon",
    "dudunsparce", "dudunsparcethreesegment", "dugtrioalola", "dustox",
    "eelektross", "flygon", "forretress", "froslass", "heracross",
    "hitmontop", "houndoom", "jellicent", "jynx", "kabutops", "kangaskhan",
    "lapras", "leavanny", "lilligant", "mawile", "mismagius", "mukalola",
    "obstagoon", "porygon2", "raichu", "reuniclus",
}
EXPECTED_ROSTER_IDS = UNCHANGED_SPECIES | set(NEW_SPECIES_VARIANTS) | set(REPLACEMENT_VARIANTS)
REMOVED_VARIANT_IDS = {
    ("starmie", "analyticbreaker"),
    ("starmie", "rapidspinutility"),
    ("swampert", "choiceband"),
    ("swampert", "cursechesto"),
    ("swampert", "rockypivot"),
    ("sylveon", "calmmind"),
    ("sylveon", "choicespecs"),
    ("sylveon", "wishsupport"),
    ("tinkaton", "assaultvest"),
    ("tinkaton", "rocksutility"),
    ("tinkaton", "swordsdance"),
    ("torkoal", "droughtspinner"),
    ("torkoal", "overheatmomentum"),
    ("torkoal", "sunbreaker"),
    ("vikavolt", "bootspivot"),
    ("vikavolt", "specsbreaker"),
    ("vikavolt", "stickyweb"),
}

EXPECTED_HUMAN_SETS = {
    ('slurpuff', 'sashwebs'): (4, 'established', 'focussash', 'unburden', 'naive', (0, 0, 0, 252, 0, 252), (0, 0, 0, 31, 0, 31), ('stickyweb', 'magiccoat', 'mistyexplosion', 'yawn'), 'maximum-fragility Focus Sash Sticky Web disruption lead'),
    ('slurpuff', 'bellydrum'): (3, 'plausible', 'sitrusberry', 'unburden', 'jolly', (4, 252, 0, 0, 0, 252), (31, 31, 31, 31, 31, 31), ('bellydrum', 'playrough', 'drainpunch', 'facade'), 'Sitrus Berry Belly Drum physical sweeper'),
    ('slurpuff', 'redcardcalmmind'): (2, 'experimental', 'redcard', 'unburden', 'modest', (4, 0, 0, 252, 0, 252), (31, 0, 31, 31, 31, 31), ('calmmind', 'flamethrower', 'drainingkiss', 'energyball'), 'Red Card Calm Mind special attacker with potential Unburden activation'),
    ('starmie', 'lifeorbanalytic'): (4, 'established', 'lifeorb', 'analytic', 'timid', (0, 0, 0, 252, 4, 252), (31, 31, 31, 31, 31, 31), ('hydropump', 'icebeam', 'thunderbolt', 'rapidspin'), 'Life Orb Analytic attacker with emergency removal'),
    ('starmie', 'bootsnaturalcure'): (4, 'established', 'heavydutyboots', 'naturalcure', 'timid', (252, 0, 0, 4, 0, 252), (31, 31, 31, 31, 31, 31), ('scald', 'recover', 'teleport', 'rapidspin'), 'fast bulky Natural Cure removal and pivot utility'),
    ('starmie', 'choicespecs'): (3, 'plausible', 'choicespecs', 'analytic', 'timid', (4, 0, 0, 252, 0, 252), (31, 0, 31, 31, 31, 31), ('thunderbolt', 'icebeam', 'psyshock', 'hydropump'), 'Choice Specs Analytic wallbreaker'),
    ('swampert', 'spdefrocks'): (4, 'established', 'leftovers', 'torrent', 'careful', (252, 4, 0, 0, 252, 0), (31, 31, 31, 31, 31, 31), ('stealthrock', 'earthquake', 'flipturn', 'knockoff'), 'specially defensive hazard setter and pivot'),
    ('swampert', 'offensiveyawn'): (3, 'plausible', 'leftovers', 'torrent', 'adamant', (252, 252, 0, 0, 4, 0), (31, 31, 31, 31, 31, 31), ('yawn', 'flipturn', 'stealthrock', 'earthquake'), 'offensive hazard setter using Yawn and Flip Turn to force progress'),
    ('swampert', 'chestorest'): (2, 'experimental', 'chestoberry', 'torrent', 'careful', (252, 4, 0, 0, 252, 0), (31, 31, 31, 31, 31, 31), ('rest', 'knockoff', 'toxic', 'earthquake'), 'specially defensive ChestoRest utility'),
    ('sylveon', 'spdefwish'): (4, 'established', 'leftovers', 'pixilate', 'calm', (252, 0, 0, 4, 252, 0), (31, 0, 31, 31, 31, 31), ('mysticalfire', 'hypervoice', 'wish', 'protect'), 'specially defensive Wish and Protect support'),
    ('sylveon', 'defensivecalmmind'): (3, 'plausible', 'leftovers', 'pixilate', 'bold', (252, 0, 252, 4, 0, 0), (31, 0, 31, 31, 31, 31), ('mysticalfire', 'hypervoice', 'wish', 'calmmind'), 'physically defensive Calm Mind win condition with Wish'),
    ('sylveon', 'toxicwish'): (3, 'plausible', 'leftovers', 'pixilate', 'calm', (252, 0, 0, 4, 252, 0), (31, 0, 31, 31, 31, 31), ('toxic', 'hypervoice', 'wish', 'protect'), 'specially defensive Toxic and Wish support'),
    ('sylveon', 'bootscleric'): (3, 'plausible', 'heavydutyboots', 'pixilate', 'calm', (252, 0, 0, 4, 252, 0), (31, 0, 31, 31, 31, 31), ('healbell', 'hypervoice', 'wish', 'protect'), 'Heavy-Duty Boots cleric and Wish support'),
    ('taurospaldeaaqua', 'choiceband'): (4, 'established', 'choiceband', 'intimidate', 'jolly', (4, 252, 0, 0, 0, 252), (31, 31, 31, 31, 31, 31), ('closecombat', 'wavecrash', 'earthquake', 'lashout'), 'immediate physical wallbreaker'),
    ('taurospaldeaaqua', 'lifeorb'): (3, 'plausible', 'lifeorb', 'intimidate', 'jolly', (4, 252, 0, 0, 0, 252), (31, 31, 31, 31, 31, 31), ('closecombat', 'wavecrash', 'earthquake', 'aquajet'), 'flexible Life Orb attacker with priority'),
    ('taurospaldeaaqua', 'choicescarf'): (3, 'plausible', 'choicescarf', 'intimidate', 'jolly', (4, 252, 0, 0, 0, 252), (31, 31, 31, 31, 31, 31), ('closecombat', 'wavecrash', 'earthquake', 'stoneedge'), 'Choice Scarf revenge killer'),
    ('tinkaton', 'defensiveutility'): (4, 'established', 'leftovers', 'moldbreaker', 'careful', (248, 0, 184, 0, 76, 0), (31, 31, 31, 31, 31, 31), ('stealthrock', 'gigatonhammer', 'knockoff', 'encore'), 'bulky Mold Breaker hazard and disruption utility'),
    ('tinkaton', 'sashsetup'): (2, 'experimental', 'focussash', 'moldbreaker', 'jolly', (0, 252, 0, 0, 4, 252), (31, 31, 31, 31, 31, 31), ('gigatonhammer', 'knockoff', 'swordsdance', 'playrough'), 'fast Focus Sash Swords Dance attacker'),
    ('torkoal', 'heatrockwillowisp'): (4, 'established', 'heatrock', 'drought', 'bold', (248, 0, 252, 0, 8, 0), (31, 31, 31, 31, 31, 31), ('lavaplume', 'rapidspin', 'stealthrock', 'willowisp'), 'physical wall, extended-sun setter, hazards, removal, and burn utility'),
    ('torkoal', 'heatrockrest'): (3, 'plausible', 'heatrock', 'drought', 'bold', (248, 0, 252, 0, 8, 0), (31, 31, 31, 31, 31, 31), ('lavaplume', 'rapidspin', 'stealthrock', 'rest'), 'extended-sun setter using Rest for recovery'),
    ('torkoal', 'bootsutility'): (4, 'established', 'heavydutyboots', 'drought', 'bold', (248, 0, 252, 8, 0, 0), (31, 31, 31, 31, 31, 31), ('stealthrock', 'rapidspin', 'lavaplume', 'bodypress'), 'Heavy-Duty Boots hazard, removal, and physical-pressure utility'),
    ('toxtricity', 'scarfsludgebomb'): (4, 'established', 'choicescarf', 'punkrock', 'timid', (4, 0, 0, 252, 0, 252), (31, 0, 31, 31, 31, 31), ('voltswitch', 'boomburst', 'overdrive', 'sludgebomb'), 'Choice Scarf special pivot and revenge killer'),
    ('toxtricity', 'scarfpsychicnoise'): (3, 'plausible', 'choicescarf', 'punkrock', 'timid', (4, 0, 0, 252, 0, 252), (31, 0, 31, 31, 31, 31), ('voltswitch', 'boomburst', 'psychicnoise', 'sludgebomb'), 'Choice Scarf pivot using Psychic Noise for recovery denial and coverage'),
    ('toxtricity', 'choicespecs'): (4, 'established', 'choicespecs', 'punkrock', 'modest', (0, 0, 0, 252, 4, 252), (31, 0, 31, 31, 31, 31), ('boomburst', 'overdrive', 'sludgewave', 'voltswitch'), 'Choice Specs Punk Rock wallbreaker'),
    ('toxtricity', 'throatspray'): (3, 'plausible', 'throatspray', 'punkrock', 'modest', (0, 0, 0, 252, 4, 252), (31, 0, 31, 31, 31, 31), ('shiftgear', 'boomburst', 'overdrive', 'sludgewave'), 'Shift Gear and Throat Spray special setup attacker'),
    ('toxtricitylowkey', 'scarfsludgebomb'): (4, 'established', 'choicescarf', 'punkrock', 'timid', (4, 0, 0, 252, 0, 252), (31, 0, 31, 31, 31, 31), ('voltswitch', 'boomburst', 'overdrive', 'sludgebomb'), 'Choice Scarf special pivot and revenge killer'),
    ('toxtricitylowkey', 'scarfpsychicnoise'): (3, 'plausible', 'choicescarf', 'punkrock', 'timid', (4, 0, 0, 252, 0, 252), (31, 0, 31, 31, 31, 31), ('voltswitch', 'boomburst', 'psychicnoise', 'sludgebomb'), 'Choice Scarf pivot using Psychic Noise for recovery denial and coverage'),
    ('toxtricitylowkey', 'choicespecs'): (4, 'established', 'choicespecs', 'punkrock', 'modest', (0, 0, 0, 252, 4, 252), (31, 0, 31, 31, 31, 31), ('boomburst', 'overdrive', 'sludgewave', 'voltswitch'), 'Choice Specs Punk Rock wallbreaker'),
    ('toxtricitylowkey', 'throatspray'): (3, 'plausible', 'throatspray', 'punkrock', 'modest', (0, 0, 0, 252, 4, 252), (31, 0, 31, 31, 31, 31), ('shiftgear', 'boomburst', 'overdrive', 'sludgewave'), 'Shift Gear and Throat Spray special setup attacker'),
    ('vikavolt', 'sashwebs'): (3, 'plausible', 'focussash', 'levitate', 'modest', (0, 0, 4, 252, 0, 252), (31, 0, 31, 31, 31, 31), ('stickyweb', 'voltswitch', 'bugbuzz', 'energyball'), 'Focus Sash Sticky Web lead with strong special pressure'),
    ('vikavolt', 'bulkyboots'): (3, 'plausible', 'heavydutyboots', 'levitate', 'bold', (252, 0, 252, 4, 0, 0), (31, 0, 31, 31, 31, 31), ('voltswitch', 'bugbuzz', 'roost', 'stickyweb'), 'physically bulky Boots Web setter and pivot'),
    ('vileplume', 'effectsporephysical'): (4, 'established', 'rockyhelmet', 'effectspore', 'bold', (252, 0, 252, 0, 4, 0), (31, 0, 31, 31, 31, 31), ('strengthsap', 'sludgebomb', 'gigadrain', 'leechseed'), 'physically defensive contact punisher and sustain utility'),
    ('vileplume', 'chlorophyllspdef'): (3, 'plausible', 'blacksludge', 'chlorophyll', 'calm', (252, 0, 0, 4, 252, 0), (31, 0, 31, 31, 31, 31), ('aromatherapy', 'synthesis', 'sleeppowder', 'sludgebomb'), 'specially defensive cleric and sleep utility with Chlorophyll'),
    ('weezing', 'levitatephysical'): (4, 'established', 'rockyhelmet', 'levitate', 'bold', (252, 0, 252, 0, 0, 4), (31, 0, 31, 31, 31, 31), ('sludgebomb', 'willowisp', 'painsplit', 'haze'), 'physically defensive Levitate contact punishment and setup control'),
    ('weezing', 'neutralizinggasphysical'): (3, 'plausible', 'blacksludge', 'neutralizinggas', 'bold', (252, 0, 252, 0, 0, 4), (31, 0, 31, 31, 31, 31), ('sludgebomb', 'flamethrower', 'painsplit', 'taunt'), 'physically defensive Neutralizing Gas disruption utility'),
    ('xatu', 'helmetphysical'): (4, 'established', 'rockyhelmet', 'magicbounce', 'bold', (252, 0, 252, 0, 0, 4), (31, 0, 31, 31, 31, 31), ('psychic', 'heatwave', 'roost', 'teleport'), 'physically defensive Magic Bounce pivot and contact punisher'),
    ('xatu', 'leftoversutility'): (3, 'plausible', 'leftovers', 'magicbounce', 'bold', (252, 0, 252, 0, 4, 0), (31, 0, 31, 31, 31, 31), ('roost', 'thunderwave', 'teleport', 'nightshade'), 'physically defensive paralysis and fixed-damage utility'),
    ('xatu', 'bootsutility'): (3, 'plausible', 'heavydutyboots', 'magicbounce', 'timid', (252, 0, 0, 4, 0, 252), (31, 31, 31, 31, 31, 31), ('defog', 'uturn', 'thunderwave', 'heatwave'), 'fast Boots removal, pivoting, and paralysis utility'),
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
    return json.dumps(
        document, ensure_ascii=False, allow_nan=False, sort_keys=True
    ).encode("utf-8")


def _startup(version, fallback):
    options = PublicPriorStartupOptions((str(VERSION_PATHS[version]),), fallback)
    return load_public_prior_runtime_configuration(options, "gen9tugs")


def _server_root():
    return Path(
        os.environ.get("TUGS_SHOWDOWN_ROOT", ROOT.parent / "TUGS-showdown")
    ).resolve()


def _server_roster_ids():
    source = (_server_root() / "data" / "mods" / "tugs" / "roster.ts").read_text(
        encoding="utf-8", errors="strict"
    )
    names = re.findall(r"^\s*'([^']+)',\s*$", source, flags=re.MULTILINE)
    return {re.sub(r"[^a-z0-9]+", "", name.casefold()) for name in names}


def _run_low_key_isolation_validator():
    script = r"""
const {Dex} = require('./dist/sim/dex');
const {TeamValidator} = require('./dist/sim/team-validator');
Dex.includeData();
const exact = {species: 'Toxtricity-Low-Key', item: 'Throat Spray', ability: 'Punk Rock', moves: ['Shift Gear', 'Boomburst', 'Overdrive', 'Sludge Wave'], nature: 'Modest', evs: {spa: 252, spd: 4, spe: 252}, ivs: {atk: 0}};
const valid = (label, team, format) => console.log(label + ':' + (TeamValidator.get(format).validateTeam(team) ? 'rejected' : 'valid'));
valid('low-key', [exact], 'gen9tugs');
valid('amped', [{...exact, species: 'Toxtricity'}], 'gen9tugs');
valid('acid', [{species: 'Toxtricity-Low-Key', ability: 'Punk Rock', moves: ['Acid'], evs: {hp: 1}}], 'gen9tugs');
valid('national', [exact], 'gen9nationaldex');
const tugs = Dex.mod('tugs').species.getLearnsetData('toxtricitylowkey').learnset;
const parent = Dex.mod('gen9').species.getLearnsetData('toxtricitylowkey').learnset;
console.log('source:' + JSON.stringify(tugs.shiftgear));
console.log('parent:' + String(parent.shiftgear));
console.log('forms:' + Dex.mod('tugs').species.get('Toxtricity').id + '/' + Dex.mod('tugs').species.get('Toxtricity-Low-Key').id);
"""
    return subprocess.run(
        ["node", "-e", script], cwd=_server_root(), capture_output=True,
        text=True, timeout=30, check=False,
    )


class TestBatchFiveDocument(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents = {version: _document(version) for version in VERSIONS}
        cls.datasets = {
            version: load_public_prior(VERSION_PATHS[version])
            for version in VERSIONS
        }

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
                self.assertFalse(
                    any(line.rstrip(b" \t") != line for line in raw.splitlines())
                )
                self.assertEqual(FILE_SIZES[version], len(raw))
                self.assertEqual(RAW_SHA256[version], hashlib.sha256(raw).hexdigest())
                self.assertEqual(
                    CANONICAL_SHA256[version],
                    hashlib.sha256(_canonical_bytes(self.documents[version])).hexdigest(),
                )
                self.assertEqual("public", dataset.visibility)
                self.assertEqual("gen9tugs", dataset.identity.format_id)
                self.assertEqual(species_count, len(dataset.species))
                self.assertEqual(
                    variant_count,
                    sum(len(record.variants) for record in dataset.species),
                )
        self.assertEqual(IDENTITY, self.datasets["1.5.0"].identity)
        self.assertEqual("1.2", self.datasets["1.5.0"].patch_version)
        self.assertEqual(7, len(NEW_SPECIES_VARIANTS))
        self.assertEqual(21, sum(map(len, NEW_SPECIES_VARIANTS.values())))
        self.assertEqual(6, len(REPLACEMENT_VARIANTS))
        self.assertEqual(17, sum(map(len, REPLACEMENT_VARIANTS.values())))
        self.assertEqual(36, len(UNCHANGED_SPECIES))

    def test_02_version_1_4_semantics_change_only_as_authorized(self):
        old = self.documents["1.4.0"]
        new = self.documents["1.5.0"]
        for key in (
            "schema_version", "visibility", "dataset_id", "format_id",
            "patch_version", "metadata", "sources",
        ):
            self.assertEqual(old[key], new[key], key)
        self.assertEqual("1.5.0", new["dataset_version"])
        self.assertEqual("Public TUGS archetypes 1.5.0", new["display_name"])
        old_records = {record["species_id"]: record for record in old["species"]}
        new_records = {record["species_id"]: record for record in new["species"]}
        self.assertEqual(set(NEW_SPECIES_VARIANTS), set(new_records) - set(old_records))
        self.assertEqual(set(), set(old_records) - set(new_records))
        changed = {
            species_id for species_id in old_records
            if old_records[species_id] != new_records[species_id]
        }
        self.assertEqual(set(REPLACEMENT_VARIANTS), changed)
        self.assertEqual(UNCHANGED_SPECIES, set(old_records) - changed)
        for species_id in UNCHANGED_SPECIES:
            self.assertEqual(old_records[species_id], new_records[species_id])
        old_replacement_count = sum(
            len(old_records[species_id]["variants"])
            for species_id in REPLACEMENT_VARIANTS
        )
        self.assertEqual(17, old_replacement_count)
        for species_id, expected_ids in REPLACEMENT_VARIANTS.items():
            self.assertEqual(
                expected_ids,
                tuple(v["variant_id"] for v in new_records[species_id]["variants"]),
            )

    def test_03_all_38_human_authored_sets_are_exact_after_normalization(self):
        dataset = self.datasets["1.5.0"]
        expected_keys = {
            (species, variant)
            for species, variants in TOUCHED_VARIANT_IDS.items()
            for variant in variants
        }
        self.assertEqual(38, len(expected_keys))
        self.assertEqual(expected_keys, set(EXPECTED_HUMAN_SETS))
        for key, expected in EXPECTED_HUMAN_SETS.items():
            species_id, variant_id = key
            variant = dataset.get_species(species_id).get_variant(variant_id)
            weight, confidence, item, ability, nature, evs, ivs, moves, role = expected
            with self.subTest(species=species_id, variant=variant_id):
                self.assertEqual(float(weight), variant.weight)
                self.assertTrue(math.isfinite(variant.weight))
                self.assertGreater(variant.weight, 0)
                self.assertEqual(item, variant.item_id)
                self.assertEqual(ability, variant.base_ability_id)
                self.assertEqual(nature, variant.nature_id)
                self.assertEqual(evs, variant.evs.as_tuple())
                self.assertEqual(ivs, variant.ivs.as_tuple())
                self.assertEqual(moves, variant.move_ids)
                self.assertEqual(100, variant.level)
                self.assertEqual(SOURCE_IDS, variant.source_ids)
                self.assertEqual(role, variant.metadata["role"])
                self.assertEqual(confidence, variant.metadata["confidence"])
                self.assertEqual(
                    {"role", "rationale", "distinguishing_evidence", "weight_reason", "confidence"},
                    set(variant.metadata),
                )
                self.assertTrue(all(variant.metadata.values()))
                self.assertLessEqual(sum(variant.evs.as_tuple()), 510)
        slurpuff = dataset.get_species("slurpuff").get_variant("sashwebs")
        self.assertEqual((0, 0, 0, 31, 0, 31), slurpuff.ivs.as_tuple())
        source = VERSION_PATHS["1.5.0"].read_text(encoding="utf-8").casefold()
        for token in ('"tera_type"', '"tera"', "terablast", "terastallization", '"gender"'):
            self.assertNotIn(token, source)

    def test_04_server_head_roster_equality_and_all_six_loaders_are_exact(self):
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_server_root(),
            capture_output=True, text=True, timeout=10, check=False,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertEqual(SERVER_HEAD, completed.stdout.strip())
        server_ids = _server_roster_ids()
        dataset_ids = {record.species_id for record in self.datasets["1.5.0"].species}
        self.assertEqual(49, len(server_ids))
        self.assertEqual(EXPECTED_ROSTER_IDS, server_ids)
        self.assertEqual(server_ids, dataset_ids)
        self.assertEqual(set(), server_ids - dataset_ids)
        self.assertEqual(set(), dataset_ids - server_ids)
        for version, (species_count, variant_count) in COUNTS.items():
            dataset = load_public_prior(VERSION_PATHS[version])
            with self.subTest(version=version):
                self.assertEqual("public", dataset.visibility)
                self.assertEqual("gen9tugs", dataset.identity.format_id)
                self.assertEqual(species_count, len(dataset.species))
                self.assertEqual(
                    variant_count,
                    sum(len(record.variants) for record in dataset.species),
                )

    def test_05_trusted_teamvalidator_accepts_all_required_groups_and_isolates_low_key(self):
        docs = self.documents
        groups = [{"label": version, "entries": _entries(docs[version])} for version in VERSIONS]
        groups.extend([
            {"label": "batch5-new", "entries": _entries(docs["1.5.0"], lambda species, variant: species["species_id"] in NEW_SPECIES_VARIANTS)},
            {"label": "batch5-replacements", "entries": _entries(docs["1.5.0"], lambda species, variant: species["species_id"] in REPLACEMENT_VARIANTS)},
            {"label": "toxtricity-forms", "entries": _entries(docs["1.5.0"], lambda species, variant: species["species_id"] in {"toxtricity", "toxtricitylowkey"})},
            {"label": "slurpuff", "entries": _entries(docs["1.5.0"], lambda species, variant: species["species_id"] == "slurpuff")},
            {"label": "low-key-throatspray", "entries": _entries(docs["1.5.0"], lambda species, variant: (species["species_id"], variant["variant_id"]) == ("toxtricitylowkey", "throatspray"))},
        ])
        completed = _run_teamvalidator(groups)
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertEqual([
            "1.0.0:23/23", "1.1.0:52/52", "1.2.0:80/80",
            "1.3.0:102/102", "1.4.0:132/132", "1.5.0:153/153",
            "batch5-new:21/21", "batch5-replacements:17/17",
            "toxtricity-forms:8/8", "slurpuff:3/3", "low-key-throatspray:1/1",
        ], completed.stdout.splitlines())
        isolation = _run_low_key_isolation_validator()
        self.assertEqual(0, isolation.returncode, isolation.stderr)
        self.assertEqual([
            "low-key:valid", "amped:valid", "acid:valid", "national:rejected",
            'source:["9L52","8L52"]', "parent:undefined",
            "forms:toxtricity/toxtricitylowkey",
        ], isolation.stdout.splitlines())

    def test_06_trick_room_hidden_power_and_prior_deliberate_structures_are_exact(self):
        dataset = self.datasets["1.5.0"]
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
        raichu = dataset.get_species("raichu").get_variant("choiceband")
        self.assertIn("voltswitch", raichu.move_ids)
        self.assertIn("fireblast", dataset.get_species("mukalola").get_variant("assaultvest").move_ids)
        self.assertIn("destinybond", dataset.get_species("mismagius").get_variant("choicespecs").move_ids)
        hidden_power = {
            ("houndoom", "nastyplotlifeorb"): "hiddenpowergrass60",
            ("lapras", "choicespecs"): "hiddenpowerfire60",
            ("lilligant", "sashquiverdance"): "hiddenpowerfire60",
            ("raichu", "sashnastyplot"): "hiddenpowerice60",
        }
        for key, move_id in hidden_power.items():
            self.assertIn(move_id, dataset.get_species(key[0]).get_variant(key[1]).move_ids)
        low_key = dataset.get_species("toxtricitylowkey").get_variant("throatspray")
        self.assertEqual(
            ("shiftgear", "boomburst", "overdrive", "sludgewave"),
            low_key.move_ids,
        )

    def test_07_authorization_firewall_and_negative_validators_are_exact(self):
        document = self.documents["1.5.0"]
        _assert_public_data_firewall(document)
        self.assertEqual(AUTHORIZATION_VALUE, document["metadata"]["authorization"])
        self.assertEqual(1, json.dumps(document).count(AUTHORIZATION_VALUE))
        invalid_firewall = []
        wrong_path = copy.deepcopy(document)
        wrong_path["metadata"]["note"] = AUTHORIZATION_VALUE
        invalid_firewall.append(("wrong_path", wrong_path))
        altered = copy.deepcopy(document)
        altered["metadata"]["authorization"] += " Altered."
        invalid_firewall.append(("altered", altered))
        private_phrase = copy.deepcopy(document)
        private_phrase["metadata"]["note"] = "Swiss participant material"
        invalid_firewall.append(("private_phrase", private_phrase))
        forbidden_key = copy.deepcopy(document)
        forbidden_key["metadata"]["candidate_id"] = "x"
        invalid_firewall.append(("forbidden_key", forbidden_key))
        trainer_value = copy.deepcopy(document)
        trainer_value["metadata"]["note"] = "trainer Alpha"
        invalid_firewall.append(("trainer_value", trainer_value))
        team_key = copy.deepcopy(document)
        team_key["metadata"]["team_id"] = "alpha"
        invalid_firewall.append(("team_key", team_key))
        for label, invalid in invalid_firewall:
            with self.subTest(firewall=label), self.assertRaises(AssertionError):
                _assert_public_data_firewall(invalid)
        self.assertNotEqual(
            CANONICAL_SHA256["1.5.0"],
            hashlib.sha256(_canonical_bytes(altered)).hexdigest(),
        )
        nonpublic = copy.deepcopy(document)
        nonpublic["visibility"] = "private"
        unsupported_source = copy.deepcopy(document)
        unsupported_source["sources"][0]["kind"] = "private"
        for label, invalid in (("visibility", nonpublic), ("source_kind", unsupported_source)):
            with self.subTest(schema=label), self.assertRaises(PublicPriorValidationError):
                validate_public_prior_document(invalid)
        illegal = copy.deepcopy(document)
        low_key = next(record for record in illegal["species"] if record["species_id"] == "toxtricitylowkey")
        low_key["variants"][0]["move_ids"][0] = "spectralthief"
        completed = _run_teamvalidator([{"label": "illegal", "entries": _entries(illegal)}])
        self.assertNotEqual(0, completed.returncode)
        self.assertIn("toxtricitylowkey/", completed.stderr)


class TestBatchFiveSelectionPopulationAndNarrowing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generic_configuration = _startup("1.5.0", PublicPriorFallback.GENERIC)
        cls.none_configuration = _startup("1.5.0", PublicPriorFallback.NONE)
        cls.old_configuration = _startup("1.4.0", PublicPriorFallback.NONE)
        cls.dataset = cls.none_configuration.registry.get(IDENTITY)
        cls.old_dataset = cls.old_configuration.registry.get(
            PublicPriorIdentity("tugspublicarchetypes", "1.4.0", "gen9tugs")
        )

    def test_08_every_new_species_and_all_38_touched_variants_are_selectable(self):
        context = self.none_configuration.create_battle_context("gen9tugs")
        for species_id in NEW_SPECIES_VARIANTS:
            result = select_public_prior_variant(
                context, battle_format="gen9tugs", species_id=species_id,
                level=100, evidence=None, rng=_FixedRng(0.5),
            )
            self.assertIs(PublicPriorSelectionStatus.SELECTED, result.status)
        selected = set()
        for species_id, variant_ids in TOUCHED_VARIANT_IDS.items():
            record = self.dataset.get_species(species_id)
            total = sum(variant.weight for variant in record.variants)
            cumulative = 0.0
            for variant in record.variants:
                rng_value = (cumulative + variant.weight / 2) / total
                result = select_public_prior_variant(
                    context, battle_format="gen9tugs", species_id=species_id,
                    level=100, evidence=None, rng=_FixedRng(rng_value),
                )
                with self.subTest(species=species_id, variant=variant.variant_id):
                    self.assertIs(PublicPriorSelectionStatus.SELECTED, result.status)
                    self.assertIs(variant, result.variant)
                selected.add((species_id, variant.variant_id))
                cumulative += variant.weight
            self.assertEqual(set(variant_ids), {v.variant_id for v in record.variants})
        expected = {
            (species, variant)
            for species, variants in TOUCHED_VARIANT_IDS.items()
            for variant in variants
        }
        self.assertEqual(38, len(selected))
        self.assertEqual(expected, selected)

    def test_09_all_36_unchanged_records_select_identically(self):
        old_context = self.old_configuration.create_battle_context("gen9tugs")
        new_context = self.none_configuration.create_battle_context("gen9tugs")
        for species_id in UNCHANGED_SPECIES:
            for rng_value in (0.0, 0.25, 0.5, 0.75, 0.999999):
                old = select_public_prior_variant(
                    old_context, battle_format="gen9tugs", species_id=species_id,
                    level=100, evidence=None, rng=_FixedRng(rng_value),
                )
                new = select_public_prior_variant(
                    new_context, battle_format="gen9tugs", species_id=species_id,
                    level=100, evidence=None, rng=_FixedRng(rng_value),
                )
                with self.subTest(species=species_id, rng=rng_value):
                    self.assertIs(old.status, new.status)
                    self.assertEqual(old.variant, new.variant)

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
                    self.assertEqual(variant.nature_id, sampled_pokemon.nature)
                    self.assertEqual(variant.evs.as_tuple(), tuple(sampled_pokemon.evs))
                    self.assertEqual(variant.ivs.as_tuple(), tuple(sampled_pokemon.ivs))
                    self.assertEqual(canonical_before, _pokemon_snapshot(pokemon))
                populated.add((species_id, variant_id))
        self.assertEqual(38, len(populated))
        self.assertEqual(
            set(),
            {name for name in set(sys.modules) - modules_before if name.startswith("fp.data.team_pools")},
        )

    def test_11_every_legal_exact_id_selects_without_evidence_and_bypasses_generic(self):
        context = self.none_configuration.create_battle_context("gen9tugs")
        selected = set()
        for species_id in EXPECTED_ROSTER_IDS:
            result = select_public_prior_variant(
                context, battle_format="gen9tugs", species_id=species_id,
                level=100, evidence=None, rng=_FixedRng(0.5),
            )
            self.assertIs(PublicPriorSelectionStatus.SELECTED, result.status)
            selected.add(species_id)
            battle = _battle(context, species_id)
            canonical_before = _pokemon_snapshot(battle.opponent.active)
            with mock.patch("fp.search.standard_battles.sample_pokemon") as generic:
                sampled = prepare_battles(battle, 1)[0][0]
            generic.assert_not_called()
            self.assertEqual(species_id, sampled.opponent.active.name)
            self.assertEqual(canonical_before, _pokemon_snapshot(battle.opponent.active))
        self.assertEqual(EXPECTED_ROSTER_IDS, selected)
        self.assertEqual(49, len(selected))

    def test_12_observation_narrowing_covers_all_thirteen_touched_records(self):
        context = self.none_configuration.create_battle_context("gen9tugs")

        def compatible(species_id, *, moves=(), item=None, ability=None):
            return _compatible_ids(
                self.dataset, species_id,
                _evidence(context, species_id, moves=moves, item=item, ability=ability),
            )

        checks = (
            ("slurpuff", {"sashwebs"}, {"item": "focussash"}),
            ("slurpuff", {"sashwebs"}, {"moves": ("magiccoat",)}),
            ("slurpuff", {"bellydrum"}, {"moves": ("bellydrum",)}),
            ("slurpuff", {"redcardcalmmind"}, {"item": "redcard"}),
            ("slurpuff", set(NEW_SPECIES_VARIANTS["slurpuff"]), {"ability": "unburden"}),
            ("taurospaldeaaqua", {"choiceband"}, {"moves": ("lashout",)}),
            ("taurospaldeaaqua", {"lifeorb"}, {"moves": ("aquajet",)}),
            ("taurospaldeaaqua", {"choicescarf"}, {"item": "choicescarf"}),
            ("taurospaldeaaqua", set(NEW_SPECIES_VARIANTS["taurospaldeaaqua"]), {"moves": ("closecombat", "wavecrash", "earthquake"), "ability": "intimidate"}),
            ("toxtricity", {"scarfsludgebomb"}, {"moves": ("sludgebomb", "overdrive")}),
            ("toxtricity", {"scarfpsychicnoise"}, {"moves": ("psychicnoise",)}),
            ("toxtricity", {"choicespecs"}, {"item": "choicespecs"}),
            ("toxtricity", {"throatspray"}, {"moves": ("shiftgear",)}),
            ("toxtricity", {"scarfsludgebomb", "scarfpsychicnoise"}, {"item": "choicescarf"}),
            ("toxtricitylowkey", {"scarfsludgebomb"}, {"moves": ("sludgebomb", "overdrive")}),
            ("toxtricitylowkey", {"scarfpsychicnoise"}, {"moves": ("psychicnoise",)}),
            ("toxtricitylowkey", {"choicespecs"}, {"item": "choicespecs"}),
            ("toxtricitylowkey", {"throatspray"}, {"item": "throatspray"}),
            ("toxtricitylowkey", {"scarfsludgebomb", "scarfpsychicnoise"}, {"item": "choicescarf"}),
            ("vileplume", {"effectsporephysical"}, {"moves": ("strengthsap",)}),
            ("vileplume", {"chlorophyllspdef"}, {"ability": "chlorophyll"}),
            ("vileplume", set(NEW_SPECIES_VARIANTS["vileplume"]), {"moves": ("sludgebomb",)}),
            ("weezing", {"levitatephysical"}, {"ability": "levitate"}),
            ("weezing", {"neutralizinggasphysical"}, {"moves": ("taunt",)}),
            ("weezing", set(NEW_SPECIES_VARIANTS["weezing"]), {"moves": ("sludgebomb", "painsplit")}),
            ("xatu", {"helmetphysical"}, {"moves": ("psychic",)}),
            ("xatu", {"leftoversutility"}, {"moves": ("nightshade",)}),
            ("xatu", {"bootsutility"}, {"moves": ("defog",)}),
            ("xatu", {"helmetphysical", "leftoversutility"}, {"moves": ("teleport",)}),
            ("xatu", {"leftoversutility", "bootsutility"}, {"moves": ("thunderwave",)}),
            ("starmie", {"lifeorbanalytic"}, {"item": "lifeorb"}),
            ("starmie", {"bootsnaturalcure"}, {"ability": "naturalcure"}),
            ("starmie", {"choicespecs"}, {"moves": ("psyshock",)}),
            ("swampert", {"spdefrocks"}, {"moves": ("knockoff",), "item": "leftovers"}),
            ("swampert", {"offensiveyawn"}, {"moves": ("yawn",)}),
            ("swampert", {"chestorest"}, {"moves": ("toxic",)}),
            ("sylveon", {"spdefwish"}, {"moves": ("mysticalfire", "protect")}),
            ("sylveon", {"defensivecalmmind"}, {"moves": ("calmmind",)}),
            ("sylveon", {"toxicwish"}, {"moves": ("toxic",)}),
            ("sylveon", {"bootscleric"}, {"item": "heavydutyboots"}),
            ("sylveon", set(REPLACEMENT_VARIANTS["sylveon"]), {"moves": ("wish", "hypervoice")}),
            ("tinkaton", {"defensiveutility"}, {"moves": ("encore",)}),
            ("tinkaton", {"sashsetup"}, {"moves": ("swordsdance",)}),
            ("tinkaton", set(REPLACEMENT_VARIANTS["tinkaton"]), {"moves": ("gigatonhammer", "knockoff"), "ability": "moldbreaker"}),
            ("torkoal", {"heatrockwillowisp"}, {"item": "heatrock", "moves": ("willowisp",)}),
            ("torkoal", {"heatrockrest"}, {"item": "heatrock", "moves": ("rest",)}),
            ("torkoal", {"bootsutility"}, {"moves": ("bodypress",)}),
            ("torkoal", set(REPLACEMENT_VARIANTS["torkoal"]), {"moves": ("lavaplume", "rapidspin", "stealthrock"), "ability": "drought"}),
            ("vikavolt", {"sashwebs"}, {"moves": ("energyball",)}),
            ("vikavolt", {"bulkyboots"}, {"moves": ("roost",)}),
            ("vikavolt", set(REPLACEMENT_VARIANTS["vikavolt"]), {"moves": ("stickyweb", "voltswitch", "bugbuzz"), "ability": "levitate"}),
        )
        self.assertEqual(set(TOUCHED_VARIANT_IDS), {species for species, _, _ in checks})
        for species_id, expected, kwargs in checks:
            with self.subTest(species=species_id, expected=expected, evidence=kwargs):
                self.assertEqual(expected, compatible(species_id, **kwargs))

    def test_13_removed_variants_and_toxtricity_exact_forms_are_isolated(self):
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
        self.assertEqual(set(), old_ids & new_ids)
        self.assertEqual(REMOVED_VARIANT_IDS, old_ids - new_ids)
        for species_id, variant_id in REMOVED_VARIANT_IDS:
            self.assertIsNone(self.dataset.get_species(species_id).get_variant(variant_id))
            self.assertNotIn(variant_id, REPLACEMENT_VARIANTS[species_id])
        amped = self.dataset.get_species("toxtricity")
        low_key = self.dataset.get_species("toxtricitylowkey")
        self.assertIsNot(amped, low_key)
        self.assertEqual("toxtricity", amped.species_id)
        self.assertEqual("toxtricitylowkey", low_key.species_id)
        self.assertEqual(amped.variants, low_key.variants)
        context = self.none_configuration.create_battle_context("gen9tugs")
        amped_result = select_public_prior_variant(
            context, battle_format="gen9tugs", species_id="toxtricity",
            level=100, evidence=_evidence(context, "toxtricity", moves=("shiftgear",)),
            rng=_FixedRng(0.5),
        )
        low_key_result = select_public_prior_variant(
            context, battle_format="gen9tugs", species_id="toxtricitylowkey",
            level=100, evidence=_evidence(context, "toxtricitylowkey", moves=("shiftgear",)),
            rng=_FixedRng(0.5),
        )
        self.assertEqual("throatspray", amped_result.variant.variant_id)
        self.assertEqual("throatspray", low_key_result.variant.variant_id)

    def test_14_forretress_toxic_spikes_uses_generic_fallback_without_splicing(self):
        context = self.generic_configuration.create_battle_context("gen9tugs")
        evidence = _evidence(context, "forretress", moves=("toxicspikes",))
        self.assertEqual(set(), _compatible_ids(self.dataset, "forretress", evidence))
        for variant in self.dataset.get_species("forretress").variants:
            self.assertNotIn("toxicspikes", variant.move_ids)
        result = select_public_prior_variant(
            context, battle_format="gen9tugs", species_id="forretress",
            level=100, evidence=evidence, rng=_FixedRng(0.5),
        )
        self.assertIs(PublicPriorSelectionStatus.NO_COMPATIBLE_VARIANT, result.status)
        battle = _battle(context, "forretress")
        battle.opponent.active.add_move("toxicspikes")
        battle.team_inference.record_selected_move("forretress", "toxicspikes")
        canonical_before = _pokemon_snapshot(battle.opponent.active)
        with mock.patch("fp.search.standard_battles.sample_pokemon") as generic:
            sampled = prepare_battles(battle, 1)[0][0]
        generic.assert_called_once()
        sampled_pokemon = sampled.opponent.active
        self.assertIn("toxicspikes", tuple(move.name for move in sampled_pokemon.moves))
        self.assertEqual(constants.UNKNOWN_ITEM, sampled_pokemon.item)
        self.assertEqual(canonical_before, _pokemon_snapshot(battle.opponent.active))

    def test_15_dudunsparce_later_item_recomputes_and_none_fallback_queries_nothing(self):
        context = self.none_configuration.create_battle_context("gen9tugs")
        calm_mind = _evidence(context, "dudunsparce", moves=("calmmind",))
        self.assertEqual({"calmmind"}, _compatible_ids(self.dataset, "dudunsparce", calm_mind))
        battle = _battle(context, "dudunsparce")
        battle.opponent.active.add_move("calmmind")
        battle.team_inference.record_selected_move("dudunsparce", "calmmind")
        canonical_before = _pokemon_snapshot(battle.opponent.active)
        with mock.patch("fp.search.standard_battles.sample_pokemon") as generic:
            first = prepare_battles(battle, 1)[0][0]
        generic.assert_not_called()
        self.assertEqual("heavydutyboots", first.opponent.active.item)
        self.assertEqual(canonical_before, _pokemon_snapshot(battle.opponent.active))
        battle.opponent.active.item = "leftovers"
        battle.team_inference.record_initial_item(
            "dudunsparce", "leftovers",
            source=PublicObservationSource.DIRECT_ITEM_REVEAL,
        )
        later = battle.team_inference.observation_ledger.member("dudunsparce")
        self.assertEqual(set(), _compatible_ids(self.dataset, "dudunsparce", later))
        with mock.patch("fp.search.standard_battles.sample_pokemon") as generic:
            second = prepare_battles(battle, 1)[0][0]
        generic.assert_not_called()
        self.assertEqual("leftovers", second.opponent.active.item)
        self.assertNotEqual("heavydutyboots", second.opponent.active.item)
        self.assertIn("calmmind", tuple(move.name for move in second.opponent.active.moves))
        self.assertNotIn("coil", tuple(move.name for move in second.opponent.active.moves))
        json.dumps(second.request_json or {})

    def test_16_public_sampling_has_no_private_pool_or_candidate_dependency(self):
        paths = (
            ROOT / "fp" / "battle" / "public_prior_context.py",
            ROOT / "fp" / "data" / "public_priors" / "runtime.py",
            ROOT / "fp" / "search" / "public_prior_sampling.py",
            ROOT / "fp" / "search" / "standard_battles.py",
        )
        forbidden = (
            "fp.data.team_pools", "TeamPool", "TeamRecord", "PokemonRecord",
            "baseline_candidate_ids", "active_candidate_ids",
        )
        for path in paths:
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, source)
        parameters = inspect.signature(select_public_prior_variant).parameters
        self.assertNotIn("candidate_id", parameters)
        self.assertNotIn("candidate_ids", parameters)
        self.assertEqual(49, len(self.dataset.species))


if __name__ == "__main__":
    unittest.main()
