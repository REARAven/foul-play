import copy
import unittest
from pathlib import Path

from fp.config import FoulPlayConfig
from fp.data import all_move_json, pokedex
from fp.data.mods.apply_mods import apply_mods
from fp.data.public_priors import PublicPriorRegistry, load_public_prior
from fp.battle.public_prior_context import (
    PublicPriorFallback,
    PublicPriorSearchContext,
)
from fp.format_spec import FormatSpec
from fp.search.public_prior_sampling import _compatible_family_candidates


ROOT = Path(__file__).resolve().parents[1]
POOLS = ROOT / "fp" / "data" / "public_priors" / "pools"
OLD_PATH = POOLS / "tugspublicarchetypes-1.5.0.json"
NEW_PATH = POOLS / "tugspublicarchetypes-1.6.0.json"

ORIGINAL_MOVES = copy.deepcopy(all_move_json)
ORIGINAL_POKEDEX = copy.deepcopy(pokedex)
ORIGINAL_FORMAT = FoulPlayConfig.pokemon_format


def setUpModule():
    FoulPlayConfig.pokemon_format = "gen9tugs"
    apply_mods(FormatSpec.from_format_string("gen9tugs"))


def tearDownModule():
    all_move_json.clear()
    all_move_json.update(ORIGINAL_MOVES)
    pokedex.clear()
    pokedex.update(ORIGINAL_POKEDEX)
    FoulPlayConfig.pokemon_format = ORIGINAL_FORMAT


class TestBatchSixPublicPriorAdditions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = load_public_prior(OLD_PATH)
        cls.new = load_public_prior(NEW_PATH)

    def test_01_identity_inventory_and_public_source_are_exact(self):
        self.assertEqual("tugspublicarchetypes", self.new.identity.dataset_id)
        self.assertEqual("1.6.0", self.new.identity.dataset_version)
        self.assertEqual("gen9tugs", self.new.identity.format_id)
        self.assertEqual("1.2", self.new.patch_version)
        self.assertEqual(32, len(self.new.species))
        self.assertEqual(54, sum(len(record.variants) for record in self.new.species))
        self.assertEqual(1, len(self.new.sources))
        self.assertEqual("publicsetpriors2", self.new.sources[0].source_id)
        self.assertEqual("2026-08-17", self.new.sources[0].public_date)

    def test_02_all_added_variants_are_complete_and_ev_legal(self):
        for species in self.new.species:
            for variant in species.variants:
                with self.subTest(species=species.species_id, variant=variant.variant_id):
                    self.assertEqual(100, variant.level)
                    self.assertEqual(4, len(variant.move_ids))
                    self.assertEqual(4, len(set(variant.move_ids)))
                    self.assertEqual(("publicsetpriors2",), variant.source_ids)
                    self.assertLessEqual(sum(variant.evs.as_tuple()), 510)
                    self.assertTrue(all(0 <= value <= 252 for value in variant.evs.as_tuple()))
                    self.assertTrue(all(0 <= value <= 31 for value in variant.ivs.as_tuple()))

    def test_03_corrected_focus_sash_mawile_spread_is_508_evs(self):
        mawile = self.new.get_species("mawile")
        variant = mawile.get_variant("sashmixednaughty")
        self.assertEqual("focussash", variant.item_id)
        self.assertEqual("closingjaws", variant.base_ability_id)
        self.assertEqual("naughty", variant.nature_id)
        self.assertEqual((0, 252, 0, 4, 0, 252), variant.evs.as_tuple())
        self.assertEqual(508, sum(variant.evs.as_tuple()))
        self.assertEqual(
            ("suckerpunch", "flamethrower", "ironhead", "playrough"),
            variant.move_ids,
        )

    def test_04_delta_variant_ids_do_not_replace_1_5_variants_accidentally(self):
        old_ids = {
            (species.species_id, variant.variant_id)
            for species in self.old.species
            for variant in species.variants
        }
        new_ids = {
            (species.species_id, variant.variant_id)
            for species in self.new.species
            for variant in species.variants
        }
        self.assertEqual(set(), old_ids & new_ids)
        self.assertEqual(153, len(old_ids))
        self.assertEqual(54, len(new_ids))
        self.assertEqual(207, len(old_ids | new_ids))

    def test_05_same_dataset_family_layers_old_and_new_variants(self):
        context = PublicPriorSearchContext(
            registry=PublicPriorRegistry((self.old, self.new)),
            selected_identities=(self.new.identity, self.old.identity),
            fallback_policy=PublicPriorFallback.NONE,
            format_id="gen9tugs",
        )
        candidates = _compatible_family_candidates(
            context,
            context.selected_identities,
            battle_format="gen9tugs",
            species_id="mawile",
            level=100,
            evidence=None,
        )
        old_mawile = self.old.get_species("mawile")
        new_mawile = self.new.get_species("mawile")
        self.assertEqual(
            len(old_mawile.variants) + len(new_mawile.variants), len(candidates)
        )
        identities = {identity for identity, _ in candidates}
        self.assertEqual({self.old.identity, self.new.identity}, identities)

    def test_06_untouched_species_falls_through_to_1_5_layer(self):
        context = PublicPriorSearchContext(
            registry=PublicPriorRegistry((self.old, self.new)),
            selected_identities=(self.new.identity, self.old.identity),
            fallback_policy=PublicPriorFallback.NONE,
            format_id="gen9tugs",
        )
        candidates = _compatible_family_candidates(
            context,
            context.selected_identities,
            battle_format="gen9tugs",
            species_id="aerodactyl",
            level=100,
            evidence=None,
        )
        self.assertEqual(len(self.old.get_species("aerodactyl").variants), len(candidates))
        self.assertTrue(all(identity == self.old.identity for identity, _ in candidates))


if __name__ == "__main__":
    unittest.main()
