import copy
import hashlib
import json
import unittest
from pathlib import Path

from fp.battle.public_prior_context import (
    PublicPriorFallback,
    PublicPriorSearchContext,
)
from fp.config import FoulPlayConfig
from fp.data import all_move_json, pokedex
from fp.data.mods.apply_mods import apply_mods
from fp.data.public_priors import PublicPriorRegistry, load_public_prior
from fp.data.public_priors.runtime import (
    PublicPriorStartupOptions,
    load_public_prior_runtime_configuration,
)
from fp.format_spec import FormatSpec
from fp.search.public_prior_sampling import (
    PublicPriorSelectionStatus,
    _compatible_family_candidates,
    select_public_prior_variant,
)

from .test_public_prior_production_batch3_data import (
    _FixedRng,
    _assert_public_data_firewall,
    _canonical_bytes,
    _entries,
    _evidence,
    _run_teamvalidator,
)


ROOT = Path(__file__).resolve().parents[1]
POOLS = ROOT / "fp" / "data" / "public_priors" / "pools"
OLD_PATH = POOLS / "tugspublicarchetypes-1.5.0.json"
NEW_PATH = POOLS / "tugspublicarchetypes-1.6.0.json"
OLD_RAW_SHA256 = "a3bf5c0748603973968d85f2dfead1948d9bb619d1dc1be727b1770c1023ea93"
NEW_CANONICAL_SHA256 = (
    "375a6a9368dce9fc9a1d13e068d93eb4c605f2f099787b6fd6ff672c48098062"
)

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
        cls.document = json.loads(NEW_PATH.read_text(encoding="utf-8", errors="strict"))
        cls.configuration = load_public_prior_runtime_configuration(
            PublicPriorStartupOptions(
                (str(NEW_PATH), str(OLD_PATH)),
                PublicPriorFallback.NONE,
            ),
            "gen9tugs",
        )

    def test_01_identity_inventory_and_public_source_are_exact(self):
        self.assertEqual(1, self.new.schema_version)
        self.assertEqual("public", self.new.visibility)
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
                with self.subTest(
                    species=species.species_id, variant=variant.variant_id
                ):
                    self.assertEqual(100, variant.level)
                    self.assertEqual(4, len(variant.move_ids))
                    self.assertEqual(4, len(set(variant.move_ids)))
                    self.assertEqual(3.0, variant.weight)
                    self.assertEqual(("publicsetpriors2",), variant.source_ids)
                    self.assertLessEqual(sum(variant.evs.as_tuple()), 510)
                    self.assertTrue(
                        all(0 <= value <= 252 for value in variant.evs.as_tuple())
                    )
                    self.assertTrue(
                        all(0 <= value <= 31 for value in variant.ivs.as_tuple())
                    )

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
        self.assertEqual(
            len(self.old.get_species("aerodactyl").variants), len(candidates)
        )
        self.assertTrue(
            all(identity == self.old.identity for identity, _ in candidates)
        )

    def test_07_committed_base_hash_and_delta_canonical_hash_are_exact(self):
        self.assertEqual(
            OLD_RAW_SHA256,
            hashlib.sha256(OLD_PATH.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            NEW_CANONICAL_SHA256,
            hashlib.sha256(_canonical_bytes(self.document)).hexdigest(),
        )

    def test_08_runtime_preserves_new_then_base_layer_order(self):
        self.assertEqual(
            (self.new.identity, self.old.identity),
            self.configuration.selected_identities,
        )
        first = self.configuration.create_battle_context("gen9tugs")
        second = self.configuration.create_battle_context("gen9tugs")
        self.assertIsNot(first, second)
        self.assertIs(first.registry, second.registry)
        self.assertIs(first, copy.deepcopy(first))

    def test_09_new_and_old_same_species_variants_are_both_selectable(self):
        context = self.configuration.create_battle_context("gen9tugs")
        new_evidence = _evidence(
            context,
            "mawile",
            moves=("suckerpunch", "flamethrower", "ironhead", "playrough"),
            item="focussash",
            ability="closingjaws",
        )
        old_evidence = _evidence(
            context,
            "mawile",
            moves=("knockoff", "ironhead", "flamethrower", "playrough"),
            item="focussash",
            ability="closingjaws",
        )

        new_result = select_public_prior_variant(
            context,
            battle_format="gen9tugs",
            species_id="mawile",
            level=100,
            evidence=new_evidence,
            rng=_FixedRng(0),
        )
        old_result = select_public_prior_variant(
            context,
            battle_format="gen9tugs",
            species_id="mawile",
            level=100,
            evidence=old_evidence,
            rng=_FixedRng(0),
        )

        self.assertIs(PublicPriorSelectionStatus.SELECTED, new_result.status)
        self.assertEqual(self.new.identity, new_result.dataset_identity)
        self.assertEqual("sashmixednaughty", new_result.variant.variant_id)
        self.assertIs(PublicPriorSelectionStatus.SELECTED, old_result.status)
        self.assertEqual(self.old.identity, old_result.dataset_identity)
        self.assertEqual("sashmixed", old_result.variant.variant_id)

    def test_10_production_family_selection_uses_authored_weights_from_both_layers(
        self,
    ):
        context = self.configuration.create_battle_context("gen9tugs")
        before = {
            (identity, variant.variant_id): variant.weight
            for identity, variant in _compatible_family_candidates(
                context,
                context.selected_identities,
                battle_format="gen9tugs",
                species_id="mawile",
                level=100,
                evidence=None,
            )
        }

        first = select_public_prior_variant(
            context,
            battle_format="gen9tugs",
            species_id="mawile",
            level=100,
            evidence=None,
            rng=_FixedRng(0),
        )
        last = select_public_prior_variant(
            context,
            battle_format="gen9tugs",
            species_id="mawile",
            level=100,
            evidence=None,
            rng=_FixedRng(0.99),
        )

        self.assertEqual(
            (self.old.identity, "choiceband"),
            (first.dataset_identity, first.variant.variant_id),
        )
        self.assertEqual(
            (self.new.identity, "sashmixednaughty"),
            (last.dataset_identity, last.variant.variant_id),
        )
        self.assertEqual(
            before,
            {
                (identity, variant.variant_id): variant.weight
                for identity, variant in _compatible_family_candidates(
                    context,
                    context.selected_identities,
                    battle_format="gen9tugs",
                    species_id="mawile",
                    level=100,
                    evidence=None,
                )
            },
        )

    def test_11_public_data_firewall_accepts_only_public_delta_content(self):
        _assert_public_data_firewall(self.document)

    def test_12_trusted_teamvalidator_accepts_all_delta_variants(self):
        completed = _run_teamvalidator(
            [{"label": "1.6.0", "entries": _entries(self.document)}]
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertEqual(["1.6.0:54/54"], completed.stdout.splitlines())


if __name__ == "__main__":
    unittest.main()
