import copy
import inspect
import json
import math
import socket
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from types import MappingProxyType
from unittest import mock

from fp.config import FoulPlayConfig
from fp.data import all_move_json, pokedex
from fp.data.public_priors import (
    PublicPriorDataset,
    PublicPriorIdentity,
    PublicPriorRegistry,
    PublicPriorRegistryError,
    PublicPriorValidationError,
    PublicSetVariant,
    PublicSourceKind,
    PublicVariantReference,
    SpeciesPrior,
    load_public_prior,
)


FIXTURE_TEMP_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "fp" / "data" / "public_priors"


def _variant(variant_id="standard", weight=2.0, **overrides):
    value = {
        "variant_id": variant_id,
        "weight": weight,
        "item_id": "lightball",
        "base_ability_id": "static",
        "move_ids": ["thunderbolt", "voltswitch", "surf", "protect"],
        "nature_id": "timid",
        "evs": {"hp": 0, "atk": 0, "def": 0, "spa": 252, "spd": 4, "spe": 252},
        "ivs": {"hp": 31, "atk": 0, "def": 31, "spa": 31, "spd": 31, "spe": 31},
        "level": 50,
        "source_ids": ["syntheticfixture"],
        "metadata": {"nested": {"labels": ["synthetic"]}},
    }
    value.update(overrides)
    return value


def _species(species_id="pikachu", **overrides):
    value = {
        "species_id": species_id,
        "variants": [_variant()],
        "metadata": {"scope": {"exact_form": True}},
    }
    value.update(overrides)
    return value


def _document(**overrides):
    value = {
        "schema_version": 1,
        "visibility": "public",
        "dataset_id": "syntheticprior",
        "dataset_version": "1.0",
        "format_id": "gen9tugs",
        "patch_version": "synthetic-patch",
        "display_name": "Synthetic public-prior fixture",
        "metadata": {"nested": {"labels": ["synthetic", "public"]}},
        "sources": [
            {
                "source_id": "syntheticfixture",
                "kind": "synthetic_test",
                "public_date": "2099-01-01",
                "description": "Synthetic test data only",
                "metadata": {"nested": {"approved": True}},
            }
        ],
        "species": [_species()],
    }
    value.update(overrides)
    return value


def _write_document(directory, document, name="public-prior.json"):
    path = Path(directory) / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _load_document(document=None):
    with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
        return load_public_prior(_write_document(directory, document or _document()))


def _validation_error(document):
    with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
        with unittest.TestCase().assertRaises(PublicPriorValidationError) as raised:
            load_public_prior(_write_document(directory, document))
    return raised.exception


def _error_text(document):
    return str(_validation_error(document))


def _second_dataset():
    document = _document(
        dataset_id="secondsyntheticprior",
        dataset_version="2.0",
        sources=[{"source_id": "syntheticfixture", "kind": "synthetic_test"}],
    )
    return _load_document(document)


class TestPublicPriorLoader(unittest.TestCase):
    def test_01_valid_schema_version_one_public_prior_loads(self):
        dataset = _load_document()
        self.assertEqual(1, dataset.schema_version)
        self.assertEqual("public", dataset.visibility)
        self.assertEqual("syntheticprior", dataset.identity.dataset_id)

    def test_02_visibility_must_be_exactly_public(self):
        for value in ("private", "closed", "Public", ""):
            with self.subTest(value=value):
                self.assertIn("visibility must be exactly 'public'", _error_text(_document(visibility=value)))

    def test_03_unsupported_schema_version_is_rejected(self):
        self.assertIn("unsupported schema version 2", _error_text(_document(schema_version=2)))

    def test_04_loaded_dataset_is_immutable(self):
        dataset = _load_document()
        with self.assertRaises(FrozenInstanceError):
            dataset.visibility = "private"

    def test_05_nested_sources_species_and_variants_are_immutable(self):
        dataset = _load_document()
        with self.assertRaises(FrozenInstanceError):
            dataset.sources[0].description = "changed"
        with self.assertRaises(FrozenInstanceError):
            dataset.species[0].species_id = "raichu"
        with self.assertRaises(FrozenInstanceError):
            dataset.species[0].variants[0].weight = 9

    def test_06_metadata_is_recursively_immutable(self):
        dataset = _load_document()
        self.assertIsInstance(dataset.metadata, MappingProxyType)
        self.assertIsInstance(dataset.metadata["nested"], MappingProxyType)
        self.assertIsInstance(dataset.metadata["nested"]["labels"], tuple)
        self.assertIsInstance(dataset.sources[0].metadata["nested"], MappingProxyType)
        self.assertIsInstance(
            dataset.species[0].variants[0].metadata["nested"], MappingProxyType
        )
        with self.assertRaises(TypeError):
            dataset.metadata["nested"]["new"] = True

    def test_07_one_species_may_contain_multiple_coherent_variants(self):
        species = _species(variants=[_variant("special"), _variant("physical", item_id="none", move_ids=["quickattack", "irontail", "nuzzle", "protect"])])
        record = _load_document(_document(species=[species])).get_species("pikachu")
        self.assertEqual(("physical", "special"), tuple(v.variant_id for v in record.variants))
        self.assertEqual("none", record.variants[0].item_id)
        self.assertEqual("lightball", record.variants[1].item_id)

    def test_08_variant_order_is_deterministic(self):
        variants = [_variant("zeta"), _variant("alpha")]
        first = _load_document(_document(species=[_species(variants=variants)]))
        second = _load_document(_document(species=[_species(variants=list(reversed(variants)))]))
        self.assertEqual(tuple(v.variant_id for v in first.species[0].variants), tuple(v.variant_id for v in second.species[0].variants))

    def test_09_species_order_is_deterministic(self):
        raichu = _species("raichu")
        pikachu = _species("pikachu")
        dataset = _load_document(_document(species=[raichu, pikachu]))
        self.assertEqual(("pikachu", "raichu"), tuple(record.species_id for record in dataset.species))

    def test_10_exact_forms_remain_distinct(self):
        wash = _species("rotomwash", variants=[_variant(base_ability_id="levitate")])
        heat = _species("rotomheat", variants=[_variant(base_ability_id="levitate")])
        dataset = _load_document(_document(species=[wash, heat]))
        self.assertIsNot(dataset.get_species("rotomwash"), dataset.get_species("rotomheat"))
        self.assertIsNone(dataset.get_species("rotom"))

    def test_11_dudunsparce_forms_remain_distinct(self):
        two = _species("dudunsparce", variants=[_variant(base_ability_id="serenegrace")])
        three = _species("dudunsparcethreesegment", base_species_id="dudunsparce", variants=[_variant(base_ability_id="serenegrace")])
        dataset = _load_document(_document(species=[three, two]))
        self.assertEqual(("dudunsparce", "dudunsparcethreesegment"), tuple(record.species_id for record in dataset.species))

    def test_12_public_dataset_identity_is_deterministic(self):
        identity = _load_document().identity
        self.assertEqual(PublicPriorIdentity("syntheticprior", "1.0", "gen9tugs"), identity)
        self.assertEqual(hash(identity), hash(PublicPriorIdentity("syntheticprior", "1.0", "gen9tugs")))

    def test_13_registry_holds_multiple_datasets(self):
        registry = PublicPriorRegistry((_second_dataset(), _load_document()))
        self.assertEqual(2, len(registry))
        self.assertIsNotNone(registry.get(PublicPriorIdentity("secondsyntheticprior", "2.0", "gen9tugs")))

    def test_14_duplicate_registry_identity_is_rejected(self):
        dataset = _load_document()
        with self.assertRaises(PublicPriorRegistryError):
            PublicPriorRegistry((dataset, dataset))

    def test_15_dataset_lookup_is_deterministic(self):
        first = _load_document()
        second = _second_dataset()
        registry = PublicPriorRegistry((second, first))
        self.assertIs(first, registry.get(first.identity))
        self.assertIs(second, registry.get(second.identity))

    def test_16_species_lookup_is_exact_form_only(self):
        dataset = _load_document(_document(species=[_species("rotomwash", variants=[_variant(base_ability_id="levitate")])]))
        self.assertIsNotNone(dataset.get_species("rotomwash"))
        self.assertIsNone(dataset.get_species("rotom"))
        self.assertIsNone(dataset.get_species("Rotom-Wash"))

    def test_17_unknown_species_lookup_returns_no_variants(self):
        dataset = _load_document()
        self.assertEqual((), dataset.variant_references("mew"))
        self.assertEqual((), dataset.normalized_probabilities("mew"))

    def test_18_dataset_qualified_variant_references_are_immutable(self):
        reference = _load_document().variant_references("pikachu")[0]
        self.assertEqual("syntheticprior", reference.dataset_identity.dataset_id)
        with self.assertRaises(FrozenInstanceError):
            reference.variant_id = "changed"

    def test_19_references_do_not_use_team_record_id(self):
        from fp.battle.team_inference import TeamPoolCandidateId

        names = {field.name for field in fields(PublicVariantReference)}
        self.assertEqual({"dataset_identity", "species_id", "variant_id"}, names)
        self.assertNotIn("TeamRecordId", inspect.getsource(PublicVariantReference))
        private_candidate = object.__new__(TeamPoolCandidateId)
        with self.assertRaises(TypeError):
            PublicVariantReference(private_candidate, "pikachu", "standard")

    def test_20_reference_team_record_cannot_be_inserted_as_public_variant(self):
        from fp.data.team_pools import TeamPool, TeamRecord

        private_record = object.__new__(TeamRecord)
        with self.assertRaises(TypeError):
            SpeciesPrior("pikachu", None, (private_record,), {})
        private_pool = object.__new__(TeamPool)
        with self.assertRaises(TypeError):
            PublicPriorRegistry((private_pool,))

    def test_21_missing_file_is_reported(self):
        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
            with self.assertRaises(PublicPriorValidationError) as raised:
                load_public_prior(Path(directory) / "missing.json")
        self.assertIn("does not exist", str(raised.exception))

    def test_22_invalid_utf8_is_reported(self):
        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
            path = Path(directory) / "invalid.json"
            path.write_bytes(b"\xff\xfe")
            with self.assertRaises(PublicPriorValidationError) as raised:
                load_public_prior(path)
        self.assertIn("not valid UTF-8", str(raised.exception))

    def test_23_malformed_json_is_reported(self):
        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
            path = Path(directory) / "malformed.json"
            path.write_text("{broken", encoding="utf-8")
            with self.assertRaises(PublicPriorValidationError) as raised:
                load_public_prior(path)
        self.assertIn("malformed JSON", str(raised.exception))

    def test_24_duplicate_json_object_field_is_reported(self):
        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
            with self.assertRaises(PublicPriorValidationError) as raised:
                load_public_prior(path)
        self.assertIn("duplicate JSON field 'schema_version'", str(raised.exception))

    def test_25_wrong_top_level_type_is_reported(self):
        self.assertIn("$: expected an object", _error_text([]))

    def test_26_missing_required_document_fields_are_aggregated(self):
        error = _validation_error({"schema_version": 1})
        paths = {issue.path for issue in error.issues}
        self.assertTrue({"$.visibility", "$.dataset_id", "$.dataset_version", "$.format_id", "$.sources", "$.species"}.issubset(paths))

    def test_27_unexpected_structured_fields_are_rejected(self):
        document = _document(unexpected={"private": ["record"]})
        self.assertIn("$.unexpected: unexpected field", _error_text(document))

    def test_28_empty_dataset_is_rejected(self):
        self.assertIn("at least one exact species prior", _error_text(_document(species=[])))

    def test_29_empty_source_list_is_rejected(self):
        self.assertIn("at least one public source", _error_text(_document(sources=[])))

    def test_30_duplicate_source_id_is_rejected(self):
        source = {"source_id": "syntheticfixture", "kind": "synthetic_test"}
        self.assertIn("duplicate source ID", _error_text(_document(sources=[source, source.copy()])))

    def test_31_noncanonical_source_id_is_rejected(self):
        source = {"source_id": "Synthetic-Fixture", "kind": "synthetic_test"}
        self.assertIn("expected canonical ID 'syntheticfixture'", _error_text(_document(sources=[source])))

    def test_32_unsupported_and_private_source_kinds_are_rejected(self):
        private_kinds = ("private_team_sheet", "closed_team_sheet", "reference_pool", "submitted_team", "private_replay", "trainer_private_data")
        for kind in private_kinds:
            with self.subTest(kind=kind):
                source = {"source_id": "syntheticfixture", "kind": kind}
                self.assertIn("unsupported source kind", _error_text(_document(sources=[source])))

    def test_33_unknown_source_reference_from_variant_is_rejected(self):
        species = _species(variants=[_variant(source_ids=["unknownsource"])])
        self.assertIn("unknown public source reference", _error_text(_document(species=[species])))

    def test_34_duplicate_source_reference_is_rejected(self):
        species = _species(variants=[_variant(source_ids=["syntheticfixture", "syntheticfixture"])])
        self.assertIn("duplicate source reference", _error_text(_document(species=[species])))

    def test_35_unknown_species_form_is_rejected(self):
        self.assertIn("unknown exact species/form ID", _error_text(_document(species=[_species("syntheticunknownmon")])))

    def test_36_noncanonical_species_is_rejected_not_rewritten(self):
        error = _error_text(_document(species=[_species("Pikachu")]))
        self.assertIn("expected canonical ID 'pikachu', received 'Pikachu'", error)

    def test_37_base_species_mismatch_is_rejected(self):
        species = _species("rotomwash", base_species_id="pikachu", variants=[_variant(base_ability_id="levitate")])
        self.assertIn("must match authoritative base species 'rotom'", _error_text(_document(species=[species])))

    def test_38_empty_species_variant_collection_is_rejected(self):
        self.assertIn("at least one public variant", _error_text(_document(species=[_species(variants=[])])))

    def test_39_duplicate_variant_id_is_rejected(self):
        species = _species(variants=[_variant("same"), _variant("same")])
        self.assertIn("duplicate variant ID", _error_text(_document(species=[species])))

    def test_40_noncanonical_variant_id_is_rejected(self):
        self.assertIn("expected canonical ID 'standardset'", _error_text(_document(species=[_species(variants=[_variant("Standard-Set")])])))

    def test_41_zero_weight_is_rejected(self):
        self.assertIn("strictly greater than zero", _error_text(_document(species=[_species(variants=[_variant(weight=0)])])))

    def test_42_negative_weight_is_rejected(self):
        self.assertIn("strictly greater than zero", _error_text(_document(species=[_species(variants=[_variant(weight=-1)])])))

    def test_43_nan_weight_is_rejected(self):
        self.assertIn("finite", _error_text(_document(species=[_species(variants=[_variant(weight=float("nan"))])])))

    def test_44_infinite_weight_is_rejected(self):
        for value in (float("inf"), float("-inf")):
            with self.subTest(value=value):
                self.assertIn("finite", _error_text(_document(species=[_species(variants=[_variant(weight=value)])])))

    def test_45_boolean_weight_is_rejected(self):
        self.assertIn("expected a number", _error_text(_document(species=[_species(variants=[_variant(weight=True)])])))

    def test_46_positive_finite_weights_are_accepted(self):
        variants = [_variant("tiny", weight=0.001), _variant("large", weight=1e300)]
        record = _load_document(_document(species=[_species(variants=variants)])).get_species("pikachu")
        self.assertEqual((1e300, 0.001), tuple(variant.weight for variant in record.variants))

    def test_47_derived_normalized_probabilities_sum_to_one(self):
        variants = [_variant("alpha", weight=1), _variant("beta", weight=3)]
        probabilities = _load_document(_document(species=[_species(variants=variants)])).normalized_probabilities("pikachu")
        self.assertTrue(math.isclose(1.0, sum(entry.probability for entry in probabilities)))
        self.assertEqual((0.25, 0.75), tuple(entry.probability for entry in probabilities))
        with self.assertRaises(FrozenInstanceError):
            probabilities[0].probability = 1.0

    def test_48_normalization_does_not_mutate_source_weights(self):
        dataset = _load_document(_document(species=[_species(variants=[_variant("a", 2), _variant("b", 5)])]))
        before = tuple(variant.weight for variant in dataset.species[0].variants)
        dataset.normalized_probabilities("pikachu")
        self.assertEqual(before, tuple(variant.weight for variant in dataset.species[0].variants))

    def test_49_exactly_four_moves_are_required(self):
        for moves in (["protect"], ["protect", "surf", "voltswitch", "thunderbolt", "nuzzle"]):
            with self.subTest(count=len(moves)):
                species = _species(variants=[_variant(move_ids=moves)])
                self.assertIn("expected exactly 4 entries", _error_text(_document(species=[species])))

    def test_50_duplicate_moves_are_rejected(self):
        species = _species(variants=[_variant(move_ids=["protect", "protect", "surf", "thunderbolt"])])
        self.assertIn("duplicate move ID 'protect'", _error_text(_document(species=[species])))

    def test_51_unknown_move_is_rejected(self):
        species = _species(variants=[_variant(move_ids=["protect", "surf", "thunderbolt", "syntheticunknownmove"])])
        self.assertIn("unknown move ID", _error_text(_document(species=[species])))

    def test_52_noncanonical_move_is_rejected_not_rewritten(self):
        species = _species(variants=[_variant(move_ids=["Protect", "surf", "thunderbolt", "voltswitch"])])
        self.assertIn("expected canonical ID 'protect', received 'Protect'", _error_text(_document(species=[species])))

    def test_53_invalid_ability_is_rejected(self):
        species = _species(variants=[_variant(base_ability_id="syntheticunknownability")])
        self.assertIn("base ability does not belong", _error_text(_document(species=[species])))

    def test_54_base_ability_must_belong_to_exact_species_form(self):
        species = _species(variants=[_variant(base_ability_id="levitate")])
        self.assertIn("base ability does not belong", _error_text(_document(species=[species])))

    def test_55_ambiguous_no_item_representation_is_rejected(self):
        for item in (None, "", "No Item"):
            with self.subTest(item=item):
                species = _species(variants=[_variant(item_id=item)])
                self.assertIn("item_id", _error_text(_document(species=[species])))
        dataset = _load_document(_document(species=[_species(variants=[_variant(item_id="none")])]))
        self.assertEqual("none", dataset.species[0].variants[0].item_id)

    def test_56_known_nature_is_accepted(self):
        dataset = _load_document(_document(species=[_species(variants=[_variant(nature_id="adamant")])]))
        self.assertEqual("adamant", dataset.species[0].variants[0].nature_id)

    def test_57_unknown_nature_is_rejected(self):
        species = _species(variants=[_variant(nature_id="syntheticnature")])
        self.assertIn("unknown nature ID", _error_text(_document(species=[species])))

    def test_58_ev_field_names_are_validated(self):
        evs = {"hp": 0, "atk": 0, "defense": 0, "spa": 252, "spd": 4, "spe": 252}
        species = _species(variants=[_variant(evs=evs)])
        error = _error_text(_document(species=[species]))
        self.assertIn("missing required stat", error)
        self.assertIn("unexpected stat field", error)

    def test_59_ev_ranges_are_validated(self):
        evs = {"hp": -1, "atk": 0, "def": 0, "spa": 253, "spd": 4, "spe": 252}
        species = _species(variants=[_variant(evs=evs)])
        self.assertGreaterEqual(_error_text(_document(species=[species])).count("between 0 and 252"), 2)

    def test_60_ev_total_is_validated(self):
        evs = {"hp": 252, "atk": 252, "def": 252, "spa": 0, "spd": 0, "spe": 0}
        species = _species(variants=[_variant(evs=evs)])
        self.assertIn("total must not exceed 510", _error_text(_document(species=[species])))

    def test_61_iv_field_names_are_validated(self):
        ivs = {"hp": 31, "atk": 31, "defense": 31, "spa": 31, "spd": 31, "spe": 31}
        species = _species(variants=[_variant(ivs=ivs)])
        self.assertIn("unexpected stat field", _error_text(_document(species=[species])))

    def test_62_iv_ranges_are_validated(self):
        ivs = {"hp": -1, "atk": 32, "def": 31, "spa": 31, "spd": 31, "spe": 31}
        species = _species(variants=[_variant(ivs=ivs)])
        self.assertGreaterEqual(_error_text(_document(species=[species])).count("between 0 and 31"), 2)

    def test_63_level_range_is_validated(self):
        for level in (0, 101):
            with self.subTest(level=level):
                species = _species(variants=[_variant(level=level)])
                self.assertIn("between 1 and 100", _error_text(_document(species=[species])))

    def test_64_boolean_level_is_rejected(self):
        species = _species(variants=[_variant(level=True)])
        self.assertIn("expected an integer", _error_text(_document(species=[species])))

    def test_65_multiple_independent_errors_are_aggregated(self):
        variant = _variant(weight=0, move_ids=["Unknown Move"], nature_id="unknownnature", level=101, source_ids=["missing"])
        error = _validation_error(_document(visibility="closed", species=[_species("unknownmon", variants=[variant])]))
        self.assertGreaterEqual(len(error.issues), 7)
        issue = next(issue for issue in error.issues if issue.variant_id == "standard")
        self.assertEqual("syntheticprior", issue.dataset_id)
        self.assertEqual("unknownmon", issue.species_id)
        self.assertIsNotNone(issue.field)

    def test_66_source_file_is_not_modified(self):
        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
            path = _write_document(directory, _document())
            before = (path.read_bytes(), path.stat().st_mtime_ns)
            load_public_prior(path)
            after = (path.read_bytes(), path.stat().st_mtime_ns)
        self.assertEqual(before, after)

    def test_67_no_adjacent_file_or_cache_is_created(self):
        with tempfile.TemporaryDirectory(dir=FIXTURE_TEMP_ROOT) as directory:
            path = _write_document(directory, _document())
            before = {item.name for item in Path(directory).iterdir()}
            load_public_prior(path)
            after = {item.name for item in Path(directory).iterdir()}
        self.assertEqual(before, after)

    def test_68_loading_does_not_mutate_all_move_json(self):
        before = copy.deepcopy(all_move_json)
        _load_document()
        self.assertEqual(before, all_move_json)

    def test_69_loading_does_not_mutate_pokedex(self):
        before = copy.deepcopy(pokedex)
        _load_document()
        self.assertEqual(before, pokedex)

    def test_70_loading_does_not_mutate_foul_play_config(self):
        before = dict(vars(FoulPlayConfig))
        _load_document()
        self.assertEqual(before, vars(FoulPlayConfig))

    def test_71_loading_does_not_mutate_battle_modes(self):
        from fp.modes.standard_battle import StandardBattleMode

        before = dict(vars(StandardBattleMode))
        _load_document()
        self.assertEqual(before, vars(StandardBattleMode))

    def test_72_loading_does_not_mutate_team_datasets(self):
        from fp.data.sets import TeamDatasets

        datasets = TeamDatasets()
        before = copy.deepcopy(vars(datasets))
        _load_document()
        self.assertEqual(before, vars(datasets))

    def test_73_loading_does_not_mutate_smogon_sets(self):
        from fp.data.sets import SmogonSets

        sets = SmogonSets()
        before = copy.deepcopy(vars(sets))
        _load_document()
        self.assertEqual(before, vars(sets))

    def test_74_loading_does_not_mutate_phase_one_reference_pools(self):
        from fp.data.team_pools import TeamPoolRegistry

        registry = TeamPoolRegistry()
        before = registry.pools
        with mock.patch("fp.data.team_pools.loader.load_team_pool") as private_loader:
            _load_document()
        self.assertEqual(before, registry.pools)
        private_loader.assert_not_called()

    def test_75_loading_does_not_mutate_phase_three_inference_context(self):
        from fp.battle.team_inference import TeamInferenceContext

        context = TeamInferenceContext()
        before = context.safe_summary()
        _load_document()
        self.assertEqual(before, context.safe_summary())

    def test_76_ordinary_non_tugs_behavior_remains_unaffected(self):
        thunderbolt = copy.deepcopy(all_move_json["thunderbolt"])
        bulbasaur = copy.deepcopy(pokedex["bulbasaur"])
        _load_document()
        self.assertEqual(thunderbolt, all_move_json["thunderbolt"])
        self.assertEqual(bulbasaur, pokedex["bulbasaur"])

    def test_77_no_battle_or_search_module_is_a_loader_dependency(self):
        sources = "\n".join(path.read_text(encoding="utf-8") for path in PACKAGE_ROOT.glob("*.py"))
        forbidden = (
            "from fp.battle.state",
            "from fp.battle.protocol",
            "from fp.battle.team_inference",
            "from fp.search",
            "from fp.modes",
            "from fp.data.sets",
        )
        for import_text in forbidden:
            self.assertNotIn(import_text, sources)
        before = set(sys.modules)
        with mock.patch.object(socket, "create_connection", side_effect=AssertionError("network attempted")):
            _load_document()
        self.assertFalse({name for name in set(sys.modules) - before if name.startswith("fp.search")})

    def test_78_no_conversion_function_from_reference_teams_exists(self):
        import fp.data.public_priors as public_priors

        forbidden_names = {
            "prior_from_team_record",
            "convert_team_pool_to_priors",
            "use_matching_team_as_prior",
            "resolve_reference_candidate_for_search",
        }
        self.assertTrue(forbidden_names.isdisjoint(dir(public_priors)))
        sources = "\n".join(path.read_text(encoding="utf-8") for path in PACKAGE_ROOT.glob("*.py"))
        self.assertNotIn("fp.data.team_pools", sources)
        self.assertNotIn("TeamInferenceContext", sources)

    def test_79_no_production_public_prior_json_was_added(self):
        self.assertEqual([], list(PACKAGE_ROOT.glob("*.json")))

    def test_80_registry_iteration_and_cross_dataset_lookup_are_deterministic(self):
        first = _load_document()
        second = _second_dataset()
        registry = PublicPriorRegistry((second, first))
        identities = tuple(dataset.identity for dataset in registry)
        self.assertEqual(tuple(sorted(identities)), identities)
        with self.assertRaises(TypeError):
            registry.dataset_lookup[first.identity] = second
        references = registry.references_for_species("pikachu", (second.identity, first.identity))
        self.assertEqual(tuple(sorted(references)), references)
        self.assertEqual({first.identity, second.identity}, {reference.dataset_identity for reference in references})
        self.assertEqual((2.0,), tuple(variant.weight for variant in first.get_species("pikachu").variants))


if __name__ == "__main__":
    unittest.main()
