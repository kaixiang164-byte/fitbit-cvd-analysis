"""Static contract checks for vocabulary SQL; these do not execute BigQuery."""

from pathlib import Path
import re
import unittest

import validate_release


SQL_PATH = Path(__file__).with_name("export_clinical_concepts.sql")
EXPECTED_SEEDS = {
    "hypertension": {316866},
    "diabetes": {
        201820, 201254, 201826, 4024659, 195771, 442793,
        443238, 435216, 761051, 4058243,
    },
    "ckd": {46271022, 192279, 201313},
}


class ClinicalConceptExportContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sql = SQL_PATH.read_text(encoding="utf-8")
        cls.code = re.sub(r"--[^\n]*", "", cls.sql)

    def test_exact_component_seed_pairs_without_duplicates(self):
        pairs = re.findall(
            r"STRUCT\('([^']+)' AS component,\s*(\d+) AS seed_concept_id\)",
            self.code,
        )
        observed = [(component, int(concept)) for component, concept in pairs]
        expected = {
            (component, concept)
            for component, concepts in EXPECTED_SEEDS.items()
            for concept in concepts
        }
        self.assertEqual(set(observed), expected)
        self.assertEqual(len(observed), len(expected))
        self.assertIn(("diabetes", 443238), observed)

    def test_only_vocabulary_tables_are_read(self):
        references = set(re.findall(r"`([^`]+)`", self.code))
        self.assertEqual(references, {
            "PROJECT.DATASET.concept", "PROJECT.DATASET.concept_ancestor",
        })
        self.assertNotRegex(
            self.code.lower(),
            r"\b(person_id|condition_occurrence|visit_occurrence|observation|export data)\b",
        )
        targets = re.findall(r"CREATE\s+TEMP\s+TABLE\s+(\w+)", self.code, re.I)
        self.assertEqual(targets, [
            "clinical_seeds", "clinical_lipid_concepts", "expanded_clinical_concepts",
        ])

    def test_expansion_retains_seeds_all_descendants_and_component_overlap(self):
        normalized = " ".join(self.code.split())
        self.assertIn("SELECT component, seed_concept_id AS concept_id", normalized)
        self.assertIn("UNION ALL", normalized)
        self.assertIn("hierarchy.descendant_concept_id AS concept_id", normalized)
        self.assertIn("hierarchy.ancestor_concept_id = seeds.seed_concept_id", normalized)
        self.assertIn("SELECT DISTINCT component, concept_id", normalized)
        self.assertNotRegex(self.code.lower(), r"min_levels_of_separation|max_levels_of_separation")

    def test_missing_seeds_and_descendant_metadata_fail_instead_of_dropping(self):
        self.assertEqual(len(re.findall(r"ASSERT NOT EXISTS\s*\(", self.code)), 3)
        self.assertIn("vocabulary.concept_id = seeds.seed_concept_id", self.code)
        self.assertIn("WHERE vocabulary.concept_id IS NULL", self.code)
        self.assertIn("A Clinical seed is absent", self.code)
        self.assertIn("An included descendant or supplied lipid concept is absent", self.code)

    def test_lipid_sets_are_required_nonempty_arrays_without_null_entries(self):
        for parameter in ("hyperlipidaemia_concept_ids", "high_cholesterol_concept_ids"):
            self.assertIn("ASSERT @" + parameter + " IS NOT NULL", self.code)
            self.assertIn("ARRAY_LENGTH(@" + parameter + ") > 0", self.code)
            self.assertIn("FROM UNNEST(@" + parameter + ") AS concept_id", self.code)
        self.assertIn("FROM clinical_lipid_concepts WHERE concept_id IS NULL", self.code)
        self.assertNotRegex(self.code, r"\b(?:432867|437827)\b")

    def test_lipid_membership_is_not_inferred_or_hierarchy_expanded(self):
        hierarchy_sources = re.findall(
            r"FROM\s+(\w+)\s+AS\s+seeds\s+INNER JOIN\s+"
            r"`PROJECT\.DATASET\.concept_ancestor`",
            self.code,
        )
        self.assertEqual(hierarchy_sources, ["clinical_seeds"])
        self.assertEqual(self.code.count("`PROJECT.DATASET.concept_ancestor`"), 1)
        normalized = " ".join(self.code.split())
        self.assertIn("SELECT component, concept_id FROM clinical_lipid_concepts", normalized)
        self.assertIn("'supplied_lipid_concept_set'", self.code)
        self.assertIn("'nonlipid_seed_and_descendants'", self.code)
        self.assertIn("AS definition_source", self.code)
        self.assertIn("AS is_explicit_inclusion", self.code)

    def test_metadata_is_reported_not_used_to_filter_membership(self):
        for field in (
            "concept_name", "domain_id", "vocabulary_id", "concept_class_id",
            "standard_concept", "concept_code", "valid_start_date",
            "valid_end_date", "invalid_reason",
        ):
            self.assertIn("vocabulary." + field, self.code)
        for field in ("standard_concept", "domain_id", "invalid_reason"):
            self.assertEqual(self.code.count("vocabulary." + field), 1)
        self.assertNotIn("condition_status_concept_id", self.code)

    def test_sql_source_allowed_without_relaxing_hash_inventory(self):
        self.assertIn(".sql", validate_release.SAFE_SUFFIXES)
        validator_source = Path(validate_release.__file__).read_text(encoding="utf-8")
        self.assertIn('expected = manifest["files_sha256"]', validator_source)
        self.assertIn("if discovered != allowed:", validator_source)
        self.assertIn("hashlib.sha256(raw).hexdigest() != expected[name]", validator_source)


if __name__ == "__main__":
    unittest.main()
