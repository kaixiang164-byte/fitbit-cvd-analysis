-- BigQuery Standard SQL: vocabulary-only Clinical component dictionary.
-- Replace PROJECT.DATASET with the project's pinned CDR v8 dataset identifier.
-- Run the complete script, then export the final result set as a dictionary CSV.
-- Non-lipid seeds include all descendants; no direct-child relationship is assumed.
-- Each lipid component uses its single exact-match concept below, without
-- descendant expansion or automatic assignment from the other component.

CREATE TEMP TABLE clinical_seeds AS
SELECT * FROM UNNEST([
  STRUCT('hypertension' AS component, 316866 AS seed_concept_id),
  STRUCT('diabetes' AS component, 201820 AS seed_concept_id),
  STRUCT('diabetes' AS component, 201254 AS seed_concept_id),
  STRUCT('diabetes' AS component, 201826 AS seed_concept_id),
  STRUCT('diabetes' AS component, 4024659 AS seed_concept_id),
  STRUCT('diabetes' AS component, 195771 AS seed_concept_id),
  STRUCT('diabetes' AS component, 442793 AS seed_concept_id),
  STRUCT('diabetes' AS component, 443238 AS seed_concept_id),
  STRUCT('diabetes' AS component, 435216 AS seed_concept_id),
  STRUCT('diabetes' AS component, 761051 AS seed_concept_id),
  STRUCT('diabetes' AS component, 4058243 AS seed_concept_id),
  STRUCT('ckd' AS component, 46271022 AS seed_concept_id),
  STRUCT('ckd' AS component, 192279 AS seed_concept_id),
  STRUCT('ckd' AS component, 201313 AS seed_concept_id)
]);

ASSERT (
  SELECT COUNT(*) = 14
    AND COUNT(DISTINCT CONCAT(component, ':', CAST(seed_concept_id AS STRING))) = 14
  FROM clinical_seeds
) AS 'Expected 14 distinct non-lipid component/seed pairs.';

ASSERT NOT EXISTS (
  SELECT 1
  FROM clinical_seeds AS seeds
  LEFT JOIN `PROJECT.DATASET.concept` AS vocabulary
    ON vocabulary.concept_id = seeds.seed_concept_id
  WHERE vocabulary.concept_id IS NULL
) AS 'A Clinical seed is absent from the selected CDR v8 vocabulary; stop rather than drop it.';

CREATE TEMP TABLE clinical_lipid_concepts AS
SELECT * FROM UNNEST([
  STRUCT('hyperlipidaemia' AS component, 432867 AS concept_id),
  STRUCT('high_cholesterol' AS component, 437827 AS concept_id)
]);

CREATE TEMP TABLE expanded_clinical_concepts AS
WITH included_concepts AS (
  SELECT component, seed_concept_id AS concept_id
  FROM clinical_seeds
  UNION ALL
  SELECT seeds.component, hierarchy.descendant_concept_id AS concept_id
  FROM clinical_seeds AS seeds
  INNER JOIN `PROJECT.DATASET.concept_ancestor` AS hierarchy
    ON hierarchy.ancestor_concept_id = seeds.seed_concept_id
  UNION ALL
  SELECT component, concept_id
  FROM clinical_lipid_concepts
)
SELECT DISTINCT component, concept_id
FROM included_concepts;

ASSERT NOT EXISTS (
  SELECT 1
  FROM expanded_clinical_concepts AS expanded
  LEFT JOIN `PROJECT.DATASET.concept` AS vocabulary
    ON vocabulary.concept_id = expanded.concept_id
  WHERE vocabulary.concept_id IS NULL
) AS 'An included descendant or exact-match lipid concept is absent from concept; stop rather than drop it.';

SELECT
  expanded.component,
  expanded.concept_id,
  vocabulary.concept_name,
  vocabulary.domain_id,
  vocabulary.vocabulary_id,
  vocabulary.concept_class_id,
  vocabulary.standard_concept,
  vocabulary.concept_code,
  vocabulary.valid_start_date,
  vocabulary.valid_end_date,
  vocabulary.invalid_reason,
  CASE
    WHEN expanded.component IN ('hyperlipidaemia', 'high_cholesterol')
      THEN 'exact_match_lipid_concept'
    ELSE 'nonlipid_seed_and_descendants'
  END AS definition_source,
  EXISTS (
    SELECT 1
    FROM clinical_seeds AS seeds
    WHERE seeds.component = expanded.component
      AND seeds.seed_concept_id = expanded.concept_id
  ) OR EXISTS (
    SELECT 1
    FROM clinical_lipid_concepts AS lipid
    WHERE lipid.component = expanded.component
      AND lipid.concept_id = expanded.concept_id
  ) AS is_explicit_inclusion
FROM expanded_clinical_concepts AS expanded
INNER JOIN `PROJECT.DATASET.concept` AS vocabulary
  ON vocabulary.concept_id = expanded.concept_id
ORDER BY expanded.component, expanded.concept_id;
