# Source-package validation

## Current checks

On 2026-10-15, using Python 3.9.25 and the dependencies documented in
[ENVIRONMENT.md](ENVIRONMENT.md):

```bash
python3 -B -m unittest discover -v -p 'test_*.py'
python3 -B run_primary_and_moderated_notebook_source.py --verify-sources
```

All **105 synthetic, mocked-input or static-contract tests passed**. All seven embedded notebook
source-cell hashes and syntax checks passed. No participant data were used.

## Test coverage

| Component | Checks |
| --- | --- |
| Clinical vocabulary export | Fourteen non-lipid seeds and their descendant expansion; exact-match lipid concepts 432867 and 437827 without descendant expansion or automatic cross-assignment; within-component deduplication, cross-component overlap, definition-source labels, missing-concept assertions and vocabulary-only scope; eight static checks without BigQuery execution |
| Primary-model wrapper | Synthetic M1/PCA/M2/pathway execution; embedded source integrity; output-directory isolation; compatibility with the sex/setting branches |
| Pooled decomposition | Probability-scale decomposition and shared-denominator identities; pooled score scaling; deterministic synthetic resamples |
| Steps and age splines | Design rank and parameter counts; likelihood/AIC identities; linear-reference agreement; case-only scaling; nonconvergence checks |
| Missing-setting allocation | Weighted calibration; endpoint scenarios; fractional multinomial parameters and full covariance against statsmodels; repeated-row equivalence; finite-difference derivatives; stable extreme logits |
| Binary any-CVD recording | Nested-cohort construction; retention of all unclassified cases as CVD recordings; fixed original scaling and categorical covariates; logistic/binomial-GLM agreement; count and missingness checks; aggregate table rendering; eight synthetic tests |
| Recording-setting descriptives | Four-group case partition; means, sample SDs and linearly interpolated Steps quartiles; SMDs against the pooled classified reference; separate handling of suppressed cells and non-applicable median SMDs; complementary suppression of counts, percentages and SMDs; six synthetic tests |
| Unadjusted/adjusted comparison | Identical rows and fixed scaling; direct multinomial agreement; cross-logit covariance; reference reproduction; invalid-input rejection; S15 rendering |
| Table 1 | Sample SDs; derived scores; full-column percentage denominators; missing/invalid input rejection; unclassified-setting exclusion; aggregate-only output; LaTeX rendering |
| Reporting provenance | Accepted source/artifact hash chains; rejection of altered numerical exports, mismatched originals and unreviewed source changes |

Synthetic runtime files are created in temporary directories. They do not
originate from participant data. The Table 1 generator implements the reported
definitions, but its agreement with the historical study table has not been
checked on participant data during packaging.

## Inventory and disclosure checks

```bash
python3 -B validate_release.py
```

The validator checks the release allowlist, SHA-256 hashes, Python syntax,
private absolute paths and selected credential patterns. It is not an
exhaustive privacy review or evidence of scientific validity. Generated data,
logs, figures and result exports are outside the source-only release inventory.

## Earlier recorded checks

| Date | Recorded result |
| --- | --- |
| 2026-09-17 | 44 tests passed; seven embedded notebook source hashes verified; `pip check` reported no broken installed requirements; source inventory and limited disclosure scan passed |
| 2026-09-24, allocation addition | 57 tests passed, including 13 allocation tests; allocation/rendering CLI imports, notebook source hashes and source inventory passed |
| 2026-09-24, S15 addition | 66 tests passed, including nine unadjusted/adjusted comparison tests |

These checks did not include a full study-data rerun, independent upstream
cohort or measurement verification, the 2,000-draw moderation bootstrap, or a
fresh dependency installation. The tests therefore do not establish complete
end-to-end reproduction of the manuscript results.
