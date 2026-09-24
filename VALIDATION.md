# Source-package checks

## Source-update checks — 2026-09-24

- `python -B -m unittest -v test_setting_reassignment_sensitivity`:
  **13 new synthetic tests passed**. Coverage includes weighted and unweighted
  probability calibration, endpoint scenarios, invalid-input rejection,
  multinomial parameters/full covariance versus statsmodels, rational
  fractional counts versus repeated rows, bootstrap multiplicity equivalence,
  warm starts, finite-difference derivatives and extreme-logit stability.
- `python -B -m unittest discover -v -p 'test_*.py'`: **57 tests passed**,
  including the existing 44 and the new 13.
- `--help` for the allocation analysis and S14/S2 renderer imported correctly
  using the existing pinned scientific dependencies and local helper module.
- The three newly included source/test files and the existing unclassified-
  setting helper matched their reviewed working sources byte-for-byte.
- The seven notebook-source hashes still passed `--verify-sources`.
- `python -B validate_release.py`: the refreshed manifest inventory, hashes,
  Python syntax and limited disclosure scan passed for the local source set.

These checks ran on manufactured inputs only; no participant dataset or
aggregate study output was copied into or analysed from this package. The
new files postdate the earlier release `9e03508`; their hashes are included
in this version's manifest. These tests are not independent
reproduction of the reported cohort/results. The renderer was import/syntax
checked here; manuscript asset rendering is separate from these source-only
package checks.

## Original package checks — 2026-09-17

The following checks were run on this source-only package using Python 3.9.25
and the installed scientific dependencies documented in ENVIRONMENT.md:

- `python -B -m unittest discover -v -p 'test_*.py'`: **44 tests passed**,
  including nine tests for the newly implemented Table 1 generator.
- `python -B run_primary_and_moderated_notebook_source.py --verify-sources`:
  all seven embedded source-cell hashes and Python syntax passed.
- The nine copied existing standalone source/test files were byte-identical
  to their current source counterparts; mathematical code was not reformulated.
- `python -B -m pip check`: no broken installed requirements.
- `python -B validate_release.py`: manifest inventory, hashes, Python syntax
  and the limited credential/path scan passed for the reviewed file set.

Tests cover synthetic M1/PCA/M2/pathway execution, input compatibility with
separate-sex and unclassified-setting branches, the age-reference adapter,
linear-reference agreement, spline rank/parameter counts, pooled decomposition
identities, score scaling, deterministic synthetic resamples and provenance
guard failures. Table 1 checks cover sample SDs, derived scores, full-column
denominators, unclassified-setting exclusion, missing/invalid input rejection,
aggregate-only output, LaTeX rendering and non-overwriting output boundaries.
Synthetic runtime files are created in temporary directories;
they do not originate from participant data.

Not performed: a full study-data rerun, independent upstream cohort/measurement
verification, the 2,000-draw moderation bootstrap, or a fresh dependency
installation as part of these local checks. The moderation source was
hash/syntax checked, not numerically
reproduced on the study data during packaging. Passing tests is not a claim of
complete end-to-end reproduction or exhaustive confidentiality certification.
