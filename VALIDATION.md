# Source-package checks — 2026-09-17

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
