# Environment and reproducibility settings

Target platform: Linux/POSIX, Python 3.9.25. Multiprocessing in the historical
scripts relies on `fork`; Windows/spawn portability is not established.
The current package checks use Python 3.9.25 and the versions in
`requirements.txt`. NumPy, pandas, SciPy, statsmodels and Matplotlib versions
are recorded in the original pooled-run manifest. Other dependency versions
describe the current inspection environment, not a recovered historical lock.
This is a pinned direct-dependency specification, not a complete container or
transitive-dependency lock. Hardware-independent bitwise identity is not claimed.
`requirements-tested.txt` additionally records the inspected dependencies of
these packages. It describes the checked environment, not a historical lockfile
or evidence that a fresh installation was tested during packaging.

Create a dedicated environment outside the release directory:

```bash
python3.9 -m venv ../fitbit-cvd-env
../fitbit-cvd-env/bin/python -m pip install -r requirements.txt
../fitbit-cvd-env/bin/python -B -m unittest discover -v -p 'test_*.py'
```

Tests use generated or mocked data only. Passing tests establishes the tested
code properties, not independent reproduction of the study's cohort or results.

## Numerical settings

- Fixed original pooled Steps and age standardization (`ddof=1`).
- Pooled bootstrap: 2,000 draws, seed 20260903, 51 residual nodes; the three
  score definitions use identical resampled participant positions. Point
  estimates are checked with 101 nodes. M1 probability bands use 500 coefficient
  simulations as configured by the original script.
- Separate sex-stratified bootstrap: 2,000 draws, seed 20260904, 51/101 residual
  nodes and grid size 1201. The global PCA definition remains pooled.
- Formal sex/age moderation uses the notebook-source implementation and its
  recorded configuration: 2,000 draws, seed 20260903, 51 residual nodes and
  a 101-node precision check, rather than substituting the sex-stratified script.
- Sleep-omission sensitivity reuses the pooled run's stored child seeds and
  primary draws; do not create new unmatched primary resamples for comparisons.
- BLAS threads are limited in the scripts to reduce nested parallelism.

Any rerun creates new run metadata and hashes. Historical output hashes must
not be rewritten to make changed computations appear identical to an old run.
Figure file bytes can differ with rendering libraries even if estimates agree.
