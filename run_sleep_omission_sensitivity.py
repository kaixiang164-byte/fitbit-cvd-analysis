#!/usr/bin/env python3
"""Post-hoc pooled mediation sensitivity omitting Sleep; no manuscript writes.

Reuse the verified canonical fitting/g-computation implementation, changing only
score construction. Read the original participant-bootstrap seeds and compare
against matching stored primary draws. Outputs contain aggregate results only.
"""

import os

for thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_variable] = "1"

import argparse
import json
import multiprocessing
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels

import run_robustness_and_figure2 as core


BASE = Path(__file__).resolve().parent
VARIANT = "PCA without Sleep (Mental + Clinical)"
ORIGINAL_SCORE = core.construct_vulnerability
REFERENCE = None


def standardised_domains(sample):
    return np.column_stack([
        core.z_standardize(sample[name].to_numpy(float))
        for name in ("Mental", "Clinical")
    ])


def sleep_omitted_score(sample, variant):
    if variant != VARIANT:
        return ORIGINAL_SCORE(sample, variant)
    domains = standardised_domains(sample)
    _, _, right = np.linalg.svd(domains, full_matrices=False)
    loading = right[0].copy()
    if loading @ REFERENCE < 0:
        loading *= -1
    return core.z_standardize(domains @ loading)


def columns_for(variant):
    return [
        f"{variant}|{scale}|{target}|{effect}"
        for scale, targets in (
            ("Pairwise-normalized", core.PAIRWISE_TARGETS),
            ("Raw category probability", core.RAW_TARGETS),
        )
        for target in targets for effect in core.EFFECTS
    ]


def fitted_effects(positions, variant, nodes=51):
    pairwise, raw = core.effects_for_variant(positions, variant, nodes)
    values = np.concatenate([pairwise.ravel(), raw.ravel()])
    if not np.isfinite(values).all():
        raise RuntimeError("Non-finite fitted effect.")
    return values


def bootstrap_one(seed):
    try:
        rng = np.random.default_rng(int(seed))
        n = len(core.ANALYSIS)
        positions = rng.integers(0, n, size=n)
        return True, fitted_effects(positions, VARIANT), ""
    except Exception as error:
        return False, np.full(18, np.nan), repr(error)


def paired_summary(primary, alternative, primary_draws, alternative_draws):
    rows = []
    for index, row in alternative.reset_index(drop=True).iterrows():
        base_row = primary.iloc[index]
        if tuple(row[k] for k in ("scale", "target", "effect")) != tuple(
            base_row[k] for k in ("scale", "target", "effect")
        ):
            raise AssertionError("Primary and alternative effects are not aligned.")
        difference = alternative_draws[:, index] - primary_draws[:, index]
        finite = difference[np.isfinite(difference)]
        lower, upper = np.quantile(finite, [0.025, 0.975])
        result = {k: row[k] for k in ("scale", "target", "effect")}
        for prefix, source in (("primary", base_row), ("without_sleep", row)):
            for name in ("estimate", "CI_lower", "CI_upper"):
                result[f"{prefix}_{name}_pp"] = 100 * float(source[name])
            result[f"{prefix}_p_value"] = float(source["p_value"])
        result.update({
            "difference_without_sleep_minus_primary_pp": 100 * (row.estimate - base_row.estimate),
            "paired_difference_CI_lower_pp": 100 * lower,
            "paired_difference_CI_upper_pp": 100 * upper,
            "successful_paired_draws": int(len(finite)),
        })
        rows.append(result)
    return pd.DataFrame(rows)


def main():
    global REFERENCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path,
                        default=BASE / "sleep_omission_sensitivity_20260915")
    args = parser.parse_args()
    if args.bootstrap < 2 or args.workers < 1:
        raise ValueError("Use at least 2 bootstrap draws and 1 worker.")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory to preserve earlier results.")
    start = time.monotonic()

    # Verify hashes before importing any original point estimates or paired draws.
    core.load_canonical_pooled_results()
    manifest = json.loads((BASE / "canonical_mediation_manifest.json").read_text())
    if manifest["residual_quantile_nodes"] != 51 or manifest["bootstrap_failed_draws"] != 0:
        raise ValueError("Expected canonical 51-node results with no failed draws.")
    seeds = pd.read_csv(BASE / "canonical_bootstrap_resamples.csv")
    saved_draws = pd.read_csv(BASE / "robustness_mediation_bootstrap_draws.csv")
    np.testing.assert_array_equal(seeds.draw, np.arange(1, len(seeds) + 1))
    np.testing.assert_array_equal(seeds.draw, saved_draws.draw)
    original_seed = manifest["configuration"]["seed"]
    regenerated_seeds = [int(child.generate_state(1)[0])
                         for child in np.random.SeedSequence(original_seed).spawn(len(seeds))]
    np.testing.assert_array_equal(seeds.child_seed, regenerated_seeds)
    if args.bootstrap > len(seeds):
        raise ValueError("Cannot request more draws than the saved paired resamples.")
    seeds = seeds.iloc[:args.bootstrap].copy()
    primary_columns = columns_for("Primary PCA")
    primary_draws = saved_draws.loc[:args.bootstrap - 1, primary_columns].to_numpy(float)
    if not np.isfinite(primary_draws).all():
        raise ValueError("Stored primary draws contain non-finite values.")
    original_summary = pd.read_csv(BASE / "robustness_mediation_summary.csv")
    lookup = original_summary.set_index(["variant", "scale", "target", "effect"])
    primary_point = np.array([lookup.loc[tuple(c.split("|")), "estimate"]
                              for c in primary_columns], dtype=float)
    positions = np.arange(len(core.ANALYSIS))
    np.testing.assert_allclose(fitted_effects(positions, "Primary PCA"), primary_point,
                               rtol=1e-9, atol=1e-11)
    # Reproducing a saved draw also checks that loading data preserved row order.
    rng = np.random.default_rng(int(seeds.child_seed.iloc[0]))
    sampled = rng.integers(0, len(positions), size=len(positions))
    np.testing.assert_allclose(fitted_effects(sampled, "Primary PCA"), primary_draws[0],
                               rtol=1e-9, atol=1e-11)

    domains = standardised_domains(core.ANALYSIS)
    _, singular_values, right = np.linalg.svd(domains, full_matrices=False)
    REFERENCE = right[0].copy()
    if np.corrcoef(domains @ REFERENCE, domains.sum(axis=1))[0, 1] < 0:
        REFERENCE *= -1
    if not np.all(REFERENCE > 0):
        raise ValueError("Two-domain PC1 does not have the expected common-burden orientation.")
    # Override only in this standalone process; never edit the canonical module.
    core.construct_vulnerability = sleep_omitted_score
    alternative_score = sleep_omitted_score(core.ANALYSIS, VARIANT)
    primary_score = ORIGINAL_SCORE(core.ANALYSIS, "Primary PCA")
    altered_sleep = core.ANALYSIS.assign(Sleep=1 - core.ANALYSIS.Sleep)
    np.testing.assert_allclose(alternative_score, sleep_omitted_score(altered_sleep, VARIANT))
    equal_weight = core.z_standardize(domains.mean(axis=1))
    np.testing.assert_allclose(alternative_score, equal_weight, atol=1e-10)
    point = fitted_effects(positions, VARIANT)
    point_101 = fitted_effects(positions, VARIANT, nodes=101)
    columns = columns_for(VARIANT)
    print(f"Verified n={len(positions)}, counts={core.EXPECTED_COUNTS}, "
          f"paired B={args.bootstrap}, nodes=51.", flush=True)
    print("Without-Sleep point effects (rows outpatient/control, acute/control, "
          "acute/outpatient; columns ACME, ADE, Total; percentage points):", flush=True)
    print((point[:9].reshape(3, 3) * 100).round(6), flush=True)

    output.mkdir(parents=True)
    draws = np.full((args.bootstrap, len(columns)), np.nan)
    failures = []
    executor = None
    if args.workers == 1:
        iterator = map(bootstrap_one, seeds.child_seed)
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers,
                                       mp_context=multiprocessing.get_context("fork"))
        iterator = executor.map(bootstrap_one, seeds.child_seed, chunksize=4)
    try:
        for index, (ok, values, message) in enumerate(iterator):
            draws[index] = values
            if not ok:
                failures.append({"draw": index + 1, "error": message})
            if (index + 1) % 100 == 0 or index + 1 == args.bootstrap:
                print(f"Bootstrap {index + 1}/{args.bootstrap}; failures={len(failures)}; "
                      f"elapsed={time.monotonic() - start:.1f}s", flush=True)
    finally:
        if executor is not None:
            executor.shutdown()
    summary = core.summarize_bootstrap(point, draws, columns, args.bootstrap, original_seed)
    primary_summary = core.summarize_bootstrap(primary_point, primary_draws,
                                               primary_columns, args.bootstrap, original_seed)
    if args.bootstrap == len(saved_draws):
        reference_rows = lookup.loc[[tuple(c.split("|")) for c in primary_columns]]
        for column in ("estimate", "CI_lower", "CI_upper", "p_value"):
            np.testing.assert_allclose(primary_summary[column], reference_rows[column],
                                       rtol=1e-9, atol=1e-11)
    comparison = paired_summary(primary_summary, summary, primary_draws, draws)
    all_summary = pd.concat([primary_summary, summary], ignore_index=True)
    all_summary.to_csv(output / "effects_summary.csv", index=False)
    comparison.to_csv(output / "paired_comparison.csv", index=False)
    draw_frame = pd.DataFrame(draws, columns=columns)
    draw_frame.insert(0, "child_seed", seeds.child_seed.to_numpy())
    draw_frame.insert(0, "draw", seeds.draw.to_numpy())
    draw_frame.to_csv(output / "bootstrap_draws.csv", index=False)
    pd.DataFrame({"component": columns, "estimate_51_nodes": point,
                  "estimate_101_nodes": point_101,
                  "difference_pp": 100 * (point_101 - point)}).to_csv(
                      output / "integration_check.csv", index=False)
    pd.DataFrame(failures, columns=["draw", "error"]).to_csv(output / "failures.csv", index=False)
    metadata = {
        "analysis": "Post-hoc pooled score-composition sensitivity, omitting Sleep",
        "variant": VARIANT, "n": len(positions), "outcome_counts": core.EXPECTED_COUNTS,
        "domains": ["Mental", "Clinical"], "loadings": REFERENCE.tolist(),
        "PC1_variance_explained": float(singular_values[0]**2 / (singular_values**2).sum()),
        "Mental_Clinical_correlation": float(np.corrcoef(domains.T)[0, 1]),
        "primary_alternative_score_correlation": float(np.corrcoef(primary_score, alternative_score)[0, 1]),
        "equal_weight_max_absolute_difference": float(np.max(np.abs(alternative_score - equal_weight))),
        "steps_mean": core.STEPS_MEAN, "steps_sd": core.STEPS_SD,
        "age_mean": core.AGE_MEAN, "age_sd": core.AGE_SD,
        "bootstrap_requested": args.bootstrap, "bootstrap_failed": len(failures),
        "bootstrap_successful": args.bootstrap - len(failures), "seed": original_seed,
        "residual_nodes": 51, "workers": args.workers,
        "standardization": "Steps and age fixed at original pooled scales; Mental/Clinical, PC1 and both regressions refitted in each participant bootstrap; PC1 sign aligned.",
        "inference": "Percentile 95% CI; two-sided sign-tail p with add-one correction; exploratory unadjusted. Paired differences use identical participant draws, but compare different score definitions, not a sleep-specific causal contribution.",
        "limitations": "Same preselected cohort: does not test sleep-based eligibility or validate original nightly measurement, thresholds, or extraction. Primary model and manuscript unchanged.",
        "canonical_data_sha256": manifest["data_file_sha256"],
        "canonical_manifest_sha256": core.sha256_file(BASE / "canonical_mediation_manifest.json"),
        "canonical_code_sha256": core.sha256_file(BASE / "run_robustness_and_figure2.py"),
        "canonical_resamples_sha256": core.sha256_file(BASE / "canonical_bootstrap_resamples.csv"),
        "script_sha256": core.sha256_file(__file__),
        "primary_point_and_first_bootstrap_reproduced": True,
        "sleep_independence_score_test_passed": True,
        "all_effect_decomposition_identities_checked": True,
        "integration_max_absolute_change_pp": float(np.max(np.abs(100 * (point_101 - point)))),
        "elapsed_seconds": time.monotonic() - start,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "statsmodels": statsmodels.__version__},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Guard the unchanged canonical results again after the new run.
    core.load_canonical_pooled_results()
    print(comparison.loc[comparison.effect.eq("ACME")].to_string(index=False), flush=True)
    print(f"Results saved only to {output}", flush=True)
    if failures:
        raise RuntimeError("Some bootstrap draws failed; inspect failures.csv before interpretation.")


if __name__ == "__main__":
    main()
