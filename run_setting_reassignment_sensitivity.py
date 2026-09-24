#!/usr/bin/env python3
"""Hypothetical recording-setting allocations; does not alter primary results.

Only the 455 missing-setting CVD cases are allocated; the 148 combined
Other/Unknown cases stay excluded. All allocations are assumptions, not recovered
records. See the generated report for the two different uncertainty intervals.
Only aggregate outputs are written, in a separate directory.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import brentq
from scipy.special import expit, logsumexp
from scipy.stats import norm

import run_unclassified_setting_sensitivity as original

BASE_DIR = Path(__file__).resolve().parent
REFERENCE_FRACTION = 385 / 2512
FRACTIONS = (0.05, 0.10, REFERENCE_FRACTION, 0.25, 0.40, 0.60, 0.80, 0.95)
ALLOCATION_ORS = (0.25, 0.5, 1.0, 2.0, 4.0)
CONTRAST_NAMES = ("Outpatient vs control", "Acute vs control", "Acute vs outpatient")
_WORKER = None


def calibrate_probabilities(steps_z, target, log_odds_slope, weights=None):
    """Return p and intercept with weighted mean(p)==target.

    exp(log_odds_slope) is the ASSUMED acute-assignment odds ratio per original
    Steps SD within missing-setting cases, not the observed M1 association.
    """
    z = np.asarray(steps_z, dtype=float)
    if z.ndim != 1 or len(z) == 0 or not np.isfinite(z).all():
        raise ValueError("Invalid steps_z.")
    if not np.isfinite(target) or not 0 <= target <= 1:
        raise ValueError("Target must be between zero and one.")
    if not np.isfinite(log_odds_slope):
        raise ValueError("Nonfinite log_odds_slope.")
    w = np.ones_like(z) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != z.shape or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Invalid calibration weights.")
    if target in (0, 1):
        return np.full_like(z, target), float("-inf" if target == 0 else "inf")
    offset = log_odds_slope * z
    if not np.isfinite(offset).all():
        raise ValueError("Nonfinite probability offset.")
    if log_odds_slope == 0:
        return np.full_like(z, target), float(np.log(target / (1 - target)))
    bound = 50 + np.max(np.abs(offset))
    intercept = brentq(lambda a: np.dot(w, expit(a + offset)) / w.sum() - target,
                      -bound, bound, xtol=1e-12)
    probabilities = expit(intercept + offset)
    if abs(np.average(probabilities, weights=w) - target) > 1e-10:
        raise RuntimeError("Probability calibration failed.")
    return probabilities, float(intercept)


def _validate_model_inputs(design, counts, params=None):
    x = np.asarray(design, dtype=float)
    y = np.asarray(counts, dtype=float)
    if x.ndim != 2 or y.shape != (len(x), 3) or x.shape[1] < 1:
        raise ValueError("Invalid design/count dimensions.")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Nonfinite or negative model input.")
    if (y.sum(axis=0) <= 0).any():
        raise ValueError("Every outcome category must have positive weight.")
    if np.linalg.matrix_rank(x[y.sum(axis=1) > 0]) != x.shape[1]:
        raise ValueError("Rank-deficient design.")
    if params is not None:
        params = np.asarray(params, dtype=float)
        if params.shape != (x.shape[1], 2) or not np.isfinite(params).all():
            raise ValueError("Invalid coefficient matrix.")
    return x, y, params


def multinomial_loglik_score_info(design, counts, params):
    """Counts can be hard, fractional or integer bootstrap multiplicities.

    Parameter order and full information matrix match statsmodels MNLogit:
    category 1 coefficients first, then category 2, baseline category 0.
    """
    x, y, params = _validate_model_inputs(design, counts, params)
    return _multinomial_loglik_score_info(x, y, params)


def _multinomial_loglik_score_info(x, y, params):
    """Internal kernel; inputs have already passed validation."""
    weights = y.sum(axis=1)
    eta = x @ params
    logden = logsumexp(np.column_stack((np.zeros(len(x)), eta)), axis=1)
    p = np.exp(eta - logden[:, None])
    ll = float(np.sum(y[:, 1:] * eta) - weights @ logden)
    score = (x.T @ (y[:, 1:] - weights[:, None] * p)).ravel(order="F")
    p1, p2 = p.T
    h11 = x.T @ ((weights * p1 * (1 - p1))[:, None] * x)
    h22 = x.T @ ((weights * p2 * (1 - p2))[:, None] * x)
    h12 = x.T @ ((-weights * p1 * p2)[:, None] * x)
    info = np.block([[h11, h12], [h12.T, h22]])
    return ll, score, info


def fit_multinomial_counts(design, counts, start=None, with_covariance=False):
    """Checked Newton maximisation of the complete/fractional multinomial LL."""
    x, y, start = _validate_model_inputs(design, counts, start)
    if start is None:
        params = np.zeros((x.shape[1], 2))
        params[0] = np.log(y.sum(axis=0)[1:] / y.sum(axis=0)[0])
    else:
        params = start.copy()
    total_weight = y.sum()
    for iteration in range(1, 81):
        ll, score, info = _multinomial_loglik_score_info(x, y, params)
        score_max = float(np.max(np.abs(score)) / total_weight)
        if score_max < 1e-9:
            break
        change = np.linalg.solve(info, score).reshape(params.shape, order="F")
        alpha = 1.0
        while alpha >= 2 ** -20:
            candidate = params + alpha * change
            eta = x @ candidate
            logden = logsumexp(np.column_stack((np.zeros(len(x)), eta)), axis=1)
            next_ll = float(np.sum(y[:, 1:] * eta) - y.sum(axis=1) @ logden)
            if np.isfinite(next_ll) and next_ll >= ll - 1e-8:
                params = candidate
                break
            alpha *= 0.5
        else:
            raise RuntimeError("Newton line search failed.")
    else:
        raise RuntimeError("Multinomial fit did not converge.")
    if not np.isfinite(params).all():
        raise RuntimeError("Nonfinite fitted coefficients.")
    result = {"params": params, "loglike": ll, "iterations": iteration,
              "max_abs_score": score_max}
    if with_covariance:
        covariance = np.linalg.inv(info)
        if np.linalg.eigvalsh(covariance).min() <= 0:
            raise RuntimeError("Invalid covariance matrix.")
        result["covariance"] = covariance
    return result


def step_contrasts(params, step_index=1):
    outpatient, acute = np.asarray(params)[step_index]
    return np.array([outpatient, acute, acute - outpatient])


def scenarios():
    rows = []
    for q in FRACTIONS:
        for allocation_or in ALLOCATION_ORS:
            rows.append({"scenario_id": f"q{q:.6f}_or{allocation_or:g}",
                         "acute_fraction": q, "allocation_OR_per_steps_SD": allocation_or,
                         "log_odds_slope": float(np.log(allocation_or)), "endpoint": False})
    for q in (0.0, 1.0):
        rows.append({"scenario_id": "all_outpatient" if q == 0 else "all_acute",
                     "acute_fraction": q, "allocation_OR_per_steps_SD": 1.0,
                     "log_odds_slope": 0.0, "endpoint": True})
    return rows


def prepare_data(directory):
    data, parameters = original.load_data(directory)
    other = data["Group"].eq("Heart Disease") & data.Onset_Type.eq("Other/Unknown")
    missing = data["Group"].eq("Heart Disease") & data.Onset_Type.isna()
    known = data.outcome.ne(3)
    counts = {"original_n": len(data), "known_n": int(known.sum()),
              "missing_setting_n": int(missing.sum()), "other_unknown_excluded_n": int(other.sum())}
    if counts != {"original_n": 8895, "known_n": 8292, "missing_setting_n": 455,
                  "other_unknown_excluded_n": 148}:
        raise ValueError("Input counts differ from the stated sensitivity cohort.")
    # Keep original controls/cases first, missing-setting cases last.
    selected = pd.concat([data.loc[known], data.loc[missing]], ignore_index=True)
    complete, design = original.build_design(selected, parameters)
    if len(complete) != len(selected):
        raise ValueError("Missing model covariates: no silent cohort change.")
    x = design.to_numpy(float)
    n_known = counts["known_n"]
    y = np.zeros((len(x), 3))
    y[np.arange(n_known), selected.outcome.to_numpy(int)[:n_known]] = 1.0
    primary = fit_multinomial_counts(x[:n_known], y[:n_known], with_covariance=True)
    native, native_diagnostics = original.fit_and_contrast(
        data.loc[known], parameters, "Reproduced primary M1")
    reproduction = original.verify_primary(native, directory)
    step_rows = native.loc[native.term.eq("steps_z")].set_index("contrast")
    np.testing.assert_allclose(step_contrasts(primary["params"]),
                               step_rows.loc[list(CONTRAST_NAMES), "log_OR"], atol=1e-8, rtol=0)
    k = x.shape[1]
    for j, vector in enumerate(([1, 0], [0, 1], [-1, 1])):
        contrast = np.zeros(2 * k)
        contrast[1], contrast[k + 1] = vector
        se = np.sqrt(contrast @ primary["covariance"] @ contrast)
        np.testing.assert_allclose(se, step_rows.loc[CONTRAST_NAMES[j], "SE"], atol=1e-8, rtol=0)
    return x, y, n_known, primary, step_rows, {
        **counts, "sensitivity_n": len(selected), "controls_n": 5780,
        "known_outpatient_n": 2127, "known_acute_n": 385,
        "standardization": parameters.reset_index().to_dict(orient="records"),
        "design_columns": list(design.columns),
        "primary_reproduction_errors": reproduction,
        "primary_native_diagnostics": native_diagnostics,
    }


def fractional_counts(observed, n_known, probabilities, weights=None):
    """Preserve observed labels; each missing case contributes one total unit."""
    y = observed.copy()
    y[n_known:, 1] = 1 - probabilities
    y[n_known:, 2] = probabilities
    if weights is not None:
        y *= np.asarray(weights)[:, None]
    return y


def init_worker(design, observed, n_known, grid, starts, seed):
    global _WORKER
    _WORKER = (design, observed, n_known, grid, starts, seed)


def bootstrap_one(rep):
    x, observed, n_known, grid, starts, seed = _WORKER
    # Resamples and per-occurrence uniforms are shared across ALL scenarios.
    rng = np.random.default_rng(np.random.SeedSequence([seed, 100, rep]))
    indices = rng.integers(len(x), size=len(x))
    weights = np.bincount(indices, minlength=len(x)).astype(float)
    missing_occurrences = indices[indices >= n_known] - n_known
    uniforms = rng.random(len(missing_occurrences))
    z = x[n_known:, 1]
    if len(missing_occurrences) == 0:
        raise RuntimeError("Bootstrap contains no missing-setting cases.")
    fractional = np.empty((len(grid), 3))
    assigned = np.empty_like(fractional)
    max_score = 0.0
    max_iterations = 0
    for j, scenario in enumerate(grid):
        p, _ = calibrate_probabilities(z, scenario["acute_fraction"],
                                       scenario["log_odds_slope"], weights[n_known:])
        y = fractional_counts(observed, n_known, p, weights)
        fit = fit_multinomial_counts(x, y, start=starts[j])
        fractional[j] = step_contrasts(fit["params"])
        max_score = max(max_score, fit["max_abs_score"])
        max_iterations = max(max_iterations, fit["iterations"])
        if scenario["endpoint"]:
            assigned[j] = fractional[j]
            continue
        # Independent completion per bootstrap occurrence (including duplicates).
        is_acute = uniforms < p[missing_occurrences]
        acute_counts = np.bincount(missing_occurrences[is_acute], minlength=len(z))
        y[n_known:, 2] = acute_counts
        y[n_known:, 1] = weights[n_known:] - acute_counts
        completed = fit_multinomial_counts(x, y, start=fit["params"])
        assigned[j] = step_contrasts(completed["params"])
        max_score = max(max_score, completed["max_abs_score"])
        max_iterations = max(max_iterations, completed["iterations"])
    return rep, fractional, assigned, max_score, max_iterations


def create_plot(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    rows = summary.loc[(summary.contrast == "Acute vs outpatient") & ~summary.endpoint]
    values = rows.pivot(index="acute_fraction", columns="allocation_OR_per_steps_SD", values="OR")
    # Actual fitted association is colour-coded, not scenario allocation odds.
    limits = max(abs(np.log(values.to_numpy())).max(), 0.02)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    im = ax.imshow(np.log(values.to_numpy()), cmap="RdBu_r", aspect="auto",
                   norm=TwoSlopeNorm(vmin=-limits, vcenter=0, vmax=limits))
    ax.set_xticks(range(len(values.columns)), [f"{v:g}" for v in values.columns])
    ax.set_yticks(range(len(values.index)), [f"{100*v:.1f}%" for v in values.index])
    ax.set_xlabel("Assumed acute-assignment odds ratio per +1 SD Steps among missing cases\n"
                  "<1: lower-step cases more likely acute     1: independent of Steps     >1: higher-step cases more likely acute",
                  fontsize=10, labelpad=10)
    ax.set_ylabel("Assumed mean acute proportion among the 455 missing-setting cases", fontsize=11)
    ax.set_title("Missing-setting allocation sensitivity: acute care vs outpatient\n"
                 "Cell: M1 Steps OR [95% sampling-and-assignment scenario interval]", fontsize=13, pad=14)
    for i, q in enumerate(values.index):
        for j, allocation in enumerate(values.columns):
            row = rows.loc[np.isclose(rows.acute_fraction, q) &
                           np.isclose(rows.allocation_OR_per_steps_SD, allocation)].iloc[0]
            dark = abs(np.log(row.OR)) > 0.6 * limits
            ax.text(j, i, f"{row.OR:.3f}\n[{row.scenario_lower:.3f}, {row.scenario_upper:.3f}]",
                    ha="center", va="center", fontsize=9, color="white" if dark else "black")
    colorbar = fig.colorbar(im, ax=ax, pad=0.025, fraction=0.04)
    ticks = np.linspace(-limits, limits, 5)
    colorbar.set_ticks(ticks, labels=[f"{np.exp(v):.2f}" for v in ticks])
    colorbar.set_label("Fitted M1 Steps OR per original SD")
    fig.text(0.08, 0.025,
             "N=8,747; 148 Other/Unknown cases remain excluded. Primary observed-setting OR=0.839 (95% CI 0.742–0.948).\n"
             "Intervals reflect hypothetical classification plus participant resampling, not calibrated confidence intervals for the true setting association.\n"
             "Allocation assumptions are not inferred from the data. The 15.3% row is a fixed reference, not an estimated missing-case proportion.",
             fontsize=9, va="bottom")
    fig.subplots_adjust(left=0.12, right=0.91, bottom=0.20, top=0.88)
    fig.savefig(output / "setting_reassignment_heatmap.pdf")
    fig.savefig(output / "setting_reassignment_heatmap.png", dpi=180)
    plt.close(fig)


def write_report(summary, reference, metadata, output):
    rows = summary.loc[summary.contrast.eq("Acute vs outpatient")]
    text = ["# Missing-setting allocation sensitivity — results for review", "",
            "This is a post-hoc scenario analysis, not a replacement for M1 or S7. "
            "No manuscript text, primary result, participant label, PCA or mediation model was changed.", "",
            "## Cohort and fixed specification", "",
            "The 455 CVD cases with missing setting were added to the 8,292 classified-cohort participants "
            "(N=8,747). The 148 combined Other/Unknown records were not allocated. "
            "The analysis assumes that each of the 455 belongs to either acute care or outpatient; this is not verified. "
            "The original pooled Steps and age scales and age, sex, smoking and drinking adjustment were retained.", "",
            "## Allocation assumptions (chosen before viewing results)", "",
            "For each missing case, p_i = expit(a + lambda * original_steps_z_i). "
            "The intercept a is calibrated so that the average p_i equals the specified acute fraction q. "
            "exp(lambda) is an assumed assignment odds ratio, not the fitted Steps–setting association. "
            "Allocation depends only on Steps; outcome-model covariate adjustment is unchanged. "
            "These are hypothetical stress tests, not a fitted imputation model or a claim of MAR/MNAR identification.", "",
            "Fractions: 5%, 10%, 385/2512 (15.3%), 25%, 40%, 60%, 80%, 95%; "
            "assignment odds ratios: 0.25, 0.5, 1, 2, 4 per original Steps SD. "
            "The known-case 15.3% fraction is held as a fixed numerical benchmark, not a recovered missing-case rate. "
            "All-outpatient and all-acute allocations are separate endpoint stress tests, not mathematical bounds.", "",
            "## Estimation and two distinct intervals", "",
            "The point estimate maximises the expected complete-data multinomial log likelihood: "
            "a missing case contributes p_i to acute and 1-p_i to outpatient. "
            "This fractional-outcome estimate is deterministic; it is not exactly the average coefficient of random completions. "
            "All observed labels are unchanged.", "",
            f"{metadata['bootstrap_successful']} ordinary participant bootstrap samples were drawn, "
            "with original standardisation fixed and the calibration intercept refitted in each sample. "
            "The same bootstrap samples and per-occurrence random uniforms were shared across scenarios.", "",
            "1. **Sampling CI:** percentile interval of the bootstrapped fractional estimator, conditional on the fixed scenario. "
            "It does not include uncertainty over whether the chosen q/lambda assumptions are correct.",
            "2. **Sampling-and-assignment scenario interval:** 2.5th–97.5th percentiles after drawing acute/outpatient "
            "labels independently for missing bootstrap occurrences and refitting M1. This additionally displays "
            "random-classification dispersion. It is NOT a calibrated 95% CI for the fractional point estimate or the unknown true association. "
            "A single completion per bootstrap is deliberate for this broader distribution, not Rubin-style MI inference.", "",
            "No scenario p-values, best-case selection, recovered participant labels, or numerical treatment of "
            "unobserved settings as known facts is implied. No claim is made that this resolves confounding or EHR selection.", "",
            "## Original observed-setting reference", "",
            "| Comparison | OR (95% Wald CI) |", "|---|---|"]
    for name in CONTRAST_NAMES:
        row = reference.loc[name]
        text.append(f"| {name} | {row.OR:.3f} ({row.CI_lower:.3f}–{row.CI_upper:.3f}) |")
    text += ["", "## Acute-versus-outpatient scenario results", "",
             "| Assumed acute fraction | Assignment OR per Steps SD | Fitted Steps OR | Sampling 95% CI | Sampling + assignment 95% scenario interval |",
             "|---:|---:|---:|---|---|"]
    for row in rows.itertuples():
        text.append(f"| {100*row.acute_fraction:.1f}% | {row.allocation_OR_per_steps_SD:g} | "
                    f"{row.OR:.3f} | {row.sampling_lower:.3f}–{row.sampling_upper:.3f} | "
                    f"{row.scenario_lower:.3f}–{row.scenario_upper:.3f} |")
    text += ["", "## Interpretation boundaries", "",
             "Use the full pattern of estimates, magnitude and intervals, not only whether an interval crosses 1. "
             "A crossing is a sensitivity threshold within a tested grid, not proof of no association. "
             "Stability within selected scenarios is not universal robustness. The 148 Other/Unknown cases remain "
             "unresolved, and the two-setting assumption for missing cases is not independently verified.", "",
             "Methods background: Leacy et al. (2017), https://doi.org/10.1093/aje/kww107; "
             "Bartlett and Hughes (2020), https://doi.org/10.1177/0962280220932189. "
             "The present fixed-probability allocation stress test is not presented as their validated MI implementation.", ""]
    (output / "setting_reassignment_report.md").write_text("\n".join(text))


def run_analysis(directory=BASE_DIR, output=None, bootstrap=2000, workers=4, seed=20260924):
    directory = Path(directory)
    output = Path(output) if output is not None else directory / "setting_reassignment_sensitivity"
    if bootstrap < 20 or workers < 1:
        raise ValueError("Need at least 20 bootstrap replicates and one worker.")
    output.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()
    x, observed, n_known, primary, reference, metadata = prepare_data(directory)
    grid = scenarios()
    point_fits = []
    probability_rows = []
    for scenario in grid:
        p, intercept = calibrate_probabilities(x[n_known:, 1], scenario["acute_fraction"], scenario["log_odds_slope"])
        fit = fit_multinomial_counts(x, fractional_counts(observed, n_known, p), start=primary["params"])
        point_fits.append(fit)
        probability_rows.append({**scenario, "intercept": intercept if np.isfinite(intercept) else None,
                                 "expected_added_acute": float(p.sum()),
                                 "probability_min": float(p.min()), "probability_max": float(p.max()),
                                 "probability_mean": float(p.mean())})
    print(f"Validated primary M1; {len(grid)} scenarios; N={len(x)}; bootstraps={bootstrap}", flush=True)
    starts = np.stack([fit["params"] for fit in point_fits])
    sample_draws = np.empty((bootstrap, len(grid), 3))
    assigned_draws = np.empty_like(sample_draws)
    max_score = 0.0
    max_iterations = 0
    arguments = (x, observed, n_known, grid, starts, seed)
    if workers == 1:
        init_worker(*arguments)
        iterator = map(bootstrap_one, range(bootstrap))
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=arguments)
        iterator = pool.map(bootstrap_one, range(bootstrap), chunksize=2)
    try:
        for completed, (rep, fractional, assigned, score, iterations) in enumerate(iterator, 1):
            sample_draws[rep] = fractional
            assigned_draws[rep] = assigned
            max_score = max(max_score, score)
            max_iterations = max(max_iterations, iterations)
            if completed % max(1, bootstrap // 20) == 0:
                print(f"Bootstrap {completed}/{bootstrap}; elapsed {time.monotonic()-start_time:.1f}s", flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    if not np.isfinite(sample_draws).all() or not np.isfinite(assigned_draws).all():
        raise RuntimeError("Nonfinite bootstrap result; no silent deletion of failed fits.")
    for j, scenario in enumerate(grid):
        if scenario["endpoint"]:
            np.testing.assert_array_equal(sample_draws[:, j], assigned_draws[:, j])
    records = []
    for j, scenario in enumerate(grid):
        point = step_contrasts(point_fits[j]["params"])
        for c, name in enumerate(CONTRAST_NAMES):
            sample_ci = np.exp(np.quantile(sample_draws[:, j, c], [0.025, 0.975]))
            scenario_interval = np.exp(np.quantile(assigned_draws[:, j, c], [0.025, 0.975]))
            records.append({**scenario, "contrast": name, "n": len(x), "log_OR": point[c], "OR": np.exp(point[c]),
                            "sampling_lower": sample_ci[0], "sampling_upper": sample_ci[1],
                            "scenario_lower": scenario_interval[0], "scenario_upper": scenario_interval[1],
                            "completed_log_OR_mean": float(assigned_draws[:, j, c].mean()),
                            "sampling_bootstrap_SE": float(sample_draws[:, j, c].std(ddof=1)),
                            "scenario_distribution_SD": float(assigned_draws[:, j, c].std(ddof=1))})
    summary = pd.DataFrame(records)
    metadata.update({"bootstrap_requested": bootstrap, "bootstrap_successful": bootstrap, "bootstrap_failed": 0,
                     "seed": seed, "workers": workers, "scenario_count": len(grid),
                     "maximum_bootstrap_normalized_score": max_score, "maximum_Newton_iterations": max_iterations,
                     "elapsed_seconds": time.monotonic() - start_time,
                     "created_utc": datetime.now(timezone.utc).isoformat(),
                     "point_estimator": "Expected complete-data multinomial log-likelihood maximizer, fractional labels",
                     "sampling_interval": "Percentiles of ordinary participant-bootstrap fractional-estimator distribution; fixed scenario",
                     "scenario_interval": "Percentiles of bootstrap-first single-random-completion distribution; not a confidence interval for fractional point estimate",
                     "allocation": "p_i=expit(a+lambda*original_steps_z_i), intercept recalibrated to fixed q in each bootstrap",
                     "missingness_assumptions": "Hypothetical two-setting assignments for 455 only; no recovered labels or estimated missingness mechanism",
                     "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
                     "input_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in
                                      ("final_analytic_cohort_with_habits.csv", "m1_standardization_parameters.csv", "m1_primary_pairwise_results_python.csv")},
                     "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    summary.to_csv(output / "setting_reassignment_summary.csv", index=False)
    pd.DataFrame(probability_rows).to_csv(output / "setting_reassignment_scenarios.csv", index=False)
    # Aggregate coefficients only; no participant IDs, features or latent labels.
    np.savez_compressed(output / "setting_reassignment_bootstrap_aggregate.npz",
                        sampling_log_OR=sample_draws, scenario_log_OR=assigned_draws)
    (output / "setting_reassignment_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    write_report(summary, reference, metadata, output)
    create_plot(summary, output)
    print(f"Completed; aggregate results in {output}", flush=True)
    return summary, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    run_analysis(args.data_dir, args.output_dir, args.bootstrap, args.workers, args.seed)
