#!/usr/bin/env python3
"""Unadjusted and M1-adjusted Steps associations on the identical analytic sample.

This STROBE reporting supplement does not change primary models or mediation.
Restricted participant-level inputs stay local; only aggregate outputs are saved.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.api as sm
from scipy.stats import norm


BASE_DIR = Path(__file__).resolve().parent
EXPECTED_COUNTS = {0: 5780, 1: 2127, 2: 385}
MODEL_FIELDS = ["avg_steps", "age", "sex_at_birth", "is_smoker", "is_drinker"]
MEDIATOR_FIELDS = [
    "has_sleep_disorder", "has_depression", "has_anxiety", "has_hypertension",
    "has_diabetes", "has_hyperlipidemia", "has_high_cholesterol", "has_ckd",
]
CONTRASTS = ["Outpatient vs control", "Acute vs control", "Acute vs outpatient"]


def validate_export_categories(data):
    """Reject inconsistent group/setting labels before any outcome selection."""
    if data["Group"].isna().any() or not set(data["Group"]).issubset({"Control", "Heart Disease"}):
        raise ValueError("Unexpected or missing original Group category.")
    controls = data["Group"].eq("Control")
    case_settings = {"Chronic (Office/Outpatient)", "Acute (Hospital/ER)", "Other/Unknown"}
    if data.loc[controls, "Onset_Type"].isin(case_settings).any():
        raise ValueError("Control participants must not carry a case setting.")
    if not set(data.loc[controls, "Onset_Type"].dropna()).issubset({"N/A (Control)"}):
        raise ValueError("Unexpected setting label among controls.")
    if not set(data.loc[~controls, "Onset_Type"].dropna()).issubset(case_settings):
        raise ValueError("Unexpected setting label among Heart Disease cases.")


def load_cohort(path):
    """Select known settings and verify complete cases; never silently drop rows."""
    original = pd.read_csv(path)
    validate_export_categories(original)
    keep = original["Group"].eq("Control") | original["Onset_Type"].isin(
        ["Chronic (Office/Outpatient)", "Acute (Hospital/ER)"]
    )
    data = original.loc[keep].copy()
    data["Outcome_Status"] = np.select(
        [data["Group"].eq("Control"),
         data["Onset_Type"].eq("Chronic (Office/Outpatient)"),
         data["Onset_Type"].eq("Acute (Hospital/ER)")], [0, 1, 2], default=-1,
    ).astype(int)
    fields = ["person_id", "Outcome_Status"] + MODEL_FIELDS + MEDIATOR_FIELDS
    missing = pd.DataFrame({
        "variable": fields, "n": len(data),
        "n_missing": [int(data[name].isna().sum()) for name in fields],
    })
    if missing.n_missing.ne(0).any():
        raise ValueError("Classified cohort has missing required fields; no rows were dropped.")
    if not data.person_id.is_unique:
        raise ValueError("Expected one unique participant per row.")
    if data.Outcome_Status.value_counts().to_dict() != EXPECTED_COUNTS:
        raise ValueError("Outcome counts differ from the canonical 8292-person cohort.")
    for name in MEDIATOR_FIELDS:
        if not set(data[name]).issubset({0, 1}):
            raise ValueError("Mediator source fields must be binary.")
    return data, missing, {"n_exported": len(original), "n_classified": len(data),
                           "n_removed_for_missing_required_fields": 0}


def load_scales(path, data):
    table = pd.read_csv(path).set_index("variable")
    scales = {}
    for source, prefix in [("avg_steps", "steps"), ("age", "age")]:
        row = table.loc[source]
        if row.ddof != 1 or row.n != len(data):
            raise ValueError("Original standardisation sample or ddof does not match.")
        if not np.isclose(row["mean"], data[source].mean(), rtol=1e-10, atol=1e-10):
            raise ValueError("Original mean does not match this cohort.")
        if not np.isclose(row.SD, data[source].std(ddof=1), rtol=1e-10, atol=1e-10):
            raise ValueError("Original SD does not match this cohort.")
        scales[prefix + "_mean"] = float(row["mean"])
        scales[prefix + "_SD"] = float(row.SD)
    return scales


def build_designs(data, scales):
    required = MODEL_FIELDS + ["Outcome_Status"]
    if data[required].isna().any().any():
        raise ValueError("Missing model field; the same complete sample is required.")
    for field, allowed in [("sex_at_birth", {"Female", "Male"}),
                           ("is_smoker", {0, 1}), ("is_drinker", {0, 1, 2}),
                           ("Outcome_Status", {0, 1, 2})]:
        if not set(data[field]).issubset(allowed):
            raise ValueError("Unexpected category in " + field)
    if set(data.Outcome_Status) != {0, 1, 2}:
        raise ValueError("All three outcome categories must be present.")
    for prefix in ["steps", "age"]:
        if not np.isfinite([scales[prefix + "_mean"], scales[prefix + "_SD"]]).all():
            raise ValueError("Non-finite original standardisation parameters.")
        if scales[prefix + "_SD"] <= 0:
            raise ValueError("Original standardisation SD must be positive.")
    crude = pd.DataFrame({
        "Intercept": np.ones(len(data)),
        "steps_z": (data.avg_steps - scales["steps_mean"]) / scales["steps_SD"],
    }, index=data.index)
    adjusted = crude.copy()
    adjusted["age_z"] = (data.age - scales["age_mean"]) / scales["age_SD"]
    adjusted["sex_male"] = data.sex_at_birth.eq("Male").astype(float)
    adjusted["smoking_frequency_1"] = data.is_smoker.eq(1).astype(float)
    adjusted["drinking_frequency_1"] = data.is_drinker.eq(1).astype(float)
    adjusted["drinking_frequency_2"] = data.is_drinker.eq(2).astype(float)
    for design in [crude, adjusted]:
        values = design.to_numpy(float)
        if not np.isfinite(values).all() or np.linalg.matrix_rank(values) != values.shape[1]:
            raise ValueError("Non-finite or rank-deficient design.")
    return {"Unadjusted": crude, "M1 adjusted": adjusted}


def fit_model(outcome, design):
    fit = sm.MNLogit(outcome, design, missing="raise").fit(
        method="newton", maxiter=250, disp=False,
    )
    if not fit.mle_retvals.get("converged", False):
        raise RuntimeError("Multinomial model did not converge.")
    covariance = np.asarray(fit.cov_params(), dtype=float)
    if not (np.isfinite(fit.llf) and np.isfinite(fit.params).all().all()
            and np.isfinite(covariance).all()):
        raise RuntimeError("Non-finite model results.")
    if not np.allclose(covariance, covariance.T) or np.linalg.eigvalsh(covariance).min() <= 0:
        raise RuntimeError("Invalid coefficient covariance matrix.")
    return fit


def steps_contrasts(fit, label, data, scales):
    """Use the full cross-logit covariance for acute versus outpatient."""
    k = len(fit.model.exog_names)
    index = fit.model.exog_names.index("steps_z")
    params = np.asarray(fit.params, dtype=float)
    covariance = np.asarray(fit.cov_params(), dtype=float)
    if params.shape != (k, 2) or list(fit.model._ynames_map.values()) != ["0", "1", "2"]:
        raise AssertionError("Unexpected outcome/equation coding.")
    if not np.allclose(np.diag(covariance), np.asarray(fit.bse).T.ravel() ** 2):
        raise AssertionError("Covariance ordering differs from equation-major ordering.")
    beta = params.T.ravel()
    weights = []
    for first, second in [(1, 0), (0, 1), (-1, 1)]:
        vector = np.zeros(2 * k)
        vector[index], vector[k + index] = first, second
        weights.append(vector)
    rows = []
    for contrast, vector in zip(CONTRASTS, weights):
        estimate = float(vector @ beta)
        variance = float(vector @ covariance @ vector)
        if variance <= 0:
            raise RuntimeError("Contrast variance must be positive.")
        se = np.sqrt(variance)
        rows.append({
            "model": label, "contrast": contrast, "log_OR": estimate, "SE": se,
            "OR": float(np.exp(estimate)),
            "CI_lower": float(np.exp(estimate - norm.ppf(.975) * se)),
            "CI_upper": float(np.exp(estimate + norm.ppf(.975) * se)),
            "p_value": float(2 * norm.sf(abs(estimate / se))),
            "n": len(data), "n_control": int(data.Outcome_Status.eq(0).sum()),
            "n_outpatient": int(data.Outcome_Status.eq(1).sum()),
            "n_acute": int(data.Outcome_Status.eq(2).sum()),
            "steps_SD": scales["steps_SD"],
        })
    return rows


def analyze(data, scales):
    rows, diagnostics = [], []
    for label, design in build_designs(data, scales).items():
        fit = fit_model(data.Outcome_Status, design)
        rows.extend(steps_contrasts(fit, label, data, scales))
        diagnostics.append({"model": label, "n": int(fit.nobs), "converged": True,
                            "log_likelihood": float(fit.llf), "AIC": float(fit.aic),
                            "design_rank": int(np.linalg.matrix_rank(design)),
                            "parameters": int(fit.params.size)})
    return pd.DataFrame(rows), diagnostics


def verify_adjusted(estimates, reference):
    saved = reference.loc[reference.term.eq("steps_z")]
    current = estimates.loc[estimates.model.eq("M1 adjusted")]
    checks = []
    for row in current.itertuples():
        match = saved.loc[saved.contrast.eq(row.contrast)]
        if len(match) != 1:
            raise ValueError("Expected exactly one saved primary Steps estimate per contrast.")
        old = match.iloc[0]
        error = max(abs(getattr(row, field) - old[field])
                    for field in ["log_OR", "SE", "OR", "p_value"])
        # Existing output used 1.96; calculate reference limits using the same
        # exact normal quantile as this supplement before comparison.
        ci = np.exp(old.log_OR + np.array([-1, 1]) * norm.ppf(.975) * old.SE)
        ci_error = max(abs(row.CI_lower - ci[0]), abs(row.CI_upper - ci[1]))
        if error > 1e-8 or ci_error > 1e-8 or row.n != old.n:
            raise AssertionError("Adjusted model failed to reproduce saved primary results.")
        checks.append({"contrast": row.contrast, "max_parameter_error": float(error),
                       "max_harmonised_CI_error": float(ci_error)})
    return checks


def render_s15(estimates):
    def value(row):
        return f"{row.OR:.3f} ({row.CI_lower:.3f}--{row.CI_upper:.3f})"

    def p_value(row):
        return "$<0.001$" if row.p_value < .001 else f"{row.p_value:.3f}"

    lines = []
    for contrast in CONTRASTS:
        crude = estimates.loc[estimates.model.eq("Unadjusted") & estimates.contrast.eq(contrast)].iloc[0]
        adjusted = estimates.loc[estimates.model.eq("M1 adjusted") & estimates.contrast.eq(contrast)].iloc[0]
        label = contrast.replace("Acute", "Acute care").replace("vs control", "vs no recorded CVD")
        lines.append(f"{label} & {value(crude)} & {p_value(crude)} & "
                     f"{value(adjusted)} & {p_value(adjusted)} \\\\")
    first = estimates.iloc[0]
    return r"""% Generated by run_crude_adjusted_steps_comparison.py from aggregate model estimates.
\documentclass[11pt]{article}
\usepackage[margin=0.7in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{booktabs,array,microtype}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.65em}
\begin{document}
\section*{S15 Table. Unadjusted and adjusted associations between mean daily Steps and CVD recording setting}

Odds ratios (ORs) and 95\% Wald confidence intervals (CIs) are reported per
1 original pooled SD higher mean daily Steps. Both three-category multinomial
models were fitted to the same complete analytic sample.

{\small\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.3}
\begin{tabular}{@{}lcccc@{}}
\toprule
& \multicolumn{2}{c}{Unadjusted} & \multicolumn{2}{c}{M1 adjusted} \\
\cmidrule(lr){2-3}\cmidrule(l){4-5}
Comparison & OR (95\% CI) & $p$ & OR (95\% CI) & $p$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}

{\footnotesize
""" + (
        f"Both models included {int(first.n)} participants: {int(first.n_control)} participants without recorded CVD, "
        f"{int(first.n_outpatient)} outpatient and {int(first.n_acute)} acute-care cases. "
        f"The original pooled Steps SD was {first.steps_SD:.2f} steps/day. "
    ) + r"""The unadjusted model included Steps only; M1 additionally included linear
standardised age, sex at birth, smoking frequency and drinking frequency.
The acute-care-versus-outpatient contrast was calculated from each fitted
multinomial model using the full covariance between its outcome equations,
not by comparing the significance of two estimates using no recorded CVD as the reference.
There were no missing values in the final sample for the outcome, exposure,
adjustment variables or the eight source indicators of the health-burden score.
Differences between unadjusted and adjusted ORs are descriptive and are not a
percentage of confounding removed; ORs are non-collapsible. All $p$ values are
two-sided. CVD=cardiovascular disease.}
\end{document}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=BASE_DIR / "final_analytic_cohort_with_habits.csv")
    parser.add_argument("--scales", type=Path, default=BASE_DIR / "m1_standardization_parameters.csv")
    parser.add_argument("--reference", type=Path, default=BASE_DIR / "m1_primary_pairwise_results_python.csv")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "crude_adjusted_steps_outputs")
    parser.add_argument("--table", type=Path, default=BASE_DIR / "S15_Table.tex")
    args = parser.parse_args()
    data, missing, metadata = load_cohort(args.data)
    scales = load_scales(args.scales, data)
    estimates, diagnostics = analyze(data, scales)
    checks = verify_adjusted(estimates, pd.read_csv(args.reference))
    metadata.update({
        "analysis": "STROBE 16(a): unadjusted and M1-adjusted Steps associations",
        "same_sample_and_original_standardisation": True,
        "scales": scales, "model_diagnostics": diagnostics, "primary_reproduction": checks,
        "CI_method": "Wald; exact standard normal 97.5th percentile",
        "upstream_cohort_changed": False, "mediation_refitted": False,
        "input_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in [args.data, args.scales, args.reference]},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "statsmodels": statsmodels.__version__},
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(args.output_dir / "crude_adjusted_steps_estimates.csv", index=False)
    missing.to_csv(args.output_dir / "final_analytic_required_fields_missingness.csv", index=False)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    args.table.write_text(render_s15(estimates), encoding="utf-8")
    print(estimates[["model", "contrast", "OR", "CI_lower", "CI_upper", "p_value", "n"]]
          .to_string(index=False))
    print("All required final-sample fields have zero missing values; adjusted primary results reproduced.")
    print("Only aggregate outputs were written. No participant rows or identifiers were exported.")


if __name__ == "__main__":
    main()
