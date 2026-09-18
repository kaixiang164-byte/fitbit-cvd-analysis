#!/usr/bin/env python3
"""Post-hoc age-form sensitivity for M1 and case-only recording-setting models.

Only age's functional form changes. This does not refit mediation, redefine
the upstream cohort, or replace any existing primary result or manuscript.
The notebook invokes run_analysis() and displays its aggregate outputs.
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
import patsy
import scipy
import statsmodels
import statsmodels.api as sm
from scipy.stats import chi2, norm

import run_robustness_and_figure2 as primary

BASE_DIR = Path(__file__).resolve().parent
SPLINE_DF = 3


def build_designs(data):
    """Fit the age basis once in the pooled cohort; subset it for case-only."""
    required = ["avg_steps", "age", "sex_at_birth", "is_smoker", "is_drinker",
                "Outcome_Status"]
    if data[required].isna().any().any():
        raise ValueError("Missing model values: do not silently change the cohort.")
    for column, allowed in [("sex_at_birth", {"Female", "Male"}),
                            ("is_smoker", {0, 1}), ("is_drinker", {0, 1, 2}),
                            ("Outcome_Status", {0, 1, 2})]:
        if not set(data[column]).issubset(allowed):
            raise ValueError(f"Unexpected category in {column}.")
    means = data[["avg_steps", "age"]].mean()
    sds = data[["avg_steps", "age"]].std(ddof=1)
    if not np.isfinite(sds).all() or (sds <= 0).any():
        raise ValueError("Invalid pooled scale.")
    steps = (data["avg_steps"] - means["avg_steps"]) / sds["avg_steps"]
    age = (data["age"] - means["age"]) / sds["age"]
    shared = pd.DataFrame({
        "Intercept": 1.0, "steps_z": steps,
        "sex_male": data["sex_at_birth"].eq("Male").astype(float),
        "smoking_code_1": data["is_smoker"].eq(1).astype(float),
        "drinking_code_1": data["is_drinker"].eq(1).astype(float),
        "drinking_code_2": data["is_drinker"].eq(2).astype(float),
    }, index=data.index)
    # Two interior knots at equally spaced quantiles and boundary knots at
    # pooled observed min/max. Explicit knots make the design reproducible.
    knots_z = np.quantile(age, [0, 1 / 3, 2 / 3, 1])
    if len(np.unique(knots_z)) != 4:
        raise ValueError("Age has insufficient distinct quantiles for this basis.")
    age_basis = patsy.dmatrix(
        'cr(age_z, knots=interior, lower_bound=lower, upper_bound=upper, '
        'constraints="center") - 1',
        {"age_z": age, "interior": knots_z[1:-1],
         "lower": knots_z[0], "upper": knots_z[-1]},
        return_type="dataframe", NA_action="raise",
    )
    if age_basis.shape[1] != SPLINE_DF:
        raise AssertionError("Expected three nonconstant age spline terms.")
    age_basis.columns = [f"age_spline_{i + 1}" for i in range(SPLINE_DF)]
    linear = shared.copy()
    linear.insert(2, "age_z", age)
    spline = pd.concat([shared.iloc[:, :2], age_basis, shared.iloc[:, 2:]], axis=1)
    if not linear.index.equals(data.index) or not spline.index.equals(data.index):
        raise AssertionError("Design row alignment changed.")
    metadata = {
        "steps_mean": float(means["avg_steps"]), "steps_SD": float(sds["avg_steps"]),
        "age_mean": float(means["age"]), "age_SD": float(sds["age"]),
        "scale_ddof": 1, "age_spline_nonconstant_df": SPLINE_DF,
        "age_knot_quantiles": [0, 1 / 3, 2 / 3, 1],
        "age_knots_years": (knots_z * sds["age"] + means["age"]).tolist(),
        "age_knots_z": knots_z.tolist(),
        "age_basis": "Natural cubic; pooled centering constraint; shared pooled knots",
        "case_only_basis": "Subset rows of the pooled design; never refit knots or scales",
        "linear_design_columns": linear.columns.tolist(),
        "spline_design_columns": spline.columns.tolist(),
    }
    return linear, spline, metadata


def fit_checked(outcome, design, multinomial):
    values = np.asarray(design, dtype=float)
    rank = int(np.linalg.matrix_rank(values))
    if not np.isfinite(values).all() or rank != values.shape[1]:
        raise ValueError("Non-finite or rank-deficient model design.")
    model = (sm.MNLogit if multinomial else sm.Logit)(outcome, design, missing="raise")
    result = model.fit(method="newton", maxiter=250, disp=False)
    covariance = np.asarray(result.cov_params(), dtype=float)
    if not result.mle_retvals.get("converged", False):
        raise RuntimeError("Model did not converge.")
    if not (np.isfinite(result.llf) and np.isfinite(np.asarray(result.params)).all()
            and np.isfinite(covariance).all()):
        raise RuntimeError("Non-finite likelihood, coefficients, or covariance.")
    if not np.allclose(covariance, covariance.T, atol=1e-10):
        raise AssertionError("Covariance is not symmetric.")
    if np.linalg.eigvalsh(covariance).min() <= 0:
        raise RuntimeError("Covariance is not positive definite.")
    k = rank * (model.J - 1 if multinomial else 1)
    if not np.isclose(result.aic, -2 * result.llf + 2 * k, atol=1e-8, rtol=1e-12):
        raise AssertionError("AIC disagrees with effective parameter count.")
    return result, rank, k


def step_rows(result, model_label, age_form, data, metadata):
    """Multinomial covariance uses outcome-equation-major coefficient ordering."""
    params = np.asarray(result.params, dtype=float)
    covariance = np.asarray(result.cov_params(), dtype=float)
    index = result.model.exog_names.index("steps_z")
    k = len(result.model.exog_names)
    if params.ndim == 2:
        if params.shape != (k, 2) or list(result.model._ynames_map.values()) != ["0", "1", "2"]:
            raise AssertionError("Unexpected multinomial outcome coding.")
        targets = [
            ("Outpatient vs control", params[index, 0], covariance[index, index]),
            ("Acute vs control", params[index, 1], covariance[k + index, k + index]),
            ("Acute vs outpatient", params[index, 1] - params[index, 0],
             covariance[k + index, k + index] + covariance[index, index]
             - 2 * covariance[k + index, index]),
        ]
        if not np.allclose(np.diag(covariance), np.asarray(result.bse).T.ravel() ** 2):
            raise AssertionError("Covariance ordering does not match equation-major SEs.")
    else:
        targets = [("Acute vs outpatient", params[index], covariance[index, index])]
    rows = []
    for contrast, estimate, variance in targets:
        if variance <= 0:
            raise RuntimeError("Nonpositive contrast variance.")
        se = np.sqrt(variance)
        rows.append({
            "model": model_label, "age_form": age_form, "contrast": contrast,
            "log_OR": float(estimate), "SE": float(se), "OR": float(np.exp(estimate)),
            "CI_lower": float(np.exp(estimate - norm.ppf(0.975) * se)),
            "CI_upper": float(np.exp(estimate + norm.ppf(0.975) * se)),
            "p_value": float(2 * norm.sf(abs(estimate / se))),
            "n": len(data), "n_control": int(data["Outcome_Status"].eq(0).sum()),
            "n_outpatient": int(data["Outcome_Status"].eq(1).sum()),
            "n_acute": int(data["Outcome_Status"].eq(2).sum()),
            "steps_SD": metadata["steps_SD"], "converged": True,
        })
    return rows


def compare_nested(linear_fit, spline_fit, label):
    linear, lr, lk = linear_fit
    spline, sr, sk = spline_fit
    if (linear.nobs != spline.nobs
            or not np.array_equal(linear.model.endog, spline.model.endog)
            or list(linear.model.data.row_labels) != list(spline.model.data.row_labels)):
        raise AssertionError("Nested comparison changed observations or outcomes.")
    if np.linalg.matrix_rank(np.column_stack([spline.model.exog, linear.model.exog])) != sr:
        raise AssertionError("Linear-age model is not nested in the spline-age model.")
    statistic = float(2 * (spline.llf - linear.llf))
    df = sk - lk
    if statistic < -1e-8 or df <= 0:
        raise AssertionError("Invalid likelihood-ratio comparison.")
    statistic = max(0.0, statistic)
    if not np.isclose(spline.aic - linear.aic, -statistic + 2 * df, atol=1e-8):
        raise AssertionError("AIC and LR comparisons disagree.")
    return {
        "model": label, "n": int(linear.nobs), "linear_rank": lr, "spline_rank": sr,
        "linear_parameter_count": lk, "spline_parameter_count": sk,
        "LR_chi2": statistic, "LR_df": df, "LR_p_value": float(chi2.sf(statistic, df)),
        "linear_AIC": float(linear.aic), "spline_AIC": float(spline.aic),
        "linear_log_likelihood": float(linear.llf), "spline_log_likelihood": float(spline.llf),
        "same_rows_and_outcomes": True, "full_rank_and_nested": True,
        "both_converged": True,
    }


def analyze(data):
    """Pure model computation; no files written and no primary globals changed."""
    linear, spline, metadata = build_designs(data)
    rows, diagnostic_rows = [], []
    for label, keep, multinomial in [
        ("M1 multinomial", np.ones(len(data), dtype=bool), True),
        ("Case-only binary", data["Outcome_Status"].isin([1, 2]), False),
    ]:
        subset = data.loc[keep]
        outcome = subset["Outcome_Status"] if multinomial else subset["Outcome_Status"].eq(2).astype(int)
        linear_fit = fit_checked(outcome, linear.loc[keep], multinomial)
        spline_fit = fit_checked(outcome, spline.loc[keep], multinomial)
        for form, fit in [("Linear age", linear_fit), ("Natural cubic spline age", spline_fit)]:
            rows.extend(step_rows(fit[0], label, form, subset, metadata))
        diagnostic_rows.append(compare_nested(linear_fit, spline_fit, label))
    estimates = pd.DataFrame(rows)
    keys = ["model", "contrast", "n"]
    values = ["OR", "CI_lower", "CI_upper", "p_value", "log_OR"]
    comparison = estimates.loc[estimates.age_form.eq("Linear age"), keys + values].merge(
        estimates.loc[estimates.age_form.eq("Natural cubic spline age"), keys + values],
        on=keys, suffixes=("_linear", "_spline"), validate="one_to_one",
    )
    comparison["OR_ratio_spline_to_linear"] = comparison.OR_spline / comparison.OR_linear
    comparison["log_OR_difference"] = comparison.log_OR_spline - comparison.log_OR_linear
    return {"estimates": estimates, "comparison": comparison,
            "diagnostics": pd.DataFrame(diagnostic_rows), "metadata": metadata}


def verify_linear_reproduction(estimates):
    """Verify against saved primary results before interpreting the sensitivity."""
    reference = pd.read_csv(BASE_DIR / "manuscript_multinomial_pairwise_results.csv")
    reference = reference.loc[reference.model.eq("M1_primary") & reference.term.eq("steps_z")]
    linear = estimates.loc[estimates.age_form.eq("Linear age")]
    checks = []
    for row in linear.loc[linear.model.eq("M1 multinomial")].itertuples():
        saved = reference.loc[reference.contrast.eq(row.contrast)].iloc[0]
        parameter_error = max(abs(getattr(row, key) - saved[key])
                              for key in ["OR", "log_OR", "SE"])
        # The legacy R/notebook exports rounded the critical value to 1.96.
        # Harmonise to the exact normal quantile used in all new rows before
        # verifying CIs; otherwise a ~2e-6 endpoint difference is expected.
        expected_ci = np.exp(saved.log_OR + np.array([-1, 1]) * norm.ppf(0.975) * saved.SE)
        error = max(abs(row.CI_lower - expected_ci[0]), abs(row.CI_upper - expected_ci[1]))
        checks.append({"model": row.model, "contrast": row.contrast,
                       "max_absolute_parameter_difference": float(parameter_error),
                       "max_absolute_harmonised_CI_difference": float(error),
                       "legacy_CI_critical_value": 1.96})
    saved_case = pd.read_csv(BASE_DIR / "case_only_acute_vs_outpatient_results.csv")
    saved_case = saved_case.loc[saved_case.term.eq("Mean daily Steps, per pooled SD")].iloc[0]
    row = linear.loc[linear.model.eq("Case-only binary")].iloc[0]
    parameter_error = max(abs(row[key] - saved_case[key]) for key in ["OR", "log_OR", "SE"])
    error = max(abs(row[key] - saved_case[key]) for key in ["CI_lower", "CI_upper"])
    checks.append({"model": row["model"], "contrast": row["contrast"],
                   "max_absolute_parameter_difference": float(parameter_error),
                   "max_absolute_harmonised_CI_difference": float(error),
                   "legacy_CI_critical_value": float(norm.ppf(0.975))})
    if any(item[key] > 1e-7 for item in checks
           for key in ["max_absolute_parameter_difference", "max_absolute_harmonised_CI_difference"]):
        raise AssertionError("Linear models failed to reproduce existing primary results.")
    return checks


def run_analysis(output_dir=BASE_DIR):
    data = primary.load_analysis_data()
    result = analyze(data)
    result["metadata"].update({
        "analysis": "Post-hoc age-form sensitivity; no Steps spline or interaction",
        "cohort": "Existing upstream-restricted t1 landmark export; unchanged",
        "counts": {str(k): int(v) for k, v in data.Outcome_Status.value_counts().sort_index().items()},
        "linear_reference_checks": verify_linear_reproduction(result["estimates"]),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(primary.DATA_FILE.read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "patsy": patsy.__version__,
                     "scipy": scipy.__version__, "statsmodels": statsmodels.__version__},
        "interpretation": "Compare Steps estimates; LR p-values do not select a preferred result. "
                          "No causal interpretation; no change to mediation or main manuscript.",
    })
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for key in ["estimates", "comparison", "diagnostics"]:
            result[key].to_csv(destination / f"age_adjustment_sensitivity_{key}.csv", index=False)
        (destination / "age_adjustment_sensitivity_metadata.json").write_text(
            json.dumps(result["metadata"], indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR)
    arguments = parser.parse_args()
    results = run_analysis(arguments.output_dir)
    print(results["comparison"].to_string(index=False))
    print(results["diagnostics"].to_string(index=False))
