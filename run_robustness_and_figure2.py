#!/usr/bin/env python3
"""Reproduce manuscript robustness analyses and the main results Figure 2."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import hashlib
import json
import platform
import shutil
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2
import scipy
import statsmodels

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "final_analytic_cohort_with_habits.csv"

VARIANTS = (
    "Primary PCA",
    "Equal-weight composite",
    "PCA with deduplicated lipid burden",
)
PAIRWISE_TARGETS = (
    "Outpatient vs control",
    "Acute vs control",
    "Acute vs outpatient",
)
RAW_TARGETS = (
    "Control probability",
    "Outpatient probability",
    "Acute probability",
)
EFFECTS = ("ACME", "ADE", "Total")

ANALYSIS = None
REFERENCE_LOADINGS = None
STEPS_MEAN = None
STEPS_SD = None
AGE_MEAN = None
AGE_SD = None
BOOTSTRAP_RESIDUAL_NODES = 51
CANONICAL_VERSION = "pooled-mediation-v2"
EXPECTED_COUNTS = {0: 5780, 1: 2127, 2: 385}


def z_standardize(values):
    values = np.asarray(values, dtype=float)
    standard_deviation = values.std(ddof=1)
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        raise ValueError("Cannot standardize a variable with invalid or zero SD.")
    return (values - values.mean()) / standard_deviation


def load_analysis_data():
    data = pd.read_csv(DATA_FILE)
    keep = (
        data["Group"].eq("Control")
        | data["Onset_Type"].eq("Chronic (Office/Outpatient)")
        | data["Onset_Type"].eq("Acute (Hospital/ER)")
    )
    data = data.loc[keep].copy()
    data["Outcome_Status"] = np.select(
        [
            data["Group"].eq("Control"),
            data["Onset_Type"].eq("Chronic (Office/Outpatient)"),
            data["Onset_Type"].eq("Acute (Hospital/ER)"),
        ],
        [0, 1, 2],
        default=np.nan,
    )
    diagnosis_columns = [
        "has_sleep_disorder",
        "has_depression",
        "has_anxiety",
        "has_hypertension",
        "has_diabetes",
        "has_hyperlipidemia",
        "has_high_cholesterol",
        "has_ckd",
    ]
    required_columns = [
        "person_id",
        "Outcome_Status",
        "avg_steps",
        "age",
        "sex_at_birth",
        "is_smoker",
        "is_drinker",
    ] + diagnosis_columns
    data = data.dropna(subset=required_columns).copy()
    data["Outcome_Status"] = data["Outcome_Status"].astype(int)

    for column in diagnosis_columns:
        observed = set(data[column].unique())
        if not observed.issubset({0, 1}):
            raise ValueError(f"{column} must be binary; observed {observed}.")

    if set(data["Outcome_Status"]) != {0, 1, 2}:
        raise ValueError("All three outcome categories must be present.")
    if not data["person_id"].is_unique:
        raise ValueError("The analysis requires one row per person.")
    if data["Outcome_Status"].value_counts().to_dict() != EXPECTED_COUNTS:
        raise ValueError("Unexpected cohort change: canonical analysis requires 8292 participants.")

    data["Sleep"] = data["has_sleep_disorder"].astype(int)
    data["Mental"] = data[["has_depression", "has_anxiety"]].sum(axis=1)
    data["Clinical"] = data[
        [
            "has_hypertension",
            "has_diabetes",
            "has_hyperlipidemia",
            "has_high_cholesterol",
            "has_ckd",
        ]
    ].sum(axis=1)
    data["lipid_any"] = data[
        ["has_hyperlipidemia", "has_high_cholesterol"]
    ].max(axis=1)
    data["Clinical_deduplicated"] = data[
        ["has_hypertension", "has_diabetes", "lipid_any", "has_ckd"]
    ].sum(axis=1)

    return data


def compute_reference_loadings(data):
    references = {}
    for variant, clinical_name in (
        ("Primary PCA", "Clinical"),
        ("PCA with deduplicated lipid burden", "Clinical_deduplicated"),
    ):
        raw = data[["Sleep", "Mental", clinical_name]].to_numpy(float)
        standardized = np.column_stack(
            [z_standardize(raw[:, index]) for index in range(raw.shape[1])]
        )
        _, _, right_vectors = np.linalg.svd(standardized, full_matrices=False)
        loading = right_vectors[0].copy()
        scores = standardized @ loading
        if np.corrcoef(scores, standardized.sum(axis=1))[0, 1] < 0:
            loading *= -1
        references[variant] = loading
    return references


def construct_vulnerability(sample, variant):
    clinical_name = (
        "Clinical_deduplicated"
        if variant == "PCA with deduplicated lipid burden"
        else "Clinical"
    )
    raw = sample[["Sleep", "Mental", clinical_name]].to_numpy(float)
    standardized = np.column_stack(
        [z_standardize(raw[:, index]) for index in range(raw.shape[1])]
    )

    if variant == "Equal-weight composite":
        return z_standardize(standardized.mean(axis=1))

    _, _, right_vectors = np.linalg.svd(standardized, full_matrices=False)
    loading = right_vectors[0].copy()
    if float(loading @ REFERENCE_LOADINGS[variant]) < 0:
        loading *= -1
    return z_standardize(standardized @ loading)


def covariate_arrays(sample):
    steps_z = (sample["avg_steps"].to_numpy(float) - STEPS_MEAN) / STEPS_SD
    age_z = (sample["age"].to_numpy(float) - AGE_MEAN) / AGE_SD
    sex_male = sample["sex_at_birth"].eq("Male").to_numpy(float)
    smoker = sample["is_smoker"].eq(1).to_numpy(float)
    drink_moderate = sample["is_drinker"].eq(1).to_numpy(float)
    drink_heavy = sample["is_drinker"].eq(2).to_numpy(float)
    return steps_z, age_z, sex_male, smoker, drink_moderate, drink_heavy


def fit_multinomial(outcome, design):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method, iterations in (("newton", 150), ("lbfgs", 600)):
            try:
                fitted = sm.MNLogit(outcome, design).fit(
                    method=method, maxiter=iterations, disp=False
                )
            except Exception:
                continue
            if bool(fitted.mle_retvals.get("converged", False)):
                parameters = np.asarray(fitted.params, dtype=float)
                if parameters.shape == (design.shape[1], 2):
                    return parameters
    raise RuntimeError("Multinomial model did not converge.")


def integrated_probabilities(base_eta, mediator_slopes, residual_nodes):
    raw_sum = np.zeros(3, dtype=float)
    pairwise_sum = np.zeros(3, dtype=float)

    for residual in residual_nodes:
        eta = base_eta + residual * mediator_slopes[None, :]
        maximum = np.maximum(0.0, np.max(eta, axis=1))
        exp_control = np.exp(-maximum)
        exp_outpatient = np.exp(eta[:, 0] - maximum)
        exp_acute = np.exp(eta[:, 1] - maximum)
        denominator = exp_control + exp_outpatient + exp_acute
        probabilities = np.column_stack(
            [
                exp_control / denominator,
                exp_outpatient / denominator,
                exp_acute / denominator,
            ]
        )
        raw_sum += probabilities.mean(axis=0)
        pairwise_sum += np.array(
            [
                np.mean(
                    probabilities[:, 1]
                    / (probabilities[:, 0] + probabilities[:, 1])
                ),
                np.mean(
                    probabilities[:, 2]
                    / (probabilities[:, 0] + probabilities[:, 2])
                ),
                np.mean(
                    probabilities[:, 2]
                    / (probabilities[:, 1] + probabilities[:, 2])
                ),
            ]
        )

    divisor = float(len(residual_nodes))
    return pairwise_sum / divisor, raw_sum / divisor


def effects_for_variant(positions, variant, residual_node_count):
    sample = ANALYSIS.iloc[positions]
    n_sample = len(sample)
    steps_z, age_z, sex_male, smoker, drink_moderate, drink_heavy = (
        covariate_arrays(sample)
    )
    vulnerability = construct_vulnerability(sample, variant)

    mediator_design = np.column_stack(
        [
            np.ones(n_sample),
            steps_z,
            age_z,
            sex_male,
            smoker,
            drink_moderate,
            drink_heavy,
        ]
    )
    mediator_parameters, _, rank, _ = np.linalg.lstsq(
        mediator_design, vulnerability, rcond=None
    )
    if rank != mediator_design.shape[1]:
        raise RuntimeError("Rank-deficient mediator model.")

    residuals = vulnerability - mediator_design @ mediator_parameters
    residuals -= residuals.mean()
    probabilities = (np.arange(residual_node_count) + 0.5) / residual_node_count
    residual_nodes = np.quantile(residuals, probabilities)

    outcome_design = np.column_stack(
        [
            np.ones(n_sample),
            steps_z,
            vulnerability,
            age_z,
            sex_male,
            smoker,
            drink_moderate,
            drink_heavy,
        ]
    )
    outcome_parameters = fit_multinomial(
        sample["Outcome_Status"].to_numpy(int), outcome_design
    )
    mediator_slopes = outcome_parameters[2, :]

    def mediator_mean(exposure):
        design = mediator_design.copy()
        design[:, 1] = exposure
        return design @ mediator_parameters

    mediator_a0 = mediator_mean(0.0)
    mediator_a1 = mediator_mean(1.0)

    def theta(outcome_exposure, mediator_values):
        design = outcome_design.copy()
        design[:, 1] = outcome_exposure
        design[:, 2] = mediator_values
        base_eta = design @ outcome_parameters
        return integrated_probabilities(
            base_eta, mediator_slopes, residual_nodes
        )

    q00_pair, q00_raw = theta(0.0, mediator_a0)
    q01_pair, q01_raw = theta(0.0, mediator_a1)
    q10_pair, q10_raw = theta(1.0, mediator_a0)
    q11_pair, q11_raw = theta(1.0, mediator_a1)

    pairwise_effects = np.column_stack(
        [
            ((q01_pair - q00_pair) + (q11_pair - q10_pair)) / 2,
            ((q10_pair - q00_pair) + (q11_pair - q01_pair)) / 2,
            q11_pair - q00_pair,
        ]
    )
    raw_effects = np.column_stack(
        [
            ((q01_raw - q00_raw) + (q11_raw - q10_raw)) / 2,
            ((q10_raw - q00_raw) + (q11_raw - q01_raw)) / 2,
            q11_raw - q00_raw,
        ]
    )

    if not np.allclose(
        pairwise_effects[:, 0] + pairwise_effects[:, 1],
        pairwise_effects[:, 2],
        atol=1e-9,
    ):
        raise AssertionError("Pairwise ACME + ADE must equal Total.")
    if not np.allclose(
        raw_effects[:, 0] + raw_effects[:, 1],
        raw_effects[:, 2],
        atol=1e-9,
    ):
        raise AssertionError("Raw ACME + ADE must equal Total.")
    if not np.allclose(raw_effects.sum(axis=0), 0.0, atol=1e-9):
        raise AssertionError("Raw category effects must sum to zero.")

    return pairwise_effects, raw_effects


def result_columns():
    columns = []
    for variant in VARIANTS:
        for scale, targets in (
            ("Pairwise-normalized", PAIRWISE_TARGETS),
            ("Raw category probability", RAW_TARGETS),
        ):
            for target in targets:
                for effect in EFFECTS:
                    columns.append(f"{variant}|{scale}|{target}|{effect}")
    return columns


def flatten_effects(positions, residual_node_count):
    values = []
    for variant in VARIANTS:
        pairwise, raw = effects_for_variant(
            positions, variant, residual_node_count
        )
        values.extend(pairwise.reshape(-1))
        values.extend(raw.reshape(-1))
    return np.asarray(values, dtype=float)


def bootstrap_worker(seed):
    try:
        generator = np.random.default_rng(int(seed))
        positions = generator.integers(
            0, len(ANALYSIS), size=len(ANALYSIS)
        )
        values = flatten_effects(positions, BOOTSTRAP_RESIDUAL_NODES)
        return True, "", values
    except Exception as error:
        return False, str(error), np.full(len(result_columns()), np.nan)


def summarize_bootstrap(point_values, draws, columns, requested, seed):
    rows = []
    for column_index, column in enumerate(columns):
        variant, scale, target, effect = column.split("|")
        values = draws[:, column_index]
        values = values[np.isfinite(values)]
        if len(values) == 0:
            raise RuntimeError(f"No successful draws for {column}.")
        lower_tail = (np.sum(values <= 0) + 1) / (len(values) + 1)
        upper_tail = (np.sum(values >= 0) + 1) / (len(values) + 1)
        p_value = min(1.0, 2 * min(lower_tail, upper_tail))
        rows.append(
            {
                "variant": variant,
                "scale": scale,
                "target": target,
                "effect": effect,
                "estimate": point_values[column_index],
                "CI_lower": np.quantile(values, 0.025),
                "CI_upper": np.quantile(values, 0.975),
                "p_value": p_value,
                "estimate_percentage_points": 100 * point_values[column_index],
                "CI_lower_percentage_points": 100 * np.quantile(values, 0.025),
                "CI_upper_percentage_points": 100 * np.quantile(values, 0.975),
                "successful_bootstrap": len(values),
                "requested_bootstrap": requested,
                "seed": seed,
                "exposure_contrast": "Steps_z 0 to 1",
            }
        )
    return pd.DataFrame(rows)


def cholesterol_overlap(data):
    hyperlipidaemia = data["has_hyperlipidemia"].astype(int)
    high_cholesterol = data["has_high_cholesterol"].astype(int)
    both = (hyperlipidaemia.eq(1) & high_cholesterol.eq(1))
    categories = pd.Series(
        np.select(
            [
                hyperlipidaemia.eq(0) & high_cholesterol.eq(0),
                hyperlipidaemia.eq(1) & high_cholesterol.eq(0),
                hyperlipidaemia.eq(0) & high_cholesterol.eq(1),
                both,
            ],
            [
                "Neither",
                "Hyperlipidaemia only",
                "High cholesterol only",
                "Both indicators",
            ],
            default="Unclassified",
        )
    )
    counts = categories.value_counts().reindex(
        [
            "Neither",
            "Hyperlipidaemia only",
            "High cholesterol only",
            "Both indicators",
        ]
    )
    table = counts.rename_axis("indicator_pattern").reset_index(name="n")
    table["percent"] = 100 * table["n"] / len(data)
    table["phi_correlation"] = np.nan
    table.loc[0, "phi_correlation"] = np.corrcoef(
        hyperlipidaemia, high_cholesterol
    )[0, 1]
    table["n_total"] = len(data)
    return table


def vulnerability_diagnostics(data):
    values = {}
    loadings_rows = []
    for variant in VARIANTS:
        values[variant] = construct_vulnerability(data, variant)
        if variant != "Equal-weight composite":
            clinical = (
                "Clinical_deduplicated"
                if variant == "PCA with deduplicated lipid burden"
                else "Clinical"
            )
            raw = data[["Sleep", "Mental", clinical]].to_numpy(float)
            standardized = np.column_stack(
                [z_standardize(raw[:, index]) for index in range(3)]
            )
            score = standardized @ REFERENCE_LOADINGS[variant]
            variance_ratio = np.var(score, ddof=1) / np.sum(
                np.var(standardized, axis=0, ddof=1)
            )
            for name, loading in zip(
                ["Sleep", "Mental", clinical], REFERENCE_LOADINGS[variant]
            ):
                loadings_rows.append(
                    {
                        "variant": variant,
                        "component": name,
                        "weight": loading,
                        "PC1_variance_explained": variance_ratio,
                    }
                )
        else:
            for name in ["Sleep", "Mental", "Clinical"]:
                loadings_rows.append(
                    {
                        "variant": variant,
                        "component": name,
                        "weight": 1 / 3,
                        "PC1_variance_explained": np.nan,
                    }
                )

    frame = pd.DataFrame(values)
    correlations = frame.corr().stack().reset_index()
    correlations.columns = ["variant_1", "variant_2", "correlation"]
    correlations = correlations.loc[
        correlations["variant_1"] < correlations["variant_2"]
    ].reset_index(drop=True)
    return pd.DataFrame(loadings_rows), correlations


def _fit_checked_spline_diagnostic_model(formula, data, label):
    """Fit an identifiable diagnostic model; never hide fitting failures."""
    model = smf.mnlogit(formula, data=data, missing="raise")
    design = np.asarray(model.exog, dtype=float)
    if not np.isfinite(design).all():
        raise ValueError(f"{label} design contains non-finite values.")
    rank = int(np.linalg.matrix_rank(design))
    if rank != design.shape[1]:
        raise ValueError(
            f"{label} design is rank deficient: rank {rank}, "
            f"{design.shape[1]} columns. Check the spline/intercept basis."
        )
    fitted = model.fit(method="newton", maxiter=250, disp=False)
    if not bool(fitted.mle_retvals.get("converged", False)):
        raise RuntimeError(f"{label} did not converge.")
    if not np.isfinite(fitted.llf) or not np.isfinite(fitted.params).all().all():
        raise RuntimeError(f"{label} returned non-finite likelihood or coefficients.")
    try:
        covariance = np.asarray(fitted.cov_params(), dtype=float)
    except (ValueError, np.linalg.LinAlgError) as error:
        raise RuntimeError(f"{label} coefficient covariance is unavailable.") from error
    if not np.isfinite(covariance).all():
        raise RuntimeError(f"{label} coefficient covariance is non-finite.")
    parameter_count = rank * (model.J - 1)
    expected_aic = -2 * fitted.llf + 2 * parameter_count
    if not np.isclose(fitted.aic, expected_aic, rtol=1e-12, atol=1e-8):
        raise AssertionError(f"{label} AIC disagrees with its effective parameter count.")
    return fitted, rank, parameter_count


def fit_spline_test(data):
    """Return the Steps diagnostic without changing files or mediation state.

    Three centered spline columns plus an intercept span the same space as the
    legacy four uncentered columns plus an intercept, but without its redundant
    intercept. Thus each logit adds two, not three, nonlinear degrees of freedom.
    Standardisation uses this supplied cohort; on the canonical cohort it equals
    the primary M1 scale and does not require initialising or fitting PCA.
    """
    data = data.copy()
    data["steps_z"] = z_standardize(data["avg_steps"])
    data["age_z"] = z_standardize(data["age"])
    linear_formula = (
        "Outcome_Status ~ steps_z + age_z + C(sex_at_birth) "
        "+ C(is_smoker) + C(is_drinker)"
    )
    spline_formula = (
        'Outcome_Status ~ cr(steps_z, df=3, constraints="center") '
        "+ age_z + C(sex_at_birth) "
        "+ C(is_smoker) + C(is_drinker)"
    )
    linear, linear_rank, linear_k = _fit_checked_spline_diagnostic_model(
        linear_formula, data, "Linear Steps model"
    )
    spline, spline_rank, spline_k = _fit_checked_spline_diagnostic_model(
        spline_formula, data, "Natural cubic spline Steps model"
    )
    if (linear.nobs != spline.nobs or linear.model.J != spline.model.J
            or not np.array_equal(linear.model.endog, spline.model.endog)
            or list(linear.model.data.row_labels) != list(spline.model.data.row_labels)):
        raise ValueError("The nested models must use identical observations and outcomes.")
    combined_rank = np.linalg.matrix_rank(
        np.column_stack([spline.model.exog, linear.model.exog])
    )
    if combined_rank != spline_rank:
        raise ValueError("The linear Steps model is not nested in the spline model.")
    statistic = float(2 * (spline.llf - linear.llf))
    degrees_freedom = int(spline_k - linear_k)
    if degrees_freedom <= 0 or statistic < -1e-8:
        raise RuntimeError("Invalid nested-model likelihood-ratio comparison.")
    statistic = max(0.0, statistic)
    if not np.isclose(
        spline.aic - linear.aic, -statistic + 2 * degrees_freedom,
        rtol=1e-12, atol=1e-8,
    ):
        raise AssertionError("AIC difference and likelihood-ratio df are inconsistent.")
    p_value = float(chi2.sf(statistic, degrees_freedom))
    return pd.DataFrame(
        [
            {
                "test": "Natural cubic spline versus linear Steps",
                "spline_df": 3,
                "spline_df_including_constant": 4,
                "spline_constraint": "center",
                "n": int(linear.nobs),
                "linear_design_rank": linear_rank,
                "spline_design_rank": spline_rank,
                "linear_parameter_count": linear_k,
                "spline_parameter_count": spline_k,
                "likelihood_ratio_chi2": statistic,
                "df": degrees_freedom,
                "p_value": p_value,
                "linear_log_likelihood": linear.llf,
                "spline_log_likelihood": spline.llf,
                "linear_AIC": linear.aic,
                "spline_AIC": spline.aic,
                "linear_BIC": linear.bic,
                "spline_BIC": spline.bic,
                "spline_converged": bool(
                    spline.mle_retvals.get("converged", False)
                ),
                "selected_curve_model": "Spline" if p_value < 0.05 else "Linear",
            }
        ]
    )


def run_spline_diagnostic_only():
    """Return a fresh diagnostic without refitting PCA/mediation or writing files."""
    return fit_spline_test(load_analysis_data())


def fit_m1_direct(data):
    steps_z, age_z, sex_male, smoker, drink_moderate, drink_heavy = (
        covariate_arrays(data)
    )
    design = np.column_stack(
        [
            np.ones(len(data)),
            steps_z,
            age_z,
            sex_male,
            smoker,
            drink_moderate,
            drink_heavy,
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = sm.MNLogit(
            data["Outcome_Status"].to_numpy(int), design
        ).fit(method="newton", maxiter=250, disp=False)
    if not bool(fitted.mle_retvals.get("converged", False)):
        raise RuntimeError("M1 did not converge.")
    return fitted, design


def fit_case_only_setting_sensitivity(data):
    """Fit acute versus outpatient logistic regression among CVD cases."""
    cases = data.loc[data["Outcome_Status"].isin([1, 2])].copy()
    steps_z, age_z, sex_male, smoker, drink_moderate, drink_heavy = (
        covariate_arrays(cases)
    )
    design = np.column_stack(
        [
            np.ones(len(cases)),
            steps_z,
            age_z,
            sex_male,
            smoker,
            drink_moderate,
            drink_heavy,
        ]
    )
    acute = cases["Outcome_Status"].eq(2).to_numpy(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = sm.Logit(acute, design).fit(
            method="newton", maxiter=250, disp=False
        )
    if not bool(fitted.mle_retvals.get("converged", False)):
        raise RuntimeError(
            "Case-only acute-versus-outpatient model did not converge."
        )

    terms = (
        "Intercept",
        "Mean daily Steps, per pooled SD",
        "Age, per pooled SD",
        "Male sex",
        "Current smoker",
        "Moderate alcohol consumption",
        "Heavy alcohol consumption",
    )
    confidence_intervals = np.asarray(
        fitted.conf_int(alpha=0.05), dtype=float
    )
    rows = []
    for index, term in enumerate(terms):
        estimate = float(fitted.params[index])
        rows.append(
            {
                "term": term,
                "log_OR": estimate,
                "SE": float(fitted.bse[index]),
                "OR": float(np.exp(estimate)),
                "CI_lower": float(np.exp(confidence_intervals[index, 0])),
                "CI_upper": float(np.exp(confidence_intervals[index, 1])),
                "p_value": float(fitted.pvalues[index]),
                "n": len(cases),
                "outpatient": int((cases["Outcome_Status"] == 1).sum()),
                "acute": int((cases["Outcome_Status"] == 2).sum()),
                "steps_SD": STEPS_SD,
                "age_SD": AGE_SD,
                "outcome_reference": "Outpatient recording",
            }
        )
    return pd.DataFrame(rows)


def softmax_probabilities(design, parameters):
    eta = design @ parameters
    maximum = np.maximum(0.0, np.max(eta, axis=1))
    exp_control = np.exp(-maximum)
    exp_outpatient = np.exp(eta[:, 0] - maximum)
    exp_acute = np.exp(eta[:, 1] - maximum)
    denominator = exp_control + exp_outpatient + exp_acute
    return np.column_stack(
        [
            exp_control / denominator,
            exp_outpatient / denominator,
            exp_acute / denominator,
        ]
    )


def standardized_probability_curves(data, simulation_draws, seed):
    fitted, base_design = fit_m1_direct(data)
    lower, upper = np.quantile(data["avg_steps"], [0.025, 0.975])
    steps_grid = np.linspace(lower, upper, 61)
    parameters = np.asarray(fitted.params, dtype=float)
    k_terms = parameters.shape[0]
    coefficient_mean = parameters.T.reshape(-1)
    covariance = np.asarray(fitted.cov_params(), dtype=float)
    generator = np.random.default_rng(seed)
    coefficient_draws = generator.multivariate_normal(
        coefficient_mean,
        covariance,
        size=simulation_draws,
        method="svd",
    )
    equation_one = coefficient_draws[:, :k_terms]
    equation_two = coefficient_draws[:, k_terms:]

    rows = []
    for steps_value in steps_grid:
        design = base_design.copy()
        design[:, 1] = (steps_value - STEPS_MEAN) / STEPS_SD
        point = softmax_probabilities(design, parameters).mean(axis=0)

        eta_one = design @ equation_one.T
        eta_two = design @ equation_two.T
        maximum = np.maximum(0.0, np.maximum(eta_one, eta_two))
        exp_control = np.exp(-maximum)
        exp_outpatient = np.exp(eta_one - maximum)
        exp_acute = np.exp(eta_two - maximum)
        denominator = exp_control + exp_outpatient + exp_acute
        simulated = np.stack(
            [
                (exp_control / denominator).mean(axis=0),
                (exp_outpatient / denominator).mean(axis=0),
                (exp_acute / denominator).mean(axis=0),
            ],
            axis=1,
        )

        for category_index, category in enumerate(
            ["Control", "Outpatient", "Acute care"]
        ):
            rows.append(
                {
                    "steps_per_day": steps_value,
                    "category": category,
                    "probability": point[category_index],
                    "CI_lower": np.quantile(
                        simulated[:, category_index], 0.025
                    ),
                    "CI_upper": np.quantile(
                        simulated[:, category_index], 0.975
                    ),
                    "CI_method": (
                        "Simulation from asymptotic coefficient covariance"
                    ),
                    "simulation_draws": simulation_draws,
                }
            )
    return pd.DataFrame(rows)


def create_figure(data, curves, spline_test, mediation_summary):
    plt.rcParams.update(
        {
            "font.family": "Nimbus Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    colors = {
        "Control": "#4B5563",
        "Outpatient": "#2563EB",
        "Acute care": "#D94841",
    }
    effect_colors = {
        "ACME": "#E07A1F",
        "ADE": "#2878B5",
        "Total": "#222222",
    }
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.5, 4.2),
        gridspec_kw={"width_ratios": [1.12, 1.0]},
    )
    ax = axes[0]
    x_min = curves["steps_per_day"].min()
    x_max = curves["steps_per_day"].max()
    histogram_values = data.loc[
        data["avg_steps"].between(x_min, x_max), "avg_steps"
    ].to_numpy(float)

    for category in ["Control", "Outpatient", "Acute care"]:
        subset = curves.loc[curves["category"].eq(category)]
        x = subset["steps_per_day"].to_numpy(float)
        estimate = subset["probability"].to_numpy(float)
        lower = subset["CI_lower"].to_numpy(float)
        upper = subset["CI_upper"].to_numpy(float)
        ax.fill_between(x, lower, upper, color=colors[category], alpha=0.14)
        ax.plot(
            x,
            estimate,
            color=colors[category],
            linewidth=2.4,
            label=category,
        )

    ax.set_title(
        "A. Standardised outcome probabilities",
        loc="left",
        fontweight="bold",
    )
    ax.set_xlabel("Mean daily step count (steps/day)")
    ax.set_ylabel("Standardised probability")
    ax.set_ylim(0, 0.86)
    ax.set_xlim(x_min, x_max)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False, loc="center right")

    distribution_axis = ax.inset_axes([0.0, 0.0, 1.0, 0.13], zorder=0)
    distribution_axis.hist(
        histogram_values,
        bins=28,
        density=True,
        color="#9CA3AF",
        alpha=0.28,
        edgecolor="none",
    )
    distribution_axis.set_xlim(x_min, x_max)
    distribution_axis.set_axis_off()
    distribution_axis.text(
        0.02,
        0.72,
        "Observed step-count distribution",
        transform=distribution_axis.transAxes,
        color="#6B7280",
        fontsize=7.5,
    )

    forest = mediation_summary.loc[
        mediation_summary["variant"].eq("Primary PCA")
        & mediation_summary["scale"].eq("Raw category probability")
        & mediation_summary["effect"].isin(EFFECTS)
    ].copy()
    forest["target"] = pd.Categorical(
        forest["target"], categories=RAW_TARGETS, ordered=True
    )
    forest["effect"] = pd.Categorical(
        forest["effect"], categories=EFFECTS, ordered=True
    )
    forest = forest.sort_values(["target", "effect"])

    ax = axes[1]
    base_positions = {
        "Control probability": 2,
        "Outpatient probability": 1,
        "Acute probability": 0,
    }
    offsets = {"ACME": 0.20, "ADE": 0.0, "Total": -0.20}
    markers = {"ACME": "o", "ADE": "s", "Total": "D"}
    for effect in EFFECTS:
        subset = forest.loc[forest["effect"].eq(effect)]
        y = np.array(
            [base_positions[str(value)] + offsets[effect] for value in subset["target"]]
        )
        estimate = subset["estimate_percentage_points"].to_numpy(float)
        lower = subset["CI_lower_percentage_points"].to_numpy(float)
        upper = subset["CI_upper_percentage_points"].to_numpy(float)
        ax.errorbar(
            estimate,
            y,
            xerr=np.vstack([estimate - lower, upper - estimate]),
            fmt=markers[effect],
            color=effect_colors[effect],
            markersize=5.8,
            capsize=3,
            linewidth=1.4,
            label={"ACME": "Indirect", "ADE": "Direct", "Total": "Total"}[effect],
        )

    ax.axvline(0, color="#6B7280", linewidth=1, linestyle="--")
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(["Control", "Outpatient", "Acute care"])
    ax.set_xlabel(
        "Change in probability (percentage points)\n"
        "Steps: approximately 7,323 to 10,622/day"
    )
    ax.set_title(
        "B. Model-based decomposition",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.21), ncol=3,
              columnspacing=0.9, handletextpad=0.4)
    ax.set_ylim(-0.55, 2.55)

    fig.tight_layout(pad=0.8)
    fig.savefig(
        BASE_DIR / "Figure2_results.png",
        dpi=300,
        facecolor="white",
    )
    fig.savefig(BASE_DIR / "Figure2_results.pdf", facecolor="white")
    fig.savefig(
        BASE_DIR / "Fig2.tif",
        dpi=300,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def write_compact_tables(summary):
    raw = summary.loc[
        summary["variant"].eq("Primary PCA")
        & summary["scale"].eq("Raw category probability")
    ].copy()
    raw.to_csv(BASE_DIR / "raw_category_natural_effects.csv", index=False)

    sensitivity = summary.loc[
        summary["scale"].eq("Pairwise-normalized")
    ].copy()
    sensitivity.to_csv(
        BASE_DIR / "vulnerability_score_robustness_summary.csv", index=False
    )


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize_analysis(residual_nodes=51):
    global ANALYSIS, REFERENCE_LOADINGS, STEPS_MEAN, STEPS_SD, AGE_MEAN, AGE_SD
    global BOOTSTRAP_RESIDUAL_NODES
    ANALYSIS = load_analysis_data()
    STEPS_MEAN = ANALYSIS["avg_steps"].mean()
    STEPS_SD = ANALYSIS["avg_steps"].std(ddof=1)
    AGE_MEAN = ANALYSIS["age"].mean()
    AGE_SD = ANALYSIS["age"].std(ddof=1)
    BOOTSTRAP_RESIDUAL_NODES = residual_nodes
    REFERENCE_LOADINGS = compute_reference_loadings(ANALYSIS)


def archive_previous_outputs():
    """Preserve recoverable copies before replacing analysis products."""
    filenames = [
        "mediation_bootstrap_summary.csv", "mediation_bootstrap_draws.csv",
        "mediation_bootstrap_draws_wide.csv", "mediation_bootstrap_metadata.csv",
        "robustness_mediation_summary.csv", "robustness_mediation_bootstrap_draws.csv",
        "robustness_analysis_metadata.csv", "raw_category_natural_effects.csv",
        "vulnerability_score_robustness_summary.csv", "Figure2_results.pdf",
        "Figure2_results.png", "Fig2.tif", "canonical_mediation_manifest.json",
        "canonical_residual_integration_sensitivity.csv", "canonical_bootstrap_resamples.csv",
        "figure2_standardized_probability_curves.csv", "cholesterol_indicator_overlap.csv",
        "steps_spline_nonlinearity_test.csv", "case_only_acute_vs_outpatient_results.csv",
        "robustness_vulnerability_weights.csv", "robustness_vulnerability_correlations.csv",
    ]
    backup = Path(tempfile.mkdtemp(prefix="canonical_mediation_outputs_", dir=BASE_DIR))
    for filename in filenames:
        source = BASE_DIR / filename
        if source.exists():
            shutil.copy2(source, backup / filename)
    return backup


def write_primary_exports(summary, draws_frame, metadata):
    """All primary views are exact projections of the one canonical run."""
    primary = summary.loc[
        summary["variant"].eq("Primary PCA")
        & summary["scale"].eq("Pairwise-normalized")
    ].drop(columns=["variant", "scale"]).rename(columns={"target": "contrast"}).copy()
    primary["steps_unit"] = f"per 1 pooled SD = {STEPS_SD:.2f} steps/day"
    primary["exposure_contrast"] = "steps_z 0 to 1 on fixed original pooled scale"
    primary["outcome_model"] = "one 3-category multinomial logistic model"
    primary["effect_scale"] = "pairwise-normalized probability difference"
    primary["bootstrap_level"] = "participant"
    primary["PCA_bootstrap"] = "pooled PCA refitted and sign-aligned in every draw"
    primary.to_csv(BASE_DIR / "mediation_bootstrap_summary.csv", index=False)
    selected = [f"Primary PCA|Pairwise-normalized|{target}|{effect}"
                for target in PAIRWISE_TARGETS for effect in EFFECTS]
    wide = draws_frame[["draw"] + selected].copy()
    wide.columns = ["draw"] + [column.split("|", 2)[2].replace("|", "__") for column in selected]
    wide.to_csv(BASE_DIR / "mediation_bootstrap_draws_wide.csv", index=False)
    long = wide.melt(id_vars="draw", var_name="contrast_effect", value_name="estimate")
    long[["contrast", "effect"]] = long["contrast_effect"].str.split("__", n=1, expand=True)
    long[["draw", "contrast", "effect", "estimate"]].to_csv(
        BASE_DIR / "mediation_bootstrap_draws.csv", index=False
    )
    metadata.to_csv(BASE_DIR / "mediation_bootstrap_metadata.csv", index=False)
    reconstructed = summary.loc[
        summary["variant"].eq("Primary PCA")
        & summary["scale"].eq("Pairwise-normalized")
    ]
    numeric = ["estimate", "CI_lower", "CI_upper", "p_value"]
    if not np.array_equal(primary[numeric].to_numpy(), reconstructed[numeric].to_numpy()):
        raise AssertionError("Primary and robustness projections differ.")


def validate_canonical_code(manifest):
    """Validate exact reviewed revisions and return visual-only hash overrides.

    The original run hash is never replaced to imply that mediation was rerun.
    Every accepted post-run source hash must have been separately reviewed;
    any subsequent unrecorded source edit still fails this guard. A documented
    Figure 2 labelling revision can replace only its three visual artifacts,
    never a scientific CSV or other original analysis artifact.
    """
    current_hash = sha256_file(__file__)
    run_hash = manifest["code_sha256"]
    if current_hash == run_hash:
        return {}
    for revision in manifest.get("post_run_code_revisions", []):
        if (revision.get("source_run_code_sha256") == run_hash
                and revision.get("revised_code_sha256") == current_hash
                and revision.get("scope") == "steps-spline-diagnostic-only"
                and revision.get("mediation_refitted") is False):
            for filename, digest in revision.get("diagnostic_artifacts_sha256", {}).items():
                if sha256_file(BASE_DIR / filename) != digest:
                    raise RuntimeError(f"Corrected diagnostic artifact changed: {filename}.")
            return {}
        if (revision.get("source_run_code_sha256") == run_hash
                and revision.get("revised_code_sha256") == current_hash
                and revision.get("scope") == "figure2-exposure-label-reporting-only"
                and revision.get("mediation_refitted") is False
                and revision.get("fitted_results_unchanged") is True):
            # This reporting edit builds on the reviewed diagnostic correction;
            # its provenance and corrected diagnostic remain independently checked.
            predecessors = [
                item for item in manifest.get("post_run_code_revisions", [])
                if (item.get("revised_code_sha256") == revision.get("previous_reviewed_code_sha256")
                    and item.get("source_run_code_sha256") == run_hash
                    and item.get("scope") == "steps-spline-diagnostic-only"
                    and item.get("mediation_refitted") is False)
            ]
            if len(predecessors) != 1:
                raise RuntimeError("Reporting revision has no unique reviewed diagnostic predecessor.")
            for filename, digest in predecessors[0].get("diagnostic_artifacts_sha256", {}).items():
                if sha256_file(BASE_DIR / filename) != digest:
                    raise RuntimeError(f"Corrected diagnostic artifact changed: {filename}.")
            overrides = revision.get("reporting_artifacts_sha256", {})
            allowed = {"Figure2_results.pdf", "Figure2_results.png", "Fig2.tif"}
            if not isinstance(overrides, dict) or not overrides or not set(overrides).issubset(allowed):
                raise RuntimeError("Reporting revision may override only the three Figure 2 visual artifacts.")
            originals = revision.get("source_reporting_artifacts_sha256", {})
            if set(originals) != set(overrides):
                raise RuntimeError("Reporting revision must retain every replaced original artifact hash.")
            for filename, digest in overrides.items():
                if originals[filename] != manifest["artifacts_sha256"].get(filename):
                    raise RuntimeError(f"Reporting revision original hash mismatch: {filename}.")
                if sha256_file(BASE_DIR / filename) != digest:
                    raise RuntimeError(f"Reporting artifact changed: {filename}.")
            return dict(overrides)
    raise RuntimeError("Canonical implementation changed without a recorded reviewed revision.")


def load_canonical_pooled_results(mediation_data=None):
    """Notebook entry point: verify the canonical artifacts; never silently refit."""
    manifest_path = BASE_DIR / "canonical_mediation_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Run python run_robustness_and_figure2.py --bootstrap 2000 --residual-nodes 51 first.")
    manifest = json.loads(manifest_path.read_text())
    if manifest["canonical_version"] != CANONICAL_VERSION:
        raise RuntimeError("Canonical analysis version mismatch; rerun the pipeline.")
    if manifest["data_file_sha256"] != sha256_file(DATA_FILE):
        raise RuntimeError("The source data changed after the canonical run.")
    reporting_overrides = validate_canonical_code(manifest)
    for filename, digest in manifest["artifacts_sha256"].items():
        if sha256_file(BASE_DIR / filename) != reporting_overrides.get(filename, digest):
            raise RuntimeError(f"Canonical artifact changed: {filename}.")
    initialize_analysis(manifest["residual_quantile_nodes"])
    if mediation_data is not None:
        if (len(mediation_data) != len(ANALYSIS)
                or not mediation_data["person_id"].is_unique
                or set(mediation_data["person_id"]) != set(ANALYSIS["person_id"])):
            raise ValueError("Notebook cohort differs from the canonical mediation cohort.")
        # Cohort identity alone is insufficient if scores/exposure were changed.
        common = [c for c in ["avg_steps", "age", "sex_at_birth", "is_smoker", "is_drinker",
                              "Outcome_Status", "Sleep", "Mental", "Clinical"]
                  if c in mediation_data.columns]
        actual = mediation_data.set_index("person_id").sort_index()
        expected = ANALYSIS.set_index("person_id").sort_index()
        for column in common:
            if not np.array_equal(actual[column].to_numpy(), expected[column].to_numpy()):
                raise ValueError(f"Notebook data differ from canonical input in {column}.")
    return (pd.read_csv(BASE_DIR / "mediation_bootstrap_summary.csv"),
            pd.read_csv(BASE_DIR / "mediation_bootstrap_metadata.csv"))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--residual-nodes", type=int, default=51)
    parser.add_argument("--probability-simulations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def main():
    global ANALYSIS
    global REFERENCE_LOADINGS
    global STEPS_MEAN
    global STEPS_SD
    global AGE_MEAN
    global AGE_SD
    global BOOTSTRAP_RESIDUAL_NODES

    arguments = parse_arguments()
    if arguments.bootstrap < 1:
        raise ValueError("--bootstrap must be positive.")
    if arguments.workers < 1:
        raise ValueError("--workers must be positive.")
    if arguments.residual_nodes < 3:
        raise ValueError("--residual-nodes must be at least 3.")

    initialize_analysis(arguments.residual_nodes)
    backup = archive_previous_outputs()
    print(f"Previous outputs preserved in {backup}", flush=True)

    overlap = cholesterol_overlap(ANALYSIS)
    overlap.to_csv(BASE_DIR / "cholesterol_indicator_overlap.csv", index=False)
    loadings, correlations = vulnerability_diagnostics(ANALYSIS)
    loadings.to_csv(
        BASE_DIR / "robustness_vulnerability_weights.csv", index=False
    )
    correlations.to_csv(
        BASE_DIR / "robustness_vulnerability_correlations.csv", index=False
    )

    spline = fit_spline_test(ANALYSIS)
    spline.to_csv(BASE_DIR / "steps_spline_nonlinearity_test.csv", index=False)

    case_only = fit_case_only_setting_sensitivity(ANALYSIS)
    case_only.to_csv(
        BASE_DIR / "case_only_acute_vs_outpatient_results.csv", index=False
    )

    curves = standardized_probability_curves(
        ANALYSIS,
        simulation_draws=arguments.probability_simulations,
        seed=arguments.seed + 17,
    )
    curves.to_csv(
        BASE_DIR / "figure2_standardized_probability_curves.csv", index=False
    )

    columns = result_columns()
    point_positions = np.arange(len(ANALYSIS))
    point_values = flatten_effects(point_positions, residual_node_count=arguments.residual_nodes)
    point_values_k101 = flatten_effects(point_positions, residual_node_count=101)
    integration_sensitivity = pd.DataFrame({
        "component": columns,
        "canonical_residual_nodes": arguments.residual_nodes,
        "canonical_estimate": point_values,
        "estimate_101_nodes": point_values_k101,
        "difference_percentage_points": 100 * (point_values - point_values_k101),
    })
    integration_sensitivity.to_csv(
        BASE_DIR / "canonical_residual_integration_sensitivity.csv", index=False
    )

    seed_sequence = np.random.SeedSequence(arguments.seed)
    worker_seeds = [
        int(sequence.generate_state(1)[0])
        for sequence in seed_sequence.spawn(arguments.bootstrap)
    ]
    pd.DataFrame({"draw": np.arange(1, arguments.bootstrap + 1),
                  "child_seed": worker_seeds}).to_csv(
        BASE_DIR / "canonical_bootstrap_resamples.csv", index=False
    )
    draws = np.full((arguments.bootstrap, len(columns)), np.nan)
    failures = []

    if arguments.workers == 1:
        iterator = map(bootstrap_worker, worker_seeds)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=arguments.workers)
        iterator = executor.map(bootstrap_worker, worker_seeds, chunksize=1)

    try:
        for draw_index, (successful, message, values) in enumerate(iterator):
            draws[draw_index] = values
            if not successful:
                failures.append((draw_index, message))
            if (
                (draw_index + 1) % 100 == 0
                or draw_index + 1 == arguments.bootstrap
            ):
                print(
                    f"Robustness bootstrap {draw_index + 1}/"
                    f"{arguments.bootstrap}; failures={len(failures)}",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown()

    draws_frame = pd.DataFrame(draws, columns=columns)
    draws_frame.insert(0, "draw", np.arange(1, len(draws_frame) + 1))
    draws_frame.to_csv(
        BASE_DIR / "robustness_mediation_bootstrap_draws.csv", index=False
    )

    summary = summarize_bootstrap(
        point_values,
        draws,
        columns,
        requested=arguments.bootstrap,
        seed=arguments.seed,
    )
    summary.to_csv(
        BASE_DIR / "robustness_mediation_summary.csv", index=False
    )
    write_compact_tables(summary)

    create_figure(ANALYSIS, curves, spline, summary)

    metadata = pd.DataFrame(
        [
            {
                "n": len(ANALYSIS),
                "unique_person_id": ANALYSIS["person_id"].nunique(),
                "control": int((ANALYSIS["Outcome_Status"] == 0).sum()),
                "outpatient": int((ANALYSIS["Outcome_Status"] == 1).sum()),
                "acute": int((ANALYSIS["Outcome_Status"] == 2).sum()),
                "steps_mean": STEPS_MEAN,
                "steps_SD": STEPS_SD,
                "steps_mean_fixed": STEPS_MEAN,
                "steps_SD_fixed": STEPS_SD,
                "age_mean_fixed": AGE_MEAN,
                "age_SD_fixed": AGE_SD,
                "bootstrap_requested": arguments.bootstrap,
                "bootstrap_failed_draws": len(failures),
                "bootstrap_successful": arguments.bootstrap - len(failures),
                "bootstrap_failed": len(failures),
                "workers": arguments.workers,
                "residual_quantile_nodes": arguments.residual_nodes,
                "point_estimate_residual_nodes": arguments.residual_nodes,
                "residual_nodes": arguments.residual_nodes,
                "max_absolute_K_vs_101_sensitivity": float(np.max(np.abs(point_values - point_values_k101))),
                "canonical_version": CANONICAL_VERSION,
                "shared_resamples_across_variants": True,
                "outcome_model": "one 3-category multinomial logistic model",
                "effect_scale": "raw category and individual-level pairwise-normalized probability differences",
                "PCA_bootstrap": "pooled PCA refitted and sign-aligned in every draw",
                "probability_curve_simulations": (
                    arguments.probability_simulations
                ),
                "seed": arguments.seed,
            }
        ]
    )
    metadata.to_csv(
        BASE_DIR / "robustness_analysis_metadata.csv", index=False
    )
    write_primary_exports(summary, draws_frame, metadata)
    if failures:
        pd.DataFrame(failures, columns=["draw", "error"]).to_csv(
            BASE_DIR / "robustness_bootstrap_failures.csv", index=False
        )

    artifact_names = [
        "mediation_bootstrap_summary.csv", "mediation_bootstrap_draws.csv",
        "mediation_bootstrap_draws_wide.csv", "mediation_bootstrap_metadata.csv",
        "robustness_mediation_summary.csv", "robustness_mediation_bootstrap_draws.csv",
        "robustness_analysis_metadata.csv", "raw_category_natural_effects.csv",
        "vulnerability_score_robustness_summary.csv", "canonical_bootstrap_resamples.csv",
        "canonical_residual_integration_sensitivity.csv", "Figure2_results.pdf",
        "Figure2_results.png", "Fig2.tif", "figure2_standardized_probability_curves.csv",
    ]
    manifest = {
        "canonical_version": CANONICAL_VERSION,
        "n": len(ANALYSIS), "outcome_counts": EXPECTED_COUNTS,
        "data_file_sha256": sha256_file(DATA_FILE),
        "code_sha256": sha256_file(__file__),
        "configuration": vars(arguments),
        "residual_quantile_nodes": arguments.residual_nodes,
        "residual_integration": "Equal-weight midpoint empirical residual quantiles (q+0.5)/K; same K for point and bootstrap.",
        "bootstrap": "SeedSequence(seed).spawn(B); child.generate_state(1)[0] seeds NumPy PCG64; n participant positions sampled with replacement; identical positions for every score variant.",
        "standardization": "Original pooled Steps and age means/SDs fixed; each bootstrap refits domain standardization, pooled PCA aligned to original loadings, standardized score, and both regressions.",
        "population_average": "Per individual: residual-integrated category or normalized pairwise probability; then average over the sampled covariate population.",
        "effect_definitions": "ACME=((q01-q00)+(q11-q10))/2; ADE=((q10-q00)+(q11-q01))/2; Total=q11-q00; exposure 0 to 1 pooled SD.",
        "inference": "Percentile95%CI; two-sided bootstrap sign-tail p with add-one correction; no Holm adjustment for primary pooled estimates.",
        "identities_validated_each_fit": ["ACME+ADE=Total on both scales", "Raw category effects sum to zero within each effect type"],
        "bootstrap_failed_draws": len(failures),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "statsmodels": statsmodels.__version__, "matplotlib": matplotlib.__version__},
        "artifacts_sha256": {name: sha256_file(BASE_DIR / name) for name in artifact_names},
        "previous_outputs_backup": str(backup),
    }
    (BASE_DIR / "canonical_mediation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print("\nSpline test")
    print(spline.to_string(index=False))
    print("\nCase-only acute-versus-outpatient sensitivity analysis")
    print(
        case_only.loc[
            case_only["term"].eq("Mean daily Steps, per pooled SD")
        ].to_string(index=False)
    )
    print("\nCholesterol indicator overlap")
    print(overlap.to_string(index=False))
    print("\nVulnerability correlations")
    print(correlations.to_string(index=False))
    print("\nPairwise ACME robustness")
    print(
        summary.loc[
            summary["scale"].eq("Pairwise-normalized")
            & summary["effect"].eq("ACME"),
            [
                "variant",
                "target",
                "estimate_percentage_points",
                "CI_lower_percentage_points",
                "CI_upper_percentage_points",
                "p_value",
            ],
        ].to_string(index=False)
    )
    print("\nRaw category natural effects: primary PCA")
    print(
        summary.loc[
            summary["variant"].eq("Primary PCA")
            & summary["scale"].eq("Raw category probability"),
            [
                "target",
                "effect",
                "estimate_percentage_points",
                "CI_lower_percentage_points",
                "CI_upper_percentage_points",
                "p_value",
            ],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
