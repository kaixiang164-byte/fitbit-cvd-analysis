#!/usr/bin/env python3
"""True sex-stratified mediation with one pooled PCA definition.

Female and male mediator and multinomial outcome models are fitted separately.
The PCA is always fitted once in the pooled sample (and once per pooled,
sex-stratified bootstrap draw), then applied on that common scale to both
strata. Sex-specific effects are descriptive secondary estimates; formal
inference about sex heterogeneity remains based on the pooled interaction
models and their direct Female-minus-Male contrasts.
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit


SEXES = ("Female", "Male")
CONTRASTS = (
    "Outpatient vs control",
    "Acute vs control",
    "Acute vs outpatient",
)
EFFECTS = ("ACME", "ADE", "Total")
BURDEN_COLUMNS = ("Sleep", "Mental", "Clinical")
MEDIATOR_TERMS = (
    "Intercept",
    "steps_z",
    "age_z",
    "Current smoker",
    "Moderate alcohol consumption",
    "Heavy alcohol consumption",
)
OUTCOME_TERMS = (
    "Intercept",
    "steps_z",
    "vulnerability",
    "age_z",
    "Current smoker",
    "Moderate alcohol consumption",
    "Heavy alcohol consumption",
)

_WORKER_DATA = None
_WORKER_REFERENCE = None
_WORKER_RESIDUAL_NODES = None
_WORKER_GRID = None


def _z_standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    standard_deviation = values.std(ddof=1)
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        raise ValueError("Cannot standardize a variable with invalid or zero SD.")
    return (values - values.mean()) / standard_deviation


def _pooled_pca_vulnerability(
    raw_burdens: np.ndarray,
    reference_loadings: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit one pooled PCA, align PC1, and return a standardized common score."""
    raw_burdens = np.asarray(raw_burdens, dtype=float)
    feature_means = raw_burdens.mean(axis=0)
    feature_sds = raw_burdens.std(axis=0, ddof=1)
    if np.any(~np.isfinite(feature_sds)) or np.any(feature_sds <= 0):
        raise ValueError("Pooled PCA input has an invalid or zero feature SD.")
    standardized = (raw_burdens - feature_means) / feature_sds
    _, singular_values, right_vectors = np.linalg.svd(
        standardized, full_matrices=False
    )
    loading = right_vectors[0].copy()
    if float(loading @ reference_loadings) < 0:
        loading *= -1
    score = _z_standardize(standardized @ loading)
    variance_ratio = float(
        singular_values[0] ** 2 / np.sum(singular_values ** 2)
    )
    return score, loading, feature_means, feature_sds, variance_ratio


def _take(data: Dict[str, np.ndarray], positions: np.ndarray) -> Dict[str, np.ndarray]:
    return {name: values[positions] for name, values in data.items()}


def _mediator_design(data: Dict[str, np.ndarray]) -> np.ndarray:
    n = len(data["steps_z"])
    return np.column_stack(
        [
            np.ones(n),
            data["steps_z"],
            data["age_z"],
            data["smoker"],
            data["drink_moderate"],
            data["drink_heavy"],
        ]
    )


def _outcome_design(
    data: Dict[str, np.ndarray],
    exposure,
    mediator,
) -> np.ndarray:
    n = len(data["steps_z"])
    exposure_array = (
        np.full(n, float(exposure))
        if np.ndim(exposure) == 0
        else np.asarray(exposure, dtype=float)
    )
    mediator_array = (
        np.full(n, float(mediator))
        if np.ndim(mediator) == 0
        else np.asarray(mediator, dtype=float)
    )
    if len(exposure_array) != n or len(mediator_array) != n:
        raise ValueError("Pseudo-population vector has the wrong length.")
    return np.column_stack(
        [
            np.ones(n),
            exposure_array,
            mediator_array,
            data["age_z"],
            data["smoker"],
            data["drink_moderate"],
            data["drink_heavy"],
        ]
    )


def _fit_mnlogit(y: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Fit one three-category multinomial model with optimizer fallback."""
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise RuntimeError("Rank-deficient sex-stratified outcome model.")
    fitted = None
    last_error = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method, maxiter in (("newton", 100), ("bfgs", 300), ("lbfgs", 300)):
            try:
                candidate = sm.MNLogit(
                    y, design, check_rank=False
                ).fit(method=method, maxiter=maxiter, disp=False)
            except Exception as error:
                last_error = error
                continue
            converged = bool(candidate.mle_retvals.get("converged", False))
            parameters = np.asarray(candidate.params, dtype=float)
            if converged and np.all(np.isfinite(parameters)):
                fitted = candidate
                break
    if fitted is None:
        detail = "" if last_error is None else ": {}".format(last_error)
        raise RuntimeError("Sex-stratified MNLogit did not converge" + detail)
    parameters = np.asarray(fitted.params, dtype=float)
    if parameters.shape != (design.shape[1], 2):
        raise RuntimeError("Unexpected sex-stratified MNLogit parameter layout.")
    return parameters


def _pairwise_coefficients(parameters: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            parameters[:, 0],
            parameters[:, 1],
            parameters[:, 1] - parameters[:, 0],
        ]
    )


def _residual_nodes(residuals: np.ndarray, node_count: int) -> np.ndarray:
    centered = np.asarray(residuals, dtype=float)
    centered = centered - centered.mean()
    probabilities = (np.arange(node_count) + 0.5) / node_count
    return np.quantile(centered, probabilities)


def _integrated_pair_means(
    linear_predictors: np.ndarray,
    mediator_slopes: np.ndarray,
    residual_nodes: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    answers = np.empty(linear_predictors.shape[1])
    for contrast_index in range(linear_predictors.shape[1]):
        convolution = expit(
            grid[:, None]
            + mediator_slopes[contrast_index] * residual_nodes[None, :]
        ).mean(axis=1)
        answers[contrast_index] = np.mean(
            np.interp(
                linear_predictors[:, contrast_index],
                grid,
                convolution,
                left=convolution[0],
                right=convolution[-1],
            )
        )
    return answers


def _fit_stratum(
    data: Dict[str, np.ndarray],
    vulnerability: np.ndarray,
    residual_node_count: int,
    grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit separate models and return contrast by ACME/ADE/Total effects."""
    mediator_x = _mediator_design(data)
    mediator_parameters, _, rank, _ = np.linalg.lstsq(
        mediator_x, vulnerability, rcond=None
    )
    if rank != mediator_x.shape[1]:
        raise RuntimeError("Rank-deficient sex-stratified mediator model.")
    residuals = vulnerability - mediator_x @ mediator_parameters

    outcome_x = _outcome_design(data, data["steps_z"], vulnerability)
    outcome_parameters = _fit_mnlogit(data["outcome"], outcome_x)
    pair_parameters = _pairwise_coefficients(outcome_parameters)
    nodes = _residual_nodes(residuals, residual_node_count)

    mediator_means = {}
    for exposure in (0.0, 1.0):
        x = mediator_x.copy()
        x[:, 1] = exposure
        mediator_means[exposure] = x @ mediator_parameters

    theta = {}
    mediator_slopes = pair_parameters[2]
    for outcome_exposure in (0.0, 1.0):
        for mediator_exposure in (0.0, 1.0):
            x = _outcome_design(
                data,
                outcome_exposure,
                mediator_means[mediator_exposure],
            )
            eta = x @ pair_parameters
            theta[(outcome_exposure, mediator_exposure)] = (
                _integrated_pair_means(
                    eta, mediator_slopes, nodes, grid
                )
            )

    q00 = theta[(0.0, 0.0)]
    q01 = theta[(0.0, 1.0)]
    q10 = theta[(1.0, 0.0)]
    q11 = theta[(1.0, 1.0)]
    acme = ((q01 - q00) + (q11 - q10)) / 2
    ade = ((q10 - q00) + (q11 - q01)) / 2
    total = q11 - q00
    effects = np.column_stack([acme, ade, total])
    if not np.allclose(effects[:, 0] + effects[:, 1], effects[:, 2], atol=1e-10):
        raise AssertionError("ACME + ADE must equal Total.")
    return effects, mediator_parameters, outcome_parameters, residuals


def _initialize_worker(
    data_by_sex: Dict[str, Dict[str, np.ndarray]],
    reference_loadings: np.ndarray,
    residual_node_count: int,
    grid: np.ndarray,
) -> None:
    global _WORKER_DATA
    global _WORKER_REFERENCE
    global _WORKER_RESIDUAL_NODES
    global _WORKER_GRID
    _WORKER_DATA = data_by_sex
    _WORKER_REFERENCE = reference_loadings
    _WORKER_RESIDUAL_NODES = residual_node_count
    _WORKER_GRID = grid


def _bootstrap_one(task: Tuple[int, int]):
    replicate, replicate_seed = task
    try:
        rng = np.random.default_rng(replicate_seed)
        sampled = {}
        pooled_raw_parts = []
        for sex in SEXES:
            source = _WORKER_DATA[sex]
            n = len(source["outcome"])
            positions = rng.integers(0, n, size=n)
            sampled[sex] = _take(source, positions)
            pooled_raw_parts.append(sampled[sex]["raw_burdens"])

        pooled_raw = np.vstack(pooled_raw_parts)
        pooled_vulnerability, _, _, _, _ = _pooled_pca_vulnerability(
            pooled_raw, _WORKER_REFERENCE
        )

        output = []
        offset = 0
        for sex in SEXES:
            n = len(sampled[sex]["outcome"])
            vulnerability = pooled_vulnerability[offset : offset + n]
            offset += n
            effects, _, _, _ = _fit_stratum(
                sampled[sex],
                vulnerability,
                _WORKER_RESIDUAL_NODES,
                _WORKER_GRID,
            )
            output.extend(effects.reshape(-1))
        return replicate, np.asarray(output, dtype=float), ""
    except Exception as error:
        return replicate, None, "{}: {}".format(type(error).__name__, error)


def _tail_count_p_value(draws: np.ndarray) -> float:
    draws = np.asarray(draws, dtype=float)
    denominator = len(draws) + 1
    lower = (np.sum(draws <= 0) + 1) / denominator
    upper = (np.sum(draws >= 0) + 1) / denominator
    return float(min(1.0, 2 * min(lower, upper)))


def _summary_rows(
    point_effects: np.ndarray,
    bootstrap_effects: np.ndarray,
    counts: pd.DataFrame,
) -> List[dict]:
    rows = []
    for sex_index, sex in enumerate(SEXES):
        sex_counts = counts.loc[sex]
        for contrast_index, contrast in enumerate(CONTRASTS):
            for effect_index, effect in enumerate(EFFECTS):
                estimate = point_effects[sex_index, contrast_index, effect_index]
                draws = bootstrap_effects[
                    :, sex_index, contrast_index, effect_index
                ]
                lower, upper = np.quantile(draws, [0.025, 0.975])
                rows.append(
                    {
                        "sex": sex,
                        "contrast": contrast,
                        "effect": effect,
                        "estimate": estimate,
                        "CI_lower": lower,
                        "CI_upper": upper,
                        "p_value": _tail_count_p_value(draws),
                        "estimate_percentage_points": 100 * estimate,
                        "CI_lower_percentage_points": 100 * lower,
                        "CI_upper_percentage_points": 100 * upper,
                        "n": int(sex_counts.sum()),
                        "n_control": int(sex_counts.get(0, 0)),
                        "n_outpatient": int(sex_counts.get(1, 0)),
                        "n_acute": int(sex_counts.get(2, 0)),
                        "role": "descriptive secondary sex-stratified estimate",
                    }
                )
    return rows


def _difference_rows(
    point_effects: np.ndarray,
    bootstrap_effects: np.ndarray,
) -> List[dict]:
    point_differences = point_effects[0] - point_effects[1]
    bootstrap_differences = bootstrap_effects[:, 0] - bootstrap_effects[:, 1]
    rows = []
    for contrast_index, contrast in enumerate(CONTRASTS):
        for effect_index, effect in enumerate(EFFECTS):
            estimate = point_differences[contrast_index, effect_index]
            draws = bootstrap_differences[:, contrast_index, effect_index]
            lower, upper = np.quantile(draws, [0.025, 0.975])
            rows.append(
                {
                    "comparison": "Female - Male",
                    "contrast": contrast,
                    "effect": effect,
                    "estimate": estimate,
                    "CI_lower": lower,
                    "CI_upper": upper,
                    "p_value": _tail_count_p_value(draws),
                    "estimate_percentage_points": 100 * estimate,
                    "CI_lower_percentage_points": 100 * lower,
                    "CI_upper_percentage_points": 100 * upper,
                    "role": (
                        "descriptive sensitivity contrast; formal sex inference "
                        "uses pooled interaction models"
                    ),
                }
            )
    return rows


def _coefficient_rows(
    models: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> List[dict]:
    rows = []
    for sex in SEXES:
        mediator_parameters, outcome_parameters = models[sex]
        for term, estimate in zip(MEDIATOR_TERMS, mediator_parameters):
            rows.append(
                {
                    "sex": sex,
                    "model": "Mediator OLS",
                    "contrast": "",
                    "term": term,
                    "estimate": estimate,
                    "OR": np.nan,
                }
            )
        pair_parameters = _pairwise_coefficients(outcome_parameters)
        for contrast_index, contrast in enumerate(CONTRASTS):
            for term, estimate in zip(
                OUTCOME_TERMS, pair_parameters[:, contrast_index]
            ):
                rows.append(
                    {
                        "sex": sex,
                        "model": "Outcome MNLogit",
                        "contrast": contrast,
                        "term": term,
                        "estimate": estimate,
                        "OR": np.exp(estimate),
                    }
                )
    return rows


def _load_data(
    input_path: Path,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    frame = pd.read_csv(input_path)
    required = {
        "Outcome_Status",
        "steps_z",
        "age_z",
        "sex_at_birth",
        "is_smoker",
        "is_drinker",
        "vulnerability",
        *BURDEN_COLUMNS,
    }
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            "Input data are missing columns: {}".format(sorted(missing_columns))
        )
    if frame[list(required)].isna().any().any():
        raise ValueError("Input data contain missing analysis values.")
    observed_sexes = set(frame["sex_at_birth"].unique())
    if observed_sexes != set(SEXES):
        raise ValueError("Expected exactly Female and Male sex categories.")
    if set(frame["Outcome_Status"].unique()) != {0, 1, 2}:
        raise ValueError("Outcome_Status must contain exactly codes 0, 1, and 2.")
    if not set(frame["is_smoker"].unique()).issubset({0, 1}):
        raise ValueError("is_smoker must use only codes 0 and 1.")
    if not set(frame["is_drinker"].unique()).issubset({0, 1, 2}):
        raise ValueError("is_drinker must use only codes 0, 1, and 2.")

    data_by_sex = {}
    for sex in SEXES:
        subset = frame.loc[frame["sex_at_birth"].eq(sex)].copy()
        if set(subset["Outcome_Status"].unique()) != {0, 1, 2}:
            raise ValueError(
                "Each sex stratum must contain all three outcome categories."
            )
        data_by_sex[sex] = {
            "outcome": subset["Outcome_Status"].to_numpy(int),
            "steps_z": subset["steps_z"].to_numpy(float),
            "age_z": subset["age_z"].to_numpy(float),
            "smoker": subset["is_smoker"].eq(1).to_numpy(float),
            "drink_moderate": subset["is_drinker"].eq(1).to_numpy(float),
            "drink_heavy": subset["is_drinker"].eq(2).to_numpy(float),
            "raw_burdens": subset[list(BURDEN_COLUMNS)].to_numpy(float),
            "saved_vulnerability": subset["vulnerability"].to_numpy(float),
        }
    return frame, data_by_sex


def run_analysis(args: argparse.Namespace) -> None:
    start_time = time.time()
    output_directory = args.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    frame, data_by_sex = _load_data(args.input.resolve())
    loading_frame = pd.read_csv(args.reference_loadings.resolve())
    reference_loadings = loading_frame.set_index("raw_feature").loc[
        list(BURDEN_COLUMNS), "PC1_loading"
    ].to_numpy(float)
    reference_loadings = reference_loadings / np.linalg.norm(reference_loadings)
    scaling = pd.read_csv(args.scaling.resolve()).set_index("variable")

    pooled_raw = np.vstack(
        [data_by_sex[sex]["raw_burdens"] for sex in SEXES]
    )
    pooled_saved = np.concatenate(
        [data_by_sex[sex]["saved_vulnerability"] for sex in SEXES]
    )
    (
        pooled_vulnerability,
        point_loading,
        point_feature_means,
        point_feature_sds,
        point_variance_ratio,
    ) = _pooled_pca_vulnerability(pooled_raw, reference_loadings)
    pca_correlation = float(np.corrcoef(pooled_vulnerability, pooled_saved)[0, 1])
    pca_maximum_error = float(
        np.max(np.abs(pooled_vulnerability - pooled_saved))
    )

    grid = np.linspace(-30.0, 30.0, args.grid_size)
    point_effects = np.empty((len(SEXES), len(CONTRASTS), len(EFFECTS)))
    sensitivity_effects = np.empty_like(point_effects)
    models = {}
    offset = 0
    for sex_index, sex in enumerate(SEXES):
        n = len(data_by_sex[sex]["outcome"])
        vulnerability = pooled_vulnerability[offset : offset + n]
        offset += n
        effects, mediator_parameters, outcome_parameters, _ = _fit_stratum(
            data_by_sex[sex],
            vulnerability,
            args.residual_nodes,
            grid,
        )
        point_effects[sex_index] = effects
        sensitivity, _, _, _ = _fit_stratum(
            data_by_sex[sex],
            vulnerability,
            args.sensitivity_residual_nodes,
            grid,
        )
        sensitivity_effects[sex_index] = sensitivity
        models[sex] = (mediator_parameters, outcome_parameters)

    if not np.all(np.isfinite(point_effects)):
        raise RuntimeError("Non-finite point estimate.")
    sensitivity_maximum_error = float(
        np.max(np.abs(point_effects - sensitivity_effects))
    )
    if sensitivity_maximum_error > args.sensitivity_tolerance:
        raise RuntimeError(
            "Residual-node sensitivity exceeded tolerance: {:.3e}".format(
                sensitivity_maximum_error
            )
        )
    print("Point estimates fitted for both independent sex strata.", flush=True)
    print("Starting {} stratified bootstrap draws with {} workers.".format(
        args.bootstrap, args.workers
    ), flush=True)

    seed_sequence = np.random.SeedSequence(args.seed)
    child_sequences = seed_sequence.spawn(args.bootstrap)
    tasks = [
        (
            replicate,
            int(sequence.generate_state(1, dtype=np.uint64)[0]),
        )
        for replicate, sequence in enumerate(child_sequences)
    ]
    raw_draws = np.full(
        (
            args.bootstrap,
            len(SEXES) * len(CONTRASTS) * len(EFFECTS),
        ),
        np.nan,
    )
    failures = []
    context = get_context("fork")
    with context.Pool(
        processes=args.workers,
        initializer=_initialize_worker,
        initargs=(
            data_by_sex,
            reference_loadings,
            args.residual_nodes,
            grid,
        ),
    ) as pool:
        completed = 0
        for replicate, values, error in pool.imap_unordered(
            _bootstrap_one, tasks, chunksize=args.chunksize
        ):
            completed += 1
            if values is None:
                failures.append((replicate, error))
            else:
                raw_draws[replicate] = values
            if (
                completed == args.bootstrap
                or completed % args.progress_every == 0
            ):
                print(
                    "Bootstrap progress: {}/{}; failures={}".format(
                        completed, args.bootstrap, len(failures)
                    ),
                    flush=True,
                )

    successful_mask = np.all(np.isfinite(raw_draws), axis=1)
    successful_draws = raw_draws[successful_mask]
    if successful_draws.shape[0] < args.bootstrap:
        raise RuntimeError(
            "Only {} of {} bootstrap draws succeeded. First failures: {}".format(
                successful_draws.shape[0], args.bootstrap, failures[:5]
            )
        )
    bootstrap_effects = successful_draws.reshape(
        args.bootstrap, len(SEXES), len(CONTRASTS), len(EFFECTS)
    )

    counts = frame.groupby(
        ["sex_at_birth", "Outcome_Status"]
    ).size().unstack(fill_value=0).reindex(index=SEXES, columns=[0, 1, 2])
    summary = pd.DataFrame(
        _summary_rows(point_effects, bootstrap_effects, counts)
    )
    differences = pd.DataFrame(
        _difference_rows(point_effects, bootstrap_effects)
    )
    coefficients = pd.DataFrame(_coefficient_rows(models))

    draw_columns = [
        "{}__{}__{}".format(sex, contrast, effect)
        for sex in SEXES
        for contrast in CONTRASTS
        for effect in EFFECTS
    ]
    draws_frame = pd.DataFrame(successful_draws, columns=draw_columns)
    draws_frame.insert(0, "replicate", np.arange(args.bootstrap))

    pca_check = pd.DataFrame(
        {
            "raw_feature": BURDEN_COLUMNS,
            "reference_loading": reference_loadings,
            "refitted_point_loading": point_loading,
            "point_feature_mean": point_feature_means,
            "point_feature_SD": point_feature_sds,
        }
    )
    sensitivity_rows = []
    for sex_index, sex in enumerate(SEXES):
        for contrast_index, contrast in enumerate(CONTRASTS):
            for effect_index, effect in enumerate(EFFECTS):
                primary = point_effects[sex_index, contrast_index, effect_index]
                sensitivity = sensitivity_effects[
                    sex_index, contrast_index, effect_index
                ]
                sensitivity_rows.append(
                    {
                        "sex": sex,
                        "contrast": contrast,
                        "effect": effect,
                        "primary_residual_nodes": args.residual_nodes,
                        "sensitivity_residual_nodes": (
                            args.sensitivity_residual_nodes
                        ),
                        "primary_estimate": primary,
                        "sensitivity_estimate": sensitivity,
                        "absolute_difference": abs(primary - sensitivity),
                        "tolerance": args.sensitivity_tolerance,
                        "within_tolerance": (
                            abs(primary - sensitivity)
                            <= args.sensitivity_tolerance
                        ),
                    }
                )
    sensitivity_frame = pd.DataFrame(sensitivity_rows)
    maximum_identity_error = float(
        np.max(
            np.abs(
                bootstrap_effects[:, :, :, 0]
                + bootstrap_effects[:, :, :, 1]
                - bootstrap_effects[:, :, :, 2]
            )
        )
    )
    metadata = pd.DataFrame(
        [
            {
                "analysis": "Independent sex-stratified mediation",
                "role": "descriptive secondary/supplementary analysis",
                "n_total": len(frame),
                "n_female": int(counts.loc["Female"].sum()),
                "n_male": int(counts.loc["Male"].sum()),
                "bootstrap_requested": args.bootstrap,
                "bootstrap_successful": successful_draws.shape[0],
                "bootstrap_failed": len(failures),
                "seed": args.seed,
                "workers": args.workers,
                "residual_nodes": args.residual_nodes,
                "sensitivity_residual_nodes": args.sensitivity_residual_nodes,
                "maximum_residual_node_sensitivity_error": (
                    sensitivity_maximum_error
                ),
                "residual_node_sensitivity_tolerance": (
                    args.sensitivity_tolerance
                ),
                "residual_node_sensitivity_passed": (
                    sensitivity_maximum_error <= args.sensitivity_tolerance
                ),
                "integration_grid_size": args.grid_size,
                "exposure_contrast": (
                    "steps_z 0 to 1 using fixed original pooled scale"
                ),
                "steps_SD": float(scaling.loc["avg_steps", "SD"]),
                "age_SD": float(scaling.loc["age", "SD"]),
                "PCA_definition": (
                    "one pooled PCA at point estimate and one pooled PCA "
                    "per sex-stratified bootstrap draw; common score for both sexes"
                ),
                "PCA_point_variance_ratio": point_variance_ratio,
                "PCA_saved_score_correlation": pca_correlation,
                "PCA_saved_score_max_absolute_error": pca_maximum_error,
                "standardization_population": (
                    "within-sex empirical covariate distribution"
                ),
                "model_fitting": (
                    "mediator OLS and outcome MNLogit fitted independently by sex"
                ),
                "effect_scale": "pairwise-normalized probability difference",
                "maximum_ACME_plus_ADE_minus_Total_error": maximum_identity_error,
                "formal_sex_inference": (
                    "pooled interaction pathway tests and pooled-model "
                    "Female-minus-Male Delta-ACME"
                ),
                "elapsed_seconds": time.time() - start_time,
                "python_version": "{}.{}.{}".format(*os.sys.version_info[:3]),
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
                "statsmodels_version": sm.__version__,
            }
        ]
    )

    summary.to_csv(
        output_directory / "sex_stratified_mediation_summary.csv", index=False
    )
    differences.to_csv(
        output_directory / "sex_stratified_mediation_difference_summary.csv",
        index=False,
    )
    draws_frame.to_csv(
        output_directory / "sex_stratified_mediation_bootstrap_draws.csv",
        index=False,
    )
    coefficients.to_csv(
        output_directory / "sex_stratified_model_coefficients.csv", index=False
    )
    pca_check.to_csv(
        output_directory / "sex_stratified_pca_check.csv", index=False
    )
    sensitivity_frame.to_csv(
        output_directory / "sex_stratified_residual_node_sensitivity.csv",
        index=False,
    )
    metadata.to_csv(
        output_directory / "sex_stratified_mediation_metadata.csv", index=False
    )
    if failures:
        pd.DataFrame(failures, columns=["replicate", "error"]).to_csv(
            output_directory / "sex_stratified_mediation_failures.csv",
            index=False,
        )

    display_columns = [
        "sex",
        "contrast",
        "effect",
        "estimate_percentage_points",
        "CI_lower_percentage_points",
        "CI_upper_percentage_points",
        "p_value",
    ]
    print(summary[display_columns].to_string(index=False), flush=True)
    print(
        "PCA reproduction: correlation={:.12f}, max_abs_error={:.3e}".format(
            pca_correlation, pca_maximum_error
        ),
        flush=True,
    )
    print(
        "Maximum |ACME + ADE - Total| across bootstrap draws: {:.3e}".format(
            maximum_identity_error
        ),
        flush=True,
    )
    print(
        "Maximum residual-node sensitivity difference: {:.3e}".format(
            sensitivity_maximum_error
        ),
        flush=True,
    )
    print(
        "Completed in {:.1f} seconds.".format(time.time() - start_time),
        flush=True,
    )


def _parse_arguments() -> argparse.Namespace:
    base_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Fit separate female and male mediation models while retaining "
            "one pooled PCA definition."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=base_directory / "pca_mediation_eligible_dataset.csv",
    )
    parser.add_argument(
        "--reference-loadings",
        type=Path,
        default=base_directory / "pca_vulnerability_loadings.csv",
    )
    parser.add_argument(
        "--scaling",
        type=Path,
        default=base_directory / "m1_standardization_parameters.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=base_directory)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--residual-nodes", type=int, default=51)
    parser.add_argument("--sensitivity-residual-nodes", type=int, default=101)
    parser.add_argument("--sensitivity-tolerance", type=float, default=1e-4)
    parser.add_argument("--grid-size", type=int, default=1201)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
    )
    parser.add_argument("--chunksize", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=100)
    arguments = parser.parse_args()
    if arguments.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    if arguments.workers < 1:
        parser.error("--workers must be positive")
    if arguments.residual_nodes < 3:
        parser.error("--residual-nodes must be at least 3")
    if arguments.sensitivity_residual_nodes < 3:
        parser.error("--sensitivity-residual-nodes must be at least 3")
    if arguments.sensitivity_tolerance <= 0:
        parser.error("--sensitivity-tolerance must be positive")
    if arguments.chunksize < 1:
        parser.error("--chunksize must be positive")
    if arguments.progress_every < 1:
        parser.error("--progress-every must be positive")
    if arguments.grid_size < 101:
        parser.error("--grid-size must be at least 101")
    return arguments


if __name__ == "__main__":
    run_analysis(_parse_arguments())

