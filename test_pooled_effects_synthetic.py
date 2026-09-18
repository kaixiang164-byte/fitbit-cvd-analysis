"""Small generated-data checks of the pooled decomposition; no study inputs."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_robustness_and_figure2 as core


class PooledEffectsSyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(72106)
        n = 600
        steps = rng.normal(size=n)
        age = rng.normal(size=n)
        burden = -0.2 * steps + 0.2 * age + rng.normal(size=n)
        sleep = (burden + rng.normal(size=n) > 0).astype(int)
        mental = ((burden[:, None] + rng.normal(size=(n, 2))) > 0.4).sum(axis=1)
        flags = (burden[:, None] + rng.normal(size=(n, 5)) > 0.7).astype(int)
        eta = np.column_stack([np.zeros(n), -0.4 - 0.2 * steps + 0.2 * burden,
                               -1.1 - 0.4 * steps + 0.3 * burden])
        probabilities = np.exp(eta - eta.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        outcome = (rng.random(n)[:, None] > probabilities.cumsum(axis=1)).sum(axis=1)
        cls.data = pd.DataFrame({
            "avg_steps": 7300 + 2000 * steps, "age": 56 + 10 * age,
            "sex_at_birth": rng.choice(["Female", "Male"], n),
            "is_smoker": rng.integers(0, 2, n), "is_drinker": rng.integers(0, 3, n),
            "Sleep": sleep, "Mental": mental, "Clinical": flags.sum(axis=1),
            "Clinical_deduplicated": flags[:, :3].sum(axis=1) + flags[:, 3:].max(axis=1),
            "Outcome_Status": outcome,
        })

    def state(self):
        return patch.multiple(core, ANALYSIS=self.data,
                              REFERENCE_LOADINGS=core.compute_reference_loadings(self.data),
                              STEPS_MEAN=self.data.avg_steps.mean(),
                              STEPS_SD=self.data.avg_steps.std(ddof=1),
                              AGE_MEAN=self.data.age.mean(), AGE_SD=self.data.age.std(ddof=1))

    def test_decomposition_and_shared_denominator_identities(self):
        with self.state():
            for variant in core.VARIANTS:
                pairwise, raw = core.effects_for_variant(np.arange(len(self.data)), variant, 15)
                self.assertEqual(raw.shape, (3, 3))
                self.assertTrue(np.isfinite(pairwise).all())
                np.testing.assert_allclose(raw[:, 0] + raw[:, 1], raw[:, 2], atol=1e-10)
                np.testing.assert_allclose(pairwise[:, 0] + pairwise[:, 1], pairwise[:, 2], atol=1e-10)
                np.testing.assert_allclose(raw.sum(axis=0), 0, atol=1e-10)

    def test_scores_have_pooled_unit_scale_and_shared_orientation(self):
        with self.state():
            for variant in core.VARIANTS:
                score = core.construct_vulnerability(self.data, variant)
                self.assertAlmostEqual(score.mean(), 0, places=12)
                self.assertAlmostEqual(score.std(ddof=1), 1, places=12)
                self.assertGreater(np.corrcoef(score, self.data.Clinical)[0, 1], 0)

    def test_repeated_synthetic_resample_is_deterministic(self):
        positions = np.random.default_rng(918).integers(0, len(self.data), len(self.data))
        with self.state():
            first = core.effects_for_variant(positions, "Primary PCA", 15)
            second = core.effects_for_variant(positions, "Primary PCA", 15)
            for a, b in zip(first, second):
                np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
