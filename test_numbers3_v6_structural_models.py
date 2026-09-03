#!/usr/bin/env python3
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import numbers3_v3_models as v3
from numbers3_champion_pool import seed_from_history
from numbers3_v4_models import (
    extended_config,
    final_joint,
    pairwise_interaction_joint,
    power_calibrate,
    regime_pairwise_joint,
    v4_config_id,
)


class V6StructuralModelsTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(20260903)
        self.history = rng.integers(0, 10, size=(800, 3), dtype=int)
        raw = rng.random((3, 10))
        self.base_position_probs = raw / raw.sum(axis=1, keepdims=True)

    def test_legacy_config_is_bitwise_path_compatible(self) -> None:
        cfg = extended_config({
            "uniform_blend": 0.5,
            "autoregressive_mix": 0.10,
            "ar_alpha": 5.0,
            "power_beta": 0.90,
        })
        self.assertNotIn("interaction_mix", cfg)
        self.assertNotIn("regime_mix", cfg)

        expected = power_calibrate(
            v3.final_joint(self.base_position_probs, self.history, cfg),
            cfg["power_beta"],
        )
        actual = final_joint(self.base_position_probs, self.history, cfg)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-15)

    def test_inactive_v6_parameters_canonicalize_to_legacy_config(self) -> None:
        legacy = extended_config({"uniform_blend": 0.5, "power_beta": 1.0})
        inactive = extended_config({
            "uniform_blend": 0.5,
            "power_beta": 1.0,
            "interaction_mix": 0.0,
            "interaction_alpha": 10.0,
            "regime_mix": 0.0,
            "regime_window": 1500,
        })
        self.assertEqual(v4_config_id(legacy), v4_config_id(inactive))
        self.assertNotIn("interaction_mix", inactive)
        self.assertNotIn("regime_window", inactive)

    def test_pairwise_only_ignores_inactive_regime_window_in_config_id(self) -> None:
        a = extended_config({"interaction_mix": 0.20, "regime_mix": 0.0, "regime_window": 100})
        b = extended_config({"interaction_mix": 0.20, "regime_mix": 0.0, "regime_window": 1500})
        self.assertEqual(v4_config_id(a), v4_config_id(b))
        self.assertEqual(a["regime_window"], 300)
        self.assertEqual(b["regime_window"], 300)

    def test_pairwise_joint_is_valid_probability_distribution(self) -> None:
        cfg = extended_config({
            "interaction_mix": 0.20,
            "interaction_alpha": 2.0,
        })
        joint = pairwise_interaction_joint(self.history, cfg)
        self.assertEqual(joint.shape, (1000,))
        self.assertTrue(np.isfinite(joint).all())
        self.assertTrue((joint > 0.0).all())
        self.assertAlmostEqual(float(joint.sum()), 1.0, places=14)

    def test_regime_component_only_uses_configured_recent_window(self) -> None:
        rng = np.random.default_rng(7)
        recent = rng.integers(0, 10, size=(100, 3), dtype=int)
        old_a = np.zeros((400, 3), dtype=int)
        old_b = np.full((400, 3), 9, dtype=int)
        cfg = extended_config({
            "regime_mix": 0.20,
            "regime_window": 100,
            "interaction_alpha": 2.0,
        })
        a = regime_pairwise_joint(np.vstack([old_a, recent]), cfg)
        b = regime_pairwise_joint(np.vstack([old_b, recent]), cfg)
        np.testing.assert_allclose(a, b, rtol=0.0, atol=2e-15)

    def test_v6_final_joint_stays_normalized(self) -> None:
        cfg = extended_config({
            "uniform_blend": 0.5,
            "power_beta": 1.10,
            "interaction_mix": 0.20,
            "interaction_alpha": 2.0,
            "regime_mix": 0.10,
            "regime_window": 300,
        })
        joint = final_joint(self.base_position_probs, self.history, cfg)
        self.assertTrue(np.isfinite(joint).all())
        self.assertTrue((joint > 0.0).all())
        self.assertAlmostEqual(float(joint.sum()), 1.0, places=14)

    def test_promoted_history_does_not_auto_activate_live_pool(self) -> None:
        primary_cfg = extended_config({"uniform_blend": 0.5, "power_beta": 1.0})
        historical_cfg = extended_config({
            "uniform_blend": 0.25,
            "power_beta": 0.9,
            "interaction_mix": 0.20,
        })
        pool = {
            "primary_champion_id": "champ-primary",
            "members": [{
                "champion_id": "champ-primary",
                "generation": 3,
                "config_id": v4_config_id(primary_cfg),
                "config": primary_cfg,
                "dev_exact_nll": 6.91,
                "weight": 1.0,
            }],
        }
        history = pd.DataFrame([{
            "decision": "PROMOTED",
            "challenger_config_json": "{}",
            "champion_after": "champ-historical",
        }])
        out = seed_from_history(pool, history, max_members=5)
        self.assertEqual(len(out["members"]), 1)
        self.assertEqual(out["members"][0]["champion_id"], "champ-primary")
        self.assertNotEqual(v4_config_id(primary_cfg), v4_config_id(historical_cfg))


if __name__ == "__main__":
    unittest.main()
