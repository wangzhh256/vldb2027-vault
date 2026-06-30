import unittest

import torch

from vault import VaultWeighter


class VaultWeighterTest(unittest.TestCase):
    def test_valid_weight_is_low_mid_high_shaped(self):
        weighter = VaultWeighter(init_kappa=1.0, init_norm=1.0)
        k = torch.tensor([0.1, 1.0, 5.0])
        valid = torch.tensor([True, True, True])

        raw = weighter.raw_weights(k, valid)

        self.assertLess(raw[0].item(), raw[1].item())
        self.assertLess(raw[2].item(), raw[1].item())

    def test_censored_weights_stay_small_without_batch_mean_renormalization(self):
        weighter = VaultWeighter(init_kappa=1.0, init_norm=0.3)
        k = torch.tensor([0.2, 1.0, 3.0, 5.0])
        valid = torch.tensor([False, False, False, False])

        weights, raw = weighter.compute_weight(k, valid)

        self.assertTrue(torch.all(weights <= 0.1))
        self.assertAlmostEqual(weighter.norm_ema, 0.3)
        self.assertTrue(torch.all(raw > 0.0))

    def test_kappa_and_normalizer_update_only_from_valid_samples(self):
        weighter = VaultWeighter(
            window_size=4,
            min_valid=2,
            init_kappa=0.3,
            init_norm=0.3,
            norm_alpha=0.5,
        )
        k = torch.tensor([0.1, 0.4, 10.0, 20.0])
        valid = torch.tensor([True, True, False, False])

        weighter.update_kappa(k, valid)
        weights, raw = weighter.compute_weight(k, valid)

        self.assertAlmostEqual(weighter.kappa, 0.25)
        expected_norm = 0.5 * 0.3 + 0.5 * raw[valid].mean().item()
        self.assertAlmostEqual(weighter.norm_ema, expected_norm)
        self.assertGreater(weights[valid].mean().item(), weights[~valid].mean().item())

    def test_kappa_has_positive_floor(self):
        weighter = VaultWeighter(window_size=4, min_valid=2, init_kappa=0.3)

        weighter.update_kappa_from_values([1e-8, 2e-8])

        self.assertAlmostEqual(weighter.kappa, 1e-4)

    def test_bell_survival_gives_unboxed_mid_difficulty_real_weight(self):
        weighter = VaultWeighter(
            init_kappa=1.0,
            init_norm=0.3,
            survival_mode="bell",
            clip_mode="global",
            bell_sigma=0.5,
        )
        k = torch.tensor([0.05, 1.0, 5.0])
        boxed = torch.tensor([False, False, False])

        weights, raw = weighter.compute_weight(k, boxed)

        self.assertLess(raw[0].item(), raw[1].item())
        self.assertLess(raw[2].item(), raw[1].item())
        self.assertGreater(weights[1].item(), 0.1)

    def test_all_kappa_update_mode_uses_unboxed_samples(self):
        weighter = VaultWeighter(
            window_size=4,
            min_valid=2,
            init_kappa=0.3,
            kappa_update_mode="all",
        )
        k = torch.tensor([0.2, 0.4, 2.0, 4.0])
        boxed = torch.tensor([False, False, False, False])

        weighter.update_kappa(k, boxed)

        self.assertAlmostEqual(weighter.kappa, 1.2)

    def test_hard_filter_mode_assigns_zero_raw_weight_to_censored_samples(self):
        weighter = VaultWeighter(
            init_kappa=1.0,
            init_norm=1.0,
            weight_mode="hard_filter",
        )
        k = torch.tensor([0.5, 0.5])
        valid = torch.tensor([True, False])

        raw = weighter.raw_weights(k, valid)

        self.assertAlmostEqual(raw[0].item(), 1.0)
        self.assertAlmostEqual(raw[1].item(), 0.0)

    def test_fixed_penalty_mode_uses_configured_censored_weight(self):
        weighter = VaultWeighter(
            init_kappa=1.0,
            init_norm=1.0,
            weight_mode="fixed_penalty",
            fixed_censored_weight=0.5,
        )
        k = torch.tensor([0.2, 4.0])
        valid = torch.tensor([True, False])

        raw = weighter.raw_weights(k, valid)

        self.assertAlmostEqual(raw[0].item(), 1.0)
        self.assertAlmostEqual(raw[1].item(), 0.5)

    def test_sweet_spot_only_mode_ignores_validity_in_raw_weight(self):
        weighter = VaultWeighter(
            init_kappa=1.0,
            init_norm=1.0,
            weight_mode="sweet_spot",
        )
        k = torch.tensor([1.0, 1.0])
        valid = torch.tensor([True, False])

        raw = weighter.raw_weights(k, valid)
        expected = (1.0 - torch.exp(torch.tensor(-1.0))) / 2.0

        self.assertAlmostEqual(raw[0].item(), expected.item())
        self.assertAlmostEqual(raw[1].item(), expected.item())

    def test_survival_only_mode_returns_survival_probability(self):
        weighter = VaultWeighter(
            init_kappa=1.0,
            init_norm=1.0,
            survival_mode="bell",
            bell_sigma=0.5,
            weight_mode="survival_only",
        )
        k = torch.tensor([1.0, 1.0, 4.0])
        valid = torch.tensor([True, False, False])

        raw = weighter.raw_weights(k, valid)

        self.assertAlmostEqual(raw[0].item(), 1.0)
        self.assertAlmostEqual(raw[1].item(), 1.0)
        self.assertAlmostEqual(raw[2].item(), 0.05)


if __name__ == "__main__":
    unittest.main()
