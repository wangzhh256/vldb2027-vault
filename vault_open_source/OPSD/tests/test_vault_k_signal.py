import unittest

import torch

from opsd_trainer import OPSDTrainer


class VAULTKSignalTest(unittest.TestCase):
    def test_unclipped_k_channel_is_nonnegative_when_clipped_loss_proxy_is_negative(self):
        student_probs = torch.tensor([[[0.99, 0.01]]], dtype=torch.float64)
        teacher_probs = torch.tensor([[[0.50, 0.50]]], dtype=torch.float64)

        loss_proxy = OPSDTrainer.generalized_jsd_token_values(
            student_probs,
            teacher_probs,
            beta=0,
            logits_are_probs=True,
            token_clip=0.05,
        )
        k_values = OPSDTrainer.generalized_jsd_k_token_values(
            student_probs,
            teacher_probs,
            beta=0,
            logits_are_probs=True,
        )

        self.assertLess(loss_proxy.item(), 0.0)
        self.assertGreaterEqual(k_values.item(), 0.0)
        self.assertAlmostEqual(k_values.item(), 1.6144630803608508)


if __name__ == "__main__":
    unittest.main()
