import unittest

from eval.evaluate_gsm8k import (
    extract_boxed_answer,
    extract_gsm8k_answer,
    normalize_answer,
    summarize_problem_outputs,
)


class EvaluateGSM8KTest(unittest.TestCase):
    def test_extracts_nested_boxed_answer(self):
        self.assertEqual(extract_boxed_answer(r"final is \boxed{\frac{1}{2}}"), r"\frac{1}{2}")

    def test_normalizes_commas_and_dollars(self):
        self.assertEqual(normalize_answer("$1,234"), "1234")

    def test_extracts_ground_truth_marker(self):
        self.assertEqual(extract_gsm8k_answer("reasoning\n#### 18"), "18")

    def test_summarizes_val_n_outputs_like_math_eval(self):
        result = summarize_problem_outputs(
            problem_id=0,
            problem="p",
            gt_answer="72",
            generations=[
                r"work \boxed{72}",
                r"work \boxed{71}",
                "unfinished 72",
                r"work \boxed{72}",
            ],
            val_n=4,
        )

        self.assertEqual(result["num_correct"], 2)
        self.assertTrue(result["pass_at_n"])
        self.assertTrue(result["majority_vote_correct"])
        self.assertEqual(result["num_formatted"], 3)


if __name__ == "__main__":
    unittest.main()
