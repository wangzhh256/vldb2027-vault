import unittest
from types import SimpleNamespace

from datasets import Dataset, DatasetDict

from dataset_utils import (
    ensure_skip_prepare_dataset,
    extract_gsm8k_final_answer,
    load_train_dataset,
    to_grpo_columns,
    to_opsd_columns,
)


class DatasetUtilsTest(unittest.TestCase):
    def test_extracts_gsm8k_final_answer_after_marker(self):
        self.assertEqual(
            extract_gsm8k_final_answer("work\n#### 1,234"),
            "1234",
        )

    def test_converts_gsm8k_dataset_for_opsd(self):
        dataset = Dataset.from_dict({"problem": ["p"], "solution": ["s\n#### 42"]})

        converted = to_opsd_columns(dataset)

        self.assertEqual(converted.column_names, ["problem", "solution"])
        self.assertEqual(converted[0]["problem"], "p")
        self.assertEqual(converted[0]["solution"], "s\n#### 42")

    def test_converts_question_answer_dataset_for_grpo(self):
        dataset = Dataset.from_dict({"Question": ["q"], "Answer": ["7"], "extra": ["drop"]})

        converted = to_grpo_columns(dataset)

        self.assertEqual(converted.column_names, ["Question", "Answer"])
        self.assertEqual(converted[0]["Question"], "q")
        self.assertEqual(converted[0]["Answer"], "7")

    def test_converts_gsm8k_dataset_for_grpo_answer_only(self):
        dataset = Dataset.from_dict({"problem": ["p"], "solution": ["reason\n#### 72"]})

        converted = to_grpo_columns(dataset)

        self.assertEqual(converted.column_names, ["Question", "Answer"])
        self.assertEqual(converted[0], {"Question": "p", "Answer": "72"})

    def test_loads_train_split_from_local_datasetdict(self):
        dataset = DatasetDict(
            {
                "train": Dataset.from_dict({"problem": ["train"], "solution": ["#### 1"]}),
                "test": Dataset.from_dict({"problem": ["test"], "solution": ["#### 2"]}),
            }
        )

        loaded = load_train_dataset(dataset)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["problem"], "train")

    def test_forces_trl_to_skip_default_text_preprocessing(self):
        args = SimpleNamespace(dataset_kwargs={"keep": "value"})

        ensure_skip_prepare_dataset(args)

        self.assertEqual(args.dataset_kwargs["keep"], "value")
        self.assertIs(args.dataset_kwargs["skip_prepare_dataset"], True)


if __name__ == "__main__":
    unittest.main()
