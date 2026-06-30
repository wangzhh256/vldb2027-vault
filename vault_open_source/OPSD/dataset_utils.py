import os
import re
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


DEFAULT_TRAIN_DATASET = "siyanzhao/Openthoughts_math_30k_opsd"


def extract_gsm8k_final_answer(solution):
    match = re.search(r"####\s*(.+?)\s*$", str(solution), flags=re.DOTALL)
    if match:
        return match.group(1).strip().replace(",", "")

    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*(?:\.\d+)?)?", str(solution))
    return numbers[-1].replace(",", "") if numbers else str(solution).strip()


def load_train_dataset(dataset_source=None, split="train"):
    source = dataset_source or DEFAULT_TRAIN_DATASET
    if isinstance(source, DatasetDict):
        if split not in source:
            raise ValueError(f"DatasetDict does not contain split {split!r}.")
        return source[split]
    if isinstance(source, Dataset):
        return source

    source_path = Path(str(source)).expanduser()
    if source_path.exists():
        dataset = load_from_disk(str(source_path))
    else:
        dataset = load_dataset(str(source))

    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise ValueError(f"Dataset {source!r} does not contain split {split!r}.")
        return dataset[split]
    return dataset


def to_opsd_columns(dataset):
    columns = set(dataset.column_names)
    if {"problem", "solution"}.issubset(columns):
        keep = {"problem", "solution"}
        return dataset.remove_columns([name for name in dataset.column_names if name not in keep])
    if {"Question", "Answer"}.issubset(columns):
        remove_columns = list(dataset.column_names)
        return dataset.map(
            lambda example: {"problem": example["Question"], "solution": example["Answer"]},
            remove_columns=remove_columns,
        )
    if {"question", "answer"}.issubset(columns):
        remove_columns = list(dataset.column_names)
        return dataset.map(
            lambda example: {"problem": example["question"], "solution": example["answer"]},
            remove_columns=remove_columns,
        )
    raise ValueError(
        "Unsupported dataset columns for OPSD training. Expected problem/solution, "
        "Question/Answer, or question/answer; got "
        f"{dataset.column_names}."
    )


def to_grpo_columns(dataset):
    columns = set(dataset.column_names)
    if {"Question", "Answer"}.issubset(columns):
        keep = {"Question", "Answer"}
        return dataset.remove_columns([name for name in dataset.column_names if name not in keep])
    if {"problem", "solution"}.issubset(columns):
        remove_columns = list(dataset.column_names)
        return dataset.map(
            lambda example: {
                "Question": example["problem"],
                "Answer": extract_gsm8k_final_answer(example["solution"]),
            },
            remove_columns=remove_columns,
        )
    if {"question", "answer"}.issubset(columns):
        remove_columns = list(dataset.column_names)
        return dataset.map(
            lambda example: {
                "Question": example["question"],
                "Answer": extract_gsm8k_final_answer(example["answer"]),
            },
            remove_columns=remove_columns,
        )
    raise ValueError(
        "Unsupported dataset columns for GRPO training. Expected problem/solution, "
        "Question/Answer, or question/answer; got "
        f"{dataset.column_names}."
    )


def dataset_source_label(dataset_source):
    source = dataset_source or DEFAULT_TRAIN_DATASET
    return os.path.basename(str(source).rstrip("/")) or str(source)


def ensure_skip_prepare_dataset(training_args):
    dataset_kwargs = dict(getattr(training_args, "dataset_kwargs", None) or {})
    dataset_kwargs["skip_prepare_dataset"] = True
    training_args.dataset_kwargs = dataset_kwargs
    return training_args
