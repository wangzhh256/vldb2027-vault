#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


METRICS = [
    ("average_at_n_pct", "Average@N"),
    ("pass_at_n_pct", "Pass@N"),
    ("majority_vote_at_n_pct", "Majority@N"),
    ("format_rate", "Format Rate"),
]


def parse_pairs(items):
    pairs = []
    for item in items:
        if ":" in item:
            key, label = item.split(":", 1)
        else:
            key, label = item, item
        pairs.append((key, label))
    return pairs


def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def format_value(value):
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value)


def build_table(eval_dir, datasets, methods, metric_key, metric_label):
    lines = [
        f"### {metric_label}",
        "",
        "| Method | " + " | ".join(label for _, label in datasets) + " | Average |",
        "|---|" + "---:|" * (len(datasets) + 1),
    ]
    for method_key, method_label in methods:
        values = []
        numeric_values = []
        for dataset_key, _ in datasets:
            data = load_json(eval_dir / dataset_key / f"{method_key}.json")
            value = None if data is None else data.get(metric_key)
            values.append(value)
            if isinstance(value, (int, float)):
                numeric_values.append(float(value))
        average = sum(numeric_values) / len(numeric_values) if numeric_values else None
        lines.append(
            "| "
            + " | ".join([method_label] + [format_value(value) for value in values] + [format_value(average)])
            + " |"
        )
    return "\n".join(lines)


def collect_settings(eval_dir, datasets, methods):
    for method_key, _ in methods:
        for dataset_key, _ in datasets:
            data = load_json(eval_dir / dataset_key / f"{method_key}.json")
            if data:
                return {
                    "val_n": data.get("val_n"),
                    "enable_thinking": data.get("enable_thinking"),
                    "temperature": data.get("temperature"),
                    "top_p": data.get("top_p"),
                    "top_k": data.get("top_k"),
                    "max_new_tokens": data.get("max_new_tokens"),
                    "base_model": data.get("base_model"),
                }
    return {}


def main():
    parser = argparse.ArgumentParser(description="Summarize eval JSONs with val-n metrics.")
    parser.add_argument("--eval_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Evaluation Summary")
    parser.add_argument("--datasets", nargs="+", required=True, help="Pairs like aime24:AIME24 or gsm8k:GSM8K")
    parser.add_argument("--methods", nargs="+", required=True, help="Pairs like base:Base or grpo:GRPO")
    args = parser.parse_args()

    datasets = parse_pairs(args.datasets)
    methods = parse_pairs(args.methods)

    sections = [f"# {args.title}", ""]
    settings = collect_settings(args.eval_dir, datasets, methods)
    if settings:
        sections += [", ".join(f"{key}={value}" for key, value in settings.items()), ""]
    for metric_key, metric_label in METRICS:
        sections.append(build_table(args.eval_dir, datasets, methods, metric_key, metric_label))
        sections.append("")

    missing = []
    for method_key, _ in methods:
        for dataset_key, _ in datasets:
            path = args.eval_dir / dataset_key / f"{method_key}.json"
            if not path.exists():
                missing.append(str(path))
    if missing:
        sections += ["Missing result files:", ""]
        sections += [f"- {item}" for item in missing]
        sections.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(sections).rstrip() + "\n"
    args.output.write_text(text, encoding="utf-8")
    print(text)
    print(f"Summary saved to: {args.output}")


if __name__ == "__main__":
    main()
