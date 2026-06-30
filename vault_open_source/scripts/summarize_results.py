#!/usr/bin/env python3
import argparse
import ast
import json
from pathlib import Path


DATASETS = [
    ("aime24", "AIME24"),
    ("aime25", "AIME25"),
    ("hmmt25", "HMMT25"),
]

METHODS = [
    ("base", "Base Qwen3-1.7B"),
    ("grpo", "GRPO"),
    ("opsd_fixed_t0p9", "OPSD fixed T=0.9"),
    ("opsd_vault_t0p9", "VAULT T=0.9"),
]

METRICS = [
    ("average_at_n_pct", "Average@N"),
    ("pass_at_n_pct", "Pass@N"),
    ("majority_vote_at_n_pct", "Majority@N"),
    ("format_rate", "Format Rate"),
]

VAULT_LOG_METRICS = [
    "vault_kappa",
    "vault_norm_ema",
    "vault_k_mean",
    "vault_survival_mean",
    "vault_w_mean",
    "vault_w_valid_mean",
    "vault_w_censored_mean",
    "vault_valid_rate",
    "vault_censored_rate",
    "vault_effective_sample_size",
    "vault_truncation_rate",
    "vault_boxed_rate",
    "generation_length_mean",
]


def load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_value(value):
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value)


def build_table(eval_dir: Path, metric_key: str):
    rows = []
    for method_key, method_name in METHODS:
        values = []
        numeric_values = []
        for dataset_key, _ in DATASETS:
            data = load_json(eval_dir / dataset_key / f"{method_key}.json")
            value = None if data is None else data.get(metric_key)
            values.append(value)
            if isinstance(value, (int, float)):
                numeric_values.append(float(value))
        average = sum(numeric_values) / len(numeric_values) if numeric_values else None
        rows.append((method_name, values, average))
    return rows


def markdown_table(title: str, rows):
    lines = [f"### {title}", "", "| Method | AIME24 | AIME25 | HMMT25 | Average |", "|---|---:|---:|---:|---:|"]
    for method_name, values, average in rows:
        lines.append(
            "| "
            + " | ".join([method_name] + [format_value(v) for v in values] + [format_value(average)])
            + " |"
        )
    return "\n".join(lines)


def collect_settings(eval_dir: Path):
    for method_key, _ in METHODS:
        for dataset_key, _ in DATASETS:
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


def parse_log_dicts(log_path: Path):
    if not log_path.exists():
        return []
    rows = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        start = line.find("{")
        end = line.rfind("}")
        if start < 0 or end <= start:
            continue
        snippet = line[start : end + 1]
        try:
            data = ast.literal_eval(snippet)
        except Exception:
            continue
        if isinstance(data, dict):
            rows.append(data)
    return rows


def vault_training_diagnostics(log_dir: Path):
    rows = parse_log_dicts(log_dir / "train_opsd_vault_t0p9.log")
    rows = [row for row in rows if any(key in row for key in VAULT_LOG_METRICS)]
    if not rows:
        return ""

    lines = ["### VAULT Training Diagnostics", ""]
    lines.append("| Metric | First | Last | Mean |")
    lines.append("|---|---:|---:|---:|")
    for metric in VAULT_LOG_METRICS:
        values = [float(row[metric]) for row in rows if isinstance(row.get(metric), (int, float))]
        if not values:
            continue
        lines.append(
            f"| {metric} | {format_value(values[0])} | {format_value(values[-1])} | {format_value(sum(values) / len(values))} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize VAULT vs fixed-temperature OPSD results.")
    parser.add_argument("--exp_dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    eval_dir = args.exp_dir / "eval_results"
    output = args.output or (args.exp_dir / "summary.md")

    sections = ["# VAULT Qwen3-1.7B AIME/HMMT Summary", ""]
    settings = collect_settings(eval_dir)
    if settings:
        settings_str = ", ".join(f"{key}={value}" for key, value in settings.items())
        sections += ["Evaluation settings: " + settings_str, ""]

    for metric_key, metric_name in METRICS:
        sections.append(markdown_table(metric_name, build_table(eval_dir, metric_key)))
        sections.append("")

    diagnostics = vault_training_diagnostics(args.exp_dir / "logs")
    if diagnostics:
        sections.append(diagnostics)

    missing = []
    for method_key, _ in METHODS:
        for dataset_key, _ in DATASETS:
            path = eval_dir / dataset_key / f"{method_key}.json"
            if not path.exists():
                missing.append(str(path.relative_to(args.exp_dir)))
    if missing:
        sections += ["Missing result files:", ""]
        sections += [f"- {item}" for item in missing]
        sections.append("")

    text = "\n".join(sections).rstrip() + "\n"
    output.write_text(text, encoding="utf-8")
    print(text)
    print(f"Summary saved to: {output}")


if __name__ == "__main__":
    main()
