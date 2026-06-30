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
    ("opsd_fixed_t0p9", "OPSD fixed T=0.9"),
    ("vault_full_t0p9", "VAULT full"),
    ("hard_filter_t0p9", "Hard filter"),
    ("fixed_penalty_0p5_t0p9", "Fixed penalty 0.5"),
    ("sweet_spot_t0p9", "Sweet-spot only"),
    ("survival_only_t0p9", "Survival only"),
]

METRICS = [
    ("average_at_n_pct", "Average@N"),
    ("pass_at_n_pct", "Pass@N"),
    ("majority_vote_at_n_pct", "Majority@N"),
    ("format_rate", "Format Rate"),
]

LOG_METRICS = [
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


def load_json(path):
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


def build_metric_table(eval_dir, metric_key, title):
    lines = [f"### {title}", "", "| Method | AIME24 | AIME25 | HMMT25 | Average |"]
    lines.append("|---|---:|---:|---:|---:|")
    for method_key, method_name in METHODS:
        values = []
        numeric = []
        for dataset_key, _ in DATASETS:
            data = load_json(eval_dir / dataset_key / f"{method_key}.json")
            value = None if data is None else data.get(metric_key)
            values.append(value)
            if isinstance(value, (int, float)):
                numeric.append(float(value))
        avg = sum(numeric) / len(numeric) if numeric else None
        lines.append("| " + " | ".join([method_name] + [format_value(v) for v in values] + [format_value(avg)]) + " |")
    return "\n".join(lines)


def collect_settings(eval_dir):
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


def parse_log_dicts(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        start = line.find("{")
        end = line.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            data = ast.literal_eval(line[start : end + 1])
        except Exception:
            continue
        if isinstance(data, dict):
            rows.append(data)
    return rows


def diagnostics_table(log_dir):
    log_map = {
        "hard_filter_t0p9": "train_hard_filter_t0p9.log",
        "fixed_penalty_0p5_t0p9": "train_fixed_penalty_0p5_t0p9.log",
        "sweet_spot_t0p9": "train_sweet_spot_t0p9.log",
        "survival_only_t0p9": "train_survival_only_t0p9.log",
    }
    lines = ["### Training Diagnostics", ""]
    lines.append("| Method | Metric | First | Last | Mean |")
    lines.append("|---|---|---:|---:|---:|")
    for method_key, log_name in log_map.items():
        rows = parse_log_dicts(log_dir / log_name)
        rows = [row for row in rows if any(key in row for key in LOG_METRICS)]
        if not rows:
            continue
        method_name = dict(METHODS)[method_key]
        for metric in LOG_METRICS:
            values = [float(row[metric]) for row in rows if isinstance(row.get(metric), (int, float))]
            if not values:
                continue
            lines.append(
                f"| {method_name} | {metric} | {format_value(values[0])} | "
                f"{format_value(values[-1])} | {format_value(sum(values) / len(values))} |"
            )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    eval_dir = args.exp_dir / "eval_results"
    output = args.output or args.exp_dir / "summary.md"

    sections = ["# VAULT Supplemental Experiments: Qwen3-1.7B AIME/HMMT", ""]
    settings = collect_settings(eval_dir)
    if settings:
        sections.append("Evaluation settings: " + ", ".join(f"{k}={v}" for k, v in settings.items()))
        sections.append("")

    for metric_key, title in METRICS:
        sections.append(build_metric_table(eval_dir, metric_key, title))
        sections.append("")
    sections.append(diagnostics_table(args.exp_dir / "logs"))
    sections.append("")

    missing = []
    for method_key, _ in METHODS:
        for dataset_key, _ in DATASETS:
            path = eval_dir / dataset_key / f"{method_key}.json"
            if not path.exists():
                missing.append(str(path.relative_to(args.exp_dir)))
    if missing:
        sections.append("Missing result files:")
        sections.append("")
        sections.extend(f"- {item}" for item in missing)
        sections.append("")

    text = "\n".join(sections).rstrip() + "\n"
    output.write_text(text, encoding="utf-8")
    print(text)
    print(f"Summary saved to: {output}")


if __name__ == "__main__":
    main()
