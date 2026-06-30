# VAULT

This repository contains the implementation used for the VAULT experiments on
OPSD-style self-distillation for mathematical reasoning.

## Contents

- `OPSD/vault.py`: trajectory valuation and weighting rule.
- `OPSD/opsd_trainer.py`: OPSD trainer integration with VAULT weighting.
- `OPSD/opsd_train.py`: training entry point for OPSD and VAULT.
- `OPSD/grpo_train.py`: GRPO baseline training entry point.
- `OPSD/data_collator.py`, `OPSD/dataset_utils.py`: data processing utilities.
- `OPSD/eval/`: evaluation scripts for AIME/HMMT and GSM8K.
- `OPSD/tests/`: unit tests for the valuation code and helpers.
- `scripts/run_vault_opsd.sh`: main OPSD-vs-VAULT experiment launcher.
- `scripts/run_supplemental.sh`: ablation and profiling launcher.
- `scripts/run_base_grpo_then_gsm8k.sh`: baseline and GSM8K launcher.

## Setup

Create the environment from `OPSD/environment.yml`, then place the base model
and datasets at the paths configured in the launch scripts. The default model
path is:

```text
models/Qwen3-1.7B
```

You can override paths and experiment settings through environment variables in
the script headers.

## Running

For the main AIME/HMMT comparison:

```bash
bash scripts/run_vault_opsd.sh
```

For ablations and profiling:

```bash
bash scripts/run_supplemental.sh
```

For baseline and GSM8K experiments:

```bash
bash scripts/run_base_grpo_then_gsm8k.sh
```

## Evaluation

The evaluation scripts are in `OPSD/eval/`. The reported setup samples four
responses per problem with thinking disabled, temperature `1.0`, top-p `0.8`,
top-k disabled, and a maximum generation length of 8192 tokens for AIME/HMMT.
