# Manifest

## Core Method

- `OPSD/vault.py`
- `OPSD/opsd_trainer.py`
- `OPSD/opsd_train.py`

## Baselines And Utilities

- `OPSD/grpo_train.py`
- `OPSD/data_collator.py`
- `OPSD/dataset_utils.py`
- `OPSD/accelerate.yaml`
- `OPSD/environment.yml`

## Evaluation

- `OPSD/eval/evaluate_math.py`
- `OPSD/eval/evaluate_gsm8k.py`
- `OPSD/eval/summarize_eval_results.py`

## Tests

- `OPSD/tests/test_vault.py`
- `OPSD/tests/test_vault_k_signal.py`
- `OPSD/tests/test_dataset_utils.py`
- `OPSD/tests/test_evaluate_gsm8k.py`

## Launch Scripts

- `scripts/run_vault_opsd.sh`
- `scripts/run_supplemental.sh`
- `scripts/run_base_grpo_then_gsm8k.sh`
- `scripts/summarize_results.py`
- `scripts/summarize_supplement.py`
