#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_DIR="${EXP_DIR:-$ROOT_DIR/experiments/supplemental}"
OPSD_DIR="${OPSD_DIR:-$ROOT_DIR/OPSD}"
MAIN_EXP_DIR="${MAIN_EXP_DIR:-$ROOT_DIR/experiments/aime_hmmt}"
LOG_DIR="$EXP_DIR/logs"
OUTPUT_DIR="$EXP_DIR/outputs"
EVAL_DIR="$EXP_DIR/eval_results"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$EVAL_DIR"

PYTHON="${PYTHON:-python}"
ACCELERATE="${ACCELERATE:-accelerate}"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/models/Qwen3-1.7B}"

GPUS="${GPUS:-2,3,6,7}"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
NUM_GPUS="${#GPU_LIST[@]}"
if [ "$NUM_GPUS" -ne 4 ]; then
  echo "This script expects exactly four GPUs, got GPUS=$GPUS" >&2
  exit 1
fi

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export HF_HOME="${HF_HOME_OVERRIDE:-$ROOT_DIR/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export MAX_JOBS="${MAX_JOBS:-4}"
export NVCC_THREADS="${NVCC_THREADS:-1}"

WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
GPU_WAIT_MAX_USED_MB="${GPU_WAIT_MAX_USED_MB:-1000}"
GPU_WAIT_POLL_SECONDS="${GPU_WAIT_POLL_SECONDS:-120}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_RQ2_RQ3="${RUN_RQ2_RQ3:-1}"
RUN_PROFILE="${RUN_PROFILE:-1}"

VAL_N="${VAL_N:-4}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-8192}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-12288}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
EVAL_TOP_P="${EVAL_TOP_P:-0.8}"
EVAL_TOP_K="${EVAL_TOP_K:--1}"
EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.55}"

MAX_STEPS="${MAX_STEPS:-600}"
PROFILE_STEPS="${PROFILE_STEPS:-80}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"
OPSD_TRAIN_BATCH_SIZE="${OPSD_TRAIN_BATCH_SIZE:-2}"
OPSD_GRAD_ACCUM="${OPSD_GRAD_ACCUM:-8}"
OPSD_VLLM_GPU_MEMORY_UTILIZATION="${OPSD_VLLM_GPU_MEMORY_UTILIZATION:-0.35}"
TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-0.9}"
LOSS_TEMPERATURE="${LOSS_TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
TOP_K_LOSS="${TOP_K_LOSS:-2048}"
JSD_TOKEN_CLIP="${JSD_TOKEN_CLIP:-0.05}"

VAULT_WINDOW_SIZE="${VAULT_WINDOW_SIZE:-256}"
VAULT_MIN_VALID="${VAULT_MIN_VALID:-16}"
VAULT_INIT_KAPPA="${VAULT_INIT_KAPPA:-0.3}"
VAULT_INIT_NORM="${VAULT_INIT_NORM:-0.3}"
VAULT_NORM_ALPHA="${VAULT_NORM_ALPHA:-0.01}"
VAULT_C="${VAULT_C:-1.0}"
VAULT_S_MIN="${VAULT_S_MIN:-0.05}"
VAULT_VALID_W_MIN="${VAULT_VALID_W_MIN:-0.05}"
VAULT_VALID_W_MAX="${VAULT_VALID_W_MAX:-2.0}"
VAULT_CENSORED_W_MIN="${VAULT_CENSORED_W_MIN:-0.0}"
VAULT_CENSORED_W_MAX="${VAULT_CENSORED_W_MAX:-0.1}"
VAULT_SURVIVAL_MODE="${VAULT_SURVIVAL_MODE:-bell}"
VAULT_BELL_SIGMA="${VAULT_BELL_SIGMA:-0.5}"
VAULT_CLIP_MODE="${VAULT_CLIP_MODE:-global}"
# The previous VAULT main run effectively used valid-only kappa updates in code.
# Keep that behavior here so RQ2/RQ3 are fair against the existing main result.
VAULT_KAPPA_UPDATE_MODE="${VAULT_KAPPA_UPDATE_MODE:-valid}"
VAULT_VALIDITY_MODE="${VAULT_VALIDITY_MODE:-boxed}"

TEMP_TAG="$(printf '%s' "$TRAIN_TEMPERATURE" | sed 's/\./p/g')"
SIGMA_TAG="$(printf '%s' "$VAULT_BELL_SIGMA" | sed 's/\./p/g')"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log_msg() {
  echo "[$(timestamp)] $*"
}

wait_for_gpus() {
  if [ "$WAIT_FOR_GPUS" != "1" ]; then
    return 0
  fi
  log_msg "Waiting for GPUs $GPUS to have <= ${GPU_WAIT_MAX_USED_MB}MiB used memory each."
  while true; do
    local busy=0
    local used
    for gpu in "${GPU_LIST[@]}"; do
      used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
      if [ "${used:-999999}" -gt "$GPU_WAIT_MAX_USED_MB" ]; then
        busy=1
        log_msg "GPU $gpu is still busy (${used}MiB used)."
      fi
    done
    if [ "$busy" -eq 0 ]; then
      log_msg "GPUs $GPUS are available."
      return 0
    fi
    sleep "$GPU_WAIT_POLL_SECONDS"
  done
}

json_done() {
  local file="$1"
  [ -s "$file" ] || return 1
  "$PYTHON" - "$file" "$VAL_N" <<'PY'
import json
import sys
path = sys.argv[1]
val_n = int(sys.argv[2])
try:
    data = json.load(open(path, "r", encoding="utf-8"))
except Exception:
    sys.exit(1)
if data.get("val_n") != val_n:
    sys.exit(1)
if data.get("num_problems", 0) <= 0 or not data.get("results"):
    sys.exit(1)
sys.exit(0)
PY
}

latest_checkpoint() {
  local out_dir="$1"
  [ -d "$out_dir" ] || return 0
  find "$out_dir" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1
}

verify_assets() {
  [ -d "$MODEL_PATH" ] || { echo "Missing model directory: $MODEL_PATH" >&2; exit 1; }
  "$PYTHON" - "$MODEL_PATH" <<'PY'
import sys
from datasets import load_dataset
from transformers import AutoTokenizer
model_path = sys.argv[1]
AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
for name, kwargs in [
    ("siyanzhao/Openthoughts_math_30k_opsd", {}),
    ("HuggingFaceH4/aime_2024", {"split": "train"}),
    ("yentinglin/aime_2025", {"split": "train", "trust_remote_code": True}),
    ("MathArena/hmmt_feb_2025", {"split": "train", "trust_remote_code": True}),
]:
    ds = load_dataset(name, **kwargs)
    length = len(ds["train"]) if isinstance(ds, dict) and "train" in ds else len(ds)
    print(f"cached {name}: {length} rows")
PY
}

import_reference_results() {
  log_msg "Importing existing OPSD and VAULT full eval results from $MAIN_EXP_DIR"
  for dataset in aime24 aime25 hmmt25; do
    mkdir -p "$EVAL_DIR/$dataset"
    cp "$MAIN_EXP_DIR/eval_results/$dataset/opsd_fixed_t${TEMP_TAG}.json" \
      "$EVAL_DIR/$dataset/opsd_fixed_t${TEMP_TAG}.json"
    cp "$MAIN_EXP_DIR/eval_results/$dataset/opsd_vault_t${TEMP_TAG}.json" \
      "$EVAL_DIR/$dataset/vault_full_t${TEMP_TAG}.json"
  done
}

train_opsd() {
  local method="$1"
  local run_name="$2"
  local log_file="$3"
  local port="$4"
  local steps="$5"
  shift 5
  local extra_args=("$@")
  local out_dir="$OUTPUT_DIR/$run_name"

  if [ -s "$out_dir/adapter_model.safetensors" ]; then
    log_msg "Skip $method: $out_dir already has adapter_model.safetensors"
    return 0
  fi

  local resume_checkpoint=""
  if [ "$AUTO_RESUME" = "1" ]; then
    resume_checkpoint="$(latest_checkpoint "$out_dir")"
  fi
  local resume_args=()
  if [ -n "$resume_checkpoint" ]; then
    log_msg "Resume $method from $resume_checkpoint"
    resume_args=(--resume_from_checkpoint "$resume_checkpoint")
  fi

  wait_for_gpus
  log_msg "Start $method training on GPUs $GPUS"
  log_msg "Train config: steps=$steps, rollout_temp=$TRAIN_TEMPERATURE, loss_temp=$LOSS_TEMPERATURE, ctx=$MAX_LENGTH, gen=$MAX_COMPLETION_LENGTH, per_device_batch=$OPSD_TRAIN_BATCH_SIZE, grad_accum=$OPSD_GRAD_ACCUM"
  printf '\n[%s] Launch %s training, output=%s\n' "$(timestamp)" "$method" "$out_dir" >> "$log_file"
  (
    cd "$OPSD_DIR"
    env CUDA_VISIBLE_DEVICES="$GPUS" "$ACCELERATE" launch \
      --config_file "$OPSD_DIR/accelerate.yaml" \
      --num_processes "$NUM_GPUS" \
      --gradient_accumulation_steps "$OPSD_GRAD_ACCUM" \
      --main_process_port "$port" \
      "$OPSD_DIR/opsd_train.py" \
      --model_name_or_path "$MODEL_PATH" \
      --learning_rate "$LEARNING_RATE" \
      --max_grad_norm 0.1 \
      --per_device_train_batch_size "$OPSD_TRAIN_BATCH_SIZE" \
      --gradient_checkpointing \
      --gradient_accumulation_steps "$OPSD_GRAD_ACCUM" \
      --output_dir "$OUTPUT_DIR" \
      --run_config "$run_name" \
      --max_steps "$steps" \
      --save_steps "$SAVE_STEPS" \
      --save_total_limit "$SAVE_TOTAL_LIMIT" \
      --logging_steps "$LOGGING_STEPS" \
      --attn_implementation flash_attention_2 \
      --torch_dtype bfloat16 \
      --max_length "$MAX_LENGTH" \
      --max_completion_length "$MAX_COMPLETION_LENGTH" \
      --beta 0 \
      --use_vllm \
      --vllm_mode colocate \
      --vllm_gpu_memory_utilization "$OPSD_VLLM_GPU_MEMORY_UTILIZATION" \
      --vllm_tensor_parallel_size 1 \
      --use_peft \
      --lora_r 64 \
      --lora_alpha 128 \
      --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
      --temperature "$TRAIN_TEMPERATURE" \
      --loss_temperature "$LOSS_TEMPERATURE" \
      --top_p "$TOP_P" \
      --top_k "$TOP_K" \
      --lmbda 1 \
      --fixed_teacher \
      --top_k_loss "$TOP_K_LOSS" \
      --jsd_token_clip "$JSD_TOKEN_CLIP" \
      --wandb_project "VAULT-Supp-Qwen3-1.7B-AIME-HMMT" \
      "${extra_args[@]}" \
      "${resume_args[@]}"
  ) >> "$log_file" 2>&1
  log_msg "Finished $method training"
}

common_vault_args() {
  printf '%s\n' \
    --use_vault \
    --vault_window_size "$VAULT_WINDOW_SIZE" \
    --vault_min_valid "$VAULT_MIN_VALID" \
    --vault_init_kappa "$VAULT_INIT_KAPPA" \
    --vault_init_norm "$VAULT_INIT_NORM" \
    --vault_norm_alpha "$VAULT_NORM_ALPHA" \
    --vault_c "$VAULT_C" \
    --vault_s_min "$VAULT_S_MIN" \
    --vault_valid_w_min "$VAULT_VALID_W_MIN" \
    --vault_valid_w_max "$VAULT_VALID_W_MAX" \
    --vault_censored_w_min "$VAULT_CENSORED_W_MIN" \
    --vault_censored_w_max "$VAULT_CENSORED_W_MAX" \
    --vault_survival_mode "$VAULT_SURVIVAL_MODE" \
    --vault_bell_sigma "$VAULT_BELL_SIGMA" \
    --vault_clip_mode "$VAULT_CLIP_MODE" \
    --vault_kappa_update_mode "$VAULT_KAPPA_UPDATE_MODE" \
    --vault_validity_mode "$VAULT_VALIDITY_MODE"
}

eval_one() {
  local method="$1"
  local checkpoint_dir="$2"
  local dataset="$3"
  local gpu="$4"
  local output_file="$EVAL_DIR/$dataset/$method.json"
  local log_file="$LOG_DIR/eval_${dataset}_${method}.log"
  mkdir -p "$EVAL_DIR/$dataset"

  if json_done "$output_file"; then
    log_msg "Skip eval $method on $dataset: already complete."
    return 0
  fi

  log_msg "Start eval $method on $dataset using GPU $gpu"
  env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$OPSD_DIR/eval/evaluate_math.py" \
    --base_model "$MODEL_PATH" \
    --checkpoint_dir "$checkpoint_dir" \
    --dataset "$dataset" \
    --no_thinking \
    --temperature "$EVAL_TEMPERATURE" \
    --top_p "$EVAL_TOP_P" \
    --top_k "$EVAL_TOP_K" \
    --max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
    --max_model_len "$EVAL_MAX_MODEL_LEN" \
    --gpu_memory_utilization "$EVAL_GPU_MEMORY_UTILIZATION" \
    --tensor_parallel_size 1 \
    --val_n "$VAL_N" \
    --output_file "$output_file" \
    > "$log_file" 2>&1
  log_msg "Finished eval $method on $dataset"
}

eval_all() {
  [ "$RUN_EVAL" = "1" ] || return 0
  local method="$1"
  local checkpoint_dir="$2"
  local datasets=(aime24 aime25 hmmt25)
  local pids=()
  local idx=0

  wait_for_gpus
  for dataset in "${datasets[@]}"; do
    eval_one "$method" "$checkpoint_dir" "$dataset" "${GPU_LIST[$idx]}" &
    pids+=("$!")
    idx=$((idx + 1))
  done

  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [ "$failed" -ne 0 ]; then
    echo "At least one eval failed for method $method. Check $LOG_DIR/eval_*_${method}.log" >&2
    exit 1
  fi
}

run_variant() {
  local method="$1"
  local label="$2"
  local run_name="$3"
  local port="$4"
  shift 4
  local extra_args=("$@")

  train_opsd "$label" "$run_name" "$LOG_DIR/train_${method}.log" "$port" "$MAX_STEPS" "${extra_args[@]}"
  "$PYTHON" "$EXP_DIR/summarize_supplement.py" --exp_dir "$EXP_DIR" > "$LOG_DIR/summary_after_train_${method}.log" 2>&1 || true
}

eval_variant() {
  local method="$1"
  local run_name="$2"

  eval_all "$method" "$OUTPUT_DIR/$run_name"
  "$PYTHON" "$EXP_DIR/summarize_supplement.py" --exp_dir "$EXP_DIR" > "$LOG_DIR/summary_after_eval_${method}.log" 2>&1 || true
}

main() {
  log_msg "Experiment directory: $EXP_DIR"
  log_msg "OPSD source directory: $OPSD_DIR"
  log_msg "Main reference experiment: $MAIN_EXP_DIR"
  log_msg "Model path: $MODEL_PATH"
  log_msg "GPUs: $GPUS"
  log_msg "Fair config: train steps=$MAX_STEPS, train T=$TRAIN_TEMPERATURE, eval T=$EVAL_TEMPERATURE, val_n=$VAL_N, eval max_new_tokens=$EVAL_MAX_NEW_TOKENS"
  log_msg "VAULT common config: survival=$VAULT_SURVIVAL_MODE, sigma=$VAULT_BELL_SIGMA, clip=$VAULT_CLIP_MODE, kappa_update=$VAULT_KAPPA_UPDATE_MODE, validity=$VAULT_VALIDITY_MODE"

  verify_assets
  import_reference_results

  if [ "$RUN_RQ2_RQ3" = "1" ]; then
    mapfile -t vault_args < <(common_vault_args)
    hard_filter_method="hard_filter_t${TEMP_TAG}"
    hard_filter_run="hard_filter_t${TEMP_TAG}_steps${MAX_STEPS}_ctx${MAX_LENGTH}_gen${MAX_COMPLETION_LENGTH}"
    fixed_penalty_method="fixed_penalty_0p5_t${TEMP_TAG}"
    fixed_penalty_run="fixed_penalty_0p5_t${TEMP_TAG}_steps${MAX_STEPS}_ctx${MAX_LENGTH}_gen${MAX_COMPLETION_LENGTH}"
    sweet_spot_method="sweet_spot_t${TEMP_TAG}"
    sweet_spot_run="sweet_spot_t${TEMP_TAG}_steps${MAX_STEPS}_ctx${MAX_LENGTH}_gen${MAX_COMPLETION_LENGTH}_w${VAULT_WINDOW_SIZE}_s${SIGMA_TAG}"
    survival_only_method="survival_only_t${TEMP_TAG}"
    survival_only_run="survival_only_t${TEMP_TAG}_steps${MAX_STEPS}_ctx${MAX_LENGTH}_gen${MAX_COMPLETION_LENGTH}_w${VAULT_WINDOW_SIZE}_s${SIGMA_TAG}"

    log_msg "RQ2/RQ3 schedule: train all variants first, then evaluate all variants."

    run_variant \
      "$hard_filter_method" \
      "RQ2 hard filter" \
      "$hard_filter_run" \
      19331 \
      "${vault_args[@]}" \
      --vault_weight_mode hard_filter \
      --vault_clip_mode by_validity \
      --vault_censored_w_min 0.0 \
      --vault_censored_w_max 0.0

    run_variant \
      "$fixed_penalty_method" \
      "RQ2 fixed censoring penalty 0.5" \
      "$fixed_penalty_run" \
      19332 \
      "${vault_args[@]}" \
      --vault_weight_mode fixed_penalty \
      --vault_fixed_censored_weight 0.5

    run_variant \
      "$sweet_spot_method" \
      "RQ3 sweet-spot-only ablation" \
      "$sweet_spot_run" \
      19333 \
      "${vault_args[@]}" \
      --vault_weight_mode sweet_spot

    run_variant \
      "$survival_only_method" \
      "RQ3 survival-only ablation" \
      "$survival_only_run" \
      19334 \
      "${vault_args[@]}" \
      --vault_weight_mode survival_only

    log_msg "All RQ2/RQ3 trainings finished. Start batched evaluation."
    eval_variant "$hard_filter_method" "$hard_filter_run"
    eval_variant "$fixed_penalty_method" "$fixed_penalty_run"
    eval_variant "$sweet_spot_method" "$sweet_spot_run"
    eval_variant "$survival_only_method" "$survival_only_run"
  fi

  if [ "$RUN_PROFILE" = "1" ]; then
    mapfile -t vault_args < <(common_vault_args)
    train_opsd \
      "RQ5 OPSD profile" \
      "profile_opsd_steps${PROFILE_STEPS}_ctx${MAX_LENGTH}_gen${MAX_COMPLETION_LENGTH}" \
      "$LOG_DIR/profile_opsd.log" \
      19335 \
      "$PROFILE_STEPS"
    train_opsd \
      "RQ5 VAULT profile" \
      "profile_vault_steps${PROFILE_STEPS}_ctx${MAX_LENGTH}_gen${MAX_COMPLETION_LENGTH}" \
      "$LOG_DIR/profile_vault.log" \
      19336 \
      "$PROFILE_STEPS" \
      "${vault_args[@]}" \
      --vault_weight_mode vault
  fi

  "$PYTHON" "$EXP_DIR/summarize_supplement.py" --exp_dir "$EXP_DIR" | tee "$LOG_DIR/final_summary.log"
  log_msg "All supplemental jobs finished."
}

main "$@"
