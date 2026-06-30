#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AIME_EXP_DIR="${AIME_EXP_DIR:-$ROOT_DIR/experiments/aime_hmmt}"
GSM_EXP_DIR="${GSM_EXP_DIR:-$ROOT_DIR/experiments/gsm8k}"
OPSD_DIR="${OPSD_DIR:-$ROOT_DIR/OPSD}"

PYTHON="${PYTHON:-python}"
ACCELERATE="${ACCELERATE:-accelerate}"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/models/Qwen3-1.7B}"
MATH_TRAIN_DATASET="${MATH_TRAIN_DATASET:-siyanzhao/Openthoughts_math_30k_opsd}"
GSM8K_DATASET="${GSM8K_DATASET:-$ROOT_DIR/data/gsm8k_opsd}"

GPUS="${GPUS:-4,5,6,7}"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
NUM_GPUS="${#GPU_LIST[@]}"
if [ "$NUM_GPUS" -ne 4 ]; then
  echo "This script expects exactly four GPUs, got GPUS=$GPUS" >&2
  exit 1
fi

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
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

RUN_AIME_BASE="${RUN_AIME_BASE:-1}"
RUN_AIME_GRPO="${RUN_AIME_GRPO:-1}"
RUN_GSM_BASE="${RUN_GSM_BASE:-1}"
RUN_GSM_GRPO="${RUN_GSM_GRPO:-1}"
RUN_GSM_FIXED_OPSD="${RUN_GSM_FIXED_OPSD:-1}"
RUN_GSM_VAULT_OPSD="${RUN_GSM_VAULT_OPSD:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
AUTO_RESUME="${AUTO_RESUME:-1}"

VAL_N="${VAL_N:-4}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
EVAL_TOP_P="${EVAL_TOP_P:-0.8}"
EVAL_TOP_K="${EVAL_TOP_K:--1}"
AIME_EVAL_MAX_NEW_TOKENS="${AIME_EVAL_MAX_NEW_TOKENS:-8192}"
AIME_EVAL_MAX_MODEL_LEN="${AIME_EVAL_MAX_MODEL_LEN:-12288}"
GSM_EVAL_MAX_NEW_TOKENS="${GSM_EVAL_MAX_NEW_TOKENS:-2048}"
GSM_EVAL_MAX_MODEL_LEN="${GSM_EVAL_MAX_MODEL_LEN:-4096}"
EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.55}"

MAX_STEPS="${MAX_STEPS:-600}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"
TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-0.9}"
LOSS_TEMPERATURE="${LOSS_TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
TOP_K_LOSS="${TOP_K_LOSS:-2048}"
JSD_TOKEN_CLIP="${JSD_TOKEN_CLIP:-0.05}"

OPSD_TRAIN_BATCH_SIZE="${OPSD_TRAIN_BATCH_SIZE:-2}"
OPSD_GRAD_ACCUM="${OPSD_GRAD_ACCUM:-8}"
OPSD_VLLM_GPU_MEMORY_UTILIZATION="${OPSD_VLLM_GPU_MEMORY_UTILIZATION:-0.35}"

GRPO_TRAIN_BATCH_SIZE="${GRPO_TRAIN_BATCH_SIZE:-1}"
GRPO_GRAD_ACCUM="${GRPO_GRAD_ACCUM:-8}"
GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-4}"
GRPO_NUM_ITERATIONS="${GRPO_NUM_ITERATIONS:-2}"
GRPO_TEMPERATURE="${GRPO_TEMPERATURE:-1.2}"
GRPO_MAX_PROMPT_LENGTH="${GRPO_MAX_PROMPT_LENGTH:-2048}"
GRPO_VLLM_GPU_MEMORY_UTILIZATION="${GRPO_VLLM_GPU_MEMORY_UTILIZATION:-0.35}"

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
VAULT_KAPPA_UPDATE_MODE="${VAULT_KAPPA_UPDATE_MODE:-all}"
VAULT_VALIDITY_MODE="${VAULT_VALIDITY_MODE:-boxed}"

TEMP_TAG="$(printf '%s' "$TRAIN_TEMPERATURE" | sed 's/\./p/g')"
SIGMA_TAG="$(printf '%s' "$VAULT_BELL_SIGMA" | sed 's/\./p/g')"
GRPO_RUN="grpo_steps${MAX_STEPS}_ctx${MAX_LENGTH}_gen${MAX_COMPLETION_LENGTH}_ng${GRPO_NUM_GENERATIONS}"
GSM_FIXED_RUN="opsd_fixed_t${TEMP_TAG}_steps${MAX_STEPS}_ctx${MAX_LENGTH}_gen${MAX_COMPLETION_LENGTH}"
GSM_VAULT_RUN="opsd_vault_t${TEMP_TAG}_steps${MAX_STEPS}_ctx${MAX_LENGTH}_gen${MAX_COMPLETION_LENGTH}_w${VAULT_WINDOW_SIZE}_s${SIGMA_TAG}"

mkdir -p "$AIME_EXP_DIR/logs" "$AIME_EXP_DIR/outputs" "$AIME_EXP_DIR/eval_results"
mkdir -p "$GSM_EXP_DIR/logs" "$GSM_EXP_DIR/outputs" "$GSM_EXP_DIR/eval_results/gsm8k"

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
if data.get("val_n") != val_n or data.get("num_problems", 0) <= 0 or not data.get("results"):
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
  [ -d "$GSM8K_DATASET" ] || { echo "Missing GSM8K dataset directory: $GSM8K_DATASET" >&2; exit 1; }
  "$PYTHON" - "$MODEL_PATH" "$GSM8K_DATASET" <<'PY'
import sys
from datasets import load_from_disk
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
ds = load_from_disk(sys.argv[2])
print(ds)
PY
}

eval_math_one() {
  local exp_dir="$1"
  local method="$2"
  local checkpoint_dir="$3"
  local dataset="$4"
  local gpu="$5"
  local output_file="$exp_dir/eval_results/$dataset/$method.json"
  local log_file="$exp_dir/logs/eval_${dataset}_${method}.log"
  mkdir -p "$exp_dir/eval_results/$dataset"

  if json_done "$output_file"; then
    log_msg "Skip eval $method on $dataset: already complete."
    return 0
  fi

  local checkpoint_args=()
  if [ -n "$checkpoint_dir" ]; then
    checkpoint_args=(--checkpoint_dir "$checkpoint_dir")
  fi

  log_msg "Start AIME/HMMT eval $method on $dataset using GPU $gpu"
  env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$OPSD_DIR/eval/evaluate_math.py" \
    --base_model "$MODEL_PATH" \
    "${checkpoint_args[@]}" \
    --dataset "$dataset" \
    --no_thinking \
    --temperature "$EVAL_TEMPERATURE" \
    --top_p "$EVAL_TOP_P" \
    --top_k "$EVAL_TOP_K" \
    --max_new_tokens "$AIME_EVAL_MAX_NEW_TOKENS" \
    --max_model_len "$AIME_EVAL_MAX_MODEL_LEN" \
    --gpu_memory_utilization "$EVAL_GPU_MEMORY_UTILIZATION" \
    --tensor_parallel_size 1 \
    --val_n "$VAL_N" \
    --output_file "$output_file" \
    > "$log_file" 2>&1
  log_msg "Finished AIME/HMMT eval $method on $dataset"
}

eval_math_all() {
  local exp_dir="$1"
  local method="$2"
  local checkpoint_dir="$3"
  [ "$RUN_EVAL" = "1" ] || return 0

  wait_for_gpus
  local datasets=(aime24 aime25 hmmt25)
  local pids=()
  local idx=0
  for dataset in "${datasets[@]}"; do
    eval_math_one "$exp_dir" "$method" "$checkpoint_dir" "$dataset" "${GPU_LIST[$idx]}" &
    pids+=("$!")
    idx=$((idx + 1))
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [ "$failed" -ne 0 ]; then
    echo "At least one AIME/HMMT eval failed for $method." >&2
    exit 1
  fi
}

eval_gsm_one() {
  local method="$1"
  local checkpoint_dir="$2"
  local gpu="$3"
  local output_file="$GSM_EXP_DIR/eval_results/gsm8k/$method.json"
  local log_file="$GSM_EXP_DIR/logs/eval_gsm8k_${method}.log"

  if json_done "$output_file"; then
    log_msg "Skip GSM8K eval $method: already complete."
    return 0
  fi

  local checkpoint_args=()
  if [ -n "$checkpoint_dir" ]; then
    checkpoint_args=(--checkpoint_dir "$checkpoint_dir")
  fi

  log_msg "Start GSM8K eval $method using GPU $gpu"
  env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$OPSD_DIR/eval/evaluate_gsm8k.py" \
    --base_model "$MODEL_PATH" \
    "${checkpoint_args[@]}" \
    --dataset_path "$GSM8K_DATASET" \
    --split test \
    --no_thinking \
    --temperature "$EVAL_TEMPERATURE" \
    --top_p "$EVAL_TOP_P" \
    --top_k "$EVAL_TOP_K" \
    --max_new_tokens "$GSM_EVAL_MAX_NEW_TOKENS" \
    --max_model_len "$GSM_EVAL_MAX_MODEL_LEN" \
    --gpu_memory_utilization "$EVAL_GPU_MEMORY_UTILIZATION" \
    --tensor_parallel_size 1 \
    --val_n "$VAL_N" \
    --run_label "$method" \
    --output_file "$output_file" \
    > "$log_file" 2>&1
  log_msg "Finished GSM8K eval $method"
}

eval_gsm_all() {
  [ "$RUN_EVAL" = "1" ] || return 0
  wait_for_gpus
  local pids=()
  eval_gsm_one "base" "" "${GPU_LIST[0]}" &
  pids+=("$!")
  eval_gsm_one "grpo" "$GSM_EXP_DIR/outputs/$GRPO_RUN" "${GPU_LIST[1]}" &
  pids+=("$!")
  eval_gsm_one "opsd_fixed_t${TEMP_TAG}" "$GSM_EXP_DIR/outputs/$GSM_FIXED_RUN" "${GPU_LIST[2]}" &
  pids+=("$!")
  eval_gsm_one "opsd_vault_t${TEMP_TAG}" "$GSM_EXP_DIR/outputs/$GSM_VAULT_RUN" "${GPU_LIST[3]}" &
  pids+=("$!")

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [ "$failed" -ne 0 ]; then
    echo "At least one GSM8K eval failed." >&2
    exit 1
  fi
}

train_opsd() {
  local exp_dir="$1"
  local dataset_source="$2"
  local method="$3"
  local run_name="$4"
  local log_file="$5"
  shift 5
  local extra_args=("$@")
  local out_dir="$exp_dir/outputs/$run_name"

  if [ -s "$out_dir/adapter_model.safetensors" ]; then
    log_msg "Skip $method training: adapter exists at $out_dir"
    return 0
  fi

  local resume_checkpoint=""
  if [ "$AUTO_RESUME" = "1" ]; then
    resume_checkpoint="$(latest_checkpoint "$out_dir")"
  fi
  local resume_args=()
  if [ -n "$resume_checkpoint" ]; then
    resume_args=(--resume_from_checkpoint "$resume_checkpoint")
    log_msg "Resume $method from $resume_checkpoint"
  fi

  wait_for_gpus
  log_msg "Start $method training on GPUs $GPUS with dataset=$dataset_source"
  (
    cd "$OPSD_DIR"
    env CUDA_VISIBLE_DEVICES="$GPUS" "$ACCELERATE" launch \
      --config_file "$OPSD_DIR/accelerate.yaml" \
      --num_processes "$NUM_GPUS" \
      --gradient_accumulation_steps "$OPSD_GRAD_ACCUM" \
      --main_process_port "${MAIN_PROCESS_PORT:-19241}" \
      "$OPSD_DIR/opsd_train.py" \
      --model_name_or_path "$MODEL_PATH" \
      --dataset_name "$dataset_source" \
      --dataset_train_split train \
      --learning_rate "$LEARNING_RATE" \
      --max_grad_norm 0.1 \
      --per_device_train_batch_size "$OPSD_TRAIN_BATCH_SIZE" \
      --gradient_checkpointing \
      --gradient_accumulation_steps "$OPSD_GRAD_ACCUM" \
      --output_dir "$exp_dir/outputs" \
      --run_config "$run_name" \
      --max_steps "$MAX_STEPS" \
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
      --wandb_project "Fair-VAULT-Qwen3-1.7B" \
      "${extra_args[@]}" \
      "${resume_args[@]}"
  ) > "$log_file" 2>&1
  log_msg "Finished $method training"
}

train_grpo() {
  local exp_dir="$1"
  local dataset_source="$2"
  local run_name="$3"
  local log_file="$4"
  local out_dir="$exp_dir/outputs/$run_name"

  if [ -s "$out_dir/adapter_model.safetensors" ]; then
    log_msg "Skip GRPO training: adapter exists at $out_dir"
    return 0
  fi

  wait_for_gpus
  log_msg "Start GRPO training on GPUs $GPUS with dataset=$dataset_source"
  (
    cd "$OPSD_DIR"
    env CUDA_VISIBLE_DEVICES="$GPUS" "$ACCELERATE" launch \
      --config_file "$OPSD_DIR/accelerate.yaml" \
      --num_processes "$NUM_GPUS" \
      --gradient_accumulation_steps "$GRPO_GRAD_ACCUM" \
      --main_process_port "${MAIN_PROCESS_PORT:-19240}" \
      "$OPSD_DIR/grpo_train.py" \
      --model_name_or_path "$MODEL_PATH" \
      --dataset_name "$dataset_source" \
      --dataset_train_split train \
      --learning_rate "$LEARNING_RATE" \
      --per_device_train_batch_size "$GRPO_TRAIN_BATCH_SIZE" \
      --gradient_accumulation_steps "$GRPO_GRAD_ACCUM" \
      --output_dir "$exp_dir/outputs" \
      --run_config "$run_name" \
      --max_steps "$MAX_STEPS" \
      --save_steps "$SAVE_STEPS" \
      --save_total_limit "$SAVE_TOTAL_LIMIT" \
      --logging_steps "$LOGGING_STEPS" \
      --attn_implementation flash_attention_2 \
      --torch_dtype bfloat16 \
      --max_prompt_length "$GRPO_MAX_PROMPT_LENGTH" \
      --max_completion_length "$MAX_COMPLETION_LENGTH" \
      --num_generations "$GRPO_NUM_GENERATIONS" \
      --num_iterations "$GRPO_NUM_ITERATIONS" \
      --temperature "$GRPO_TEMPERATURE" \
      --top_p "$TOP_P" \
      --top_k "$TOP_K" \
      --use_vllm \
      --vllm_mode colocate \
      --vllm_gpu_memory_utilization "$GRPO_VLLM_GPU_MEMORY_UTILIZATION" \
      --vllm_tensor_parallel_size 1 \
      --use_peft \
      --lora_r 64 \
      --lora_alpha 128 \
      --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
      --gradient_checkpointing \
      --beta 0.0 \
      --loss_type grpo \
      --scale_rewards group \
      --wandb_project "Fair-GRPO-Qwen3-1.7B"
  ) > "$log_file" 2>&1
  log_msg "Finished GRPO training"
}

summarize_aime() {
  "$PYTHON" "$AIME_EXP_DIR/summarize_results.py" --exp_dir "$AIME_EXP_DIR" > "$AIME_EXP_DIR/logs/final_summary_with_base_grpo.log" 2>&1 || true
  "$PYTHON" "$OPSD_DIR/eval/summarize_eval_results.py" \
    --eval_dir "$AIME_EXP_DIR/eval_results" \
    --output "$AIME_EXP_DIR/fair_summary.md" \
    --title "Qwen3-1.7B AIME/HMMT Fair Summary" \
    --datasets aime24:AIME24 aime25:AIME25 hmmt25:HMMT25 \
    --methods base:Base grpo:GRPO opsd_fixed_t${TEMP_TAG}:"OPSD fixed T=${TRAIN_TEMPERATURE}" opsd_vault_t${TEMP_TAG}:"VAULT T=${TRAIN_TEMPERATURE}" \
    > "$AIME_EXP_DIR/logs/fair_summary.log" 2>&1 || true
}

summarize_gsm() {
  "$PYTHON" "$OPSD_DIR/eval/summarize_eval_results.py" \
    --eval_dir "$GSM_EXP_DIR/eval_results" \
    --output "$GSM_EXP_DIR/summary.md" \
    --title "Qwen3-1.7B GSM8K Fair Summary" \
    --datasets gsm8k:GSM8K \
    --methods base:Base grpo:GRPO opsd_fixed_t${TEMP_TAG}:"OPSD fixed T=${TRAIN_TEMPERATURE}" opsd_vault_t${TEMP_TAG}:"VAULT T=${TRAIN_TEMPERATURE}" \
    > "$GSM_EXP_DIR/logs/final_summary.log" 2>&1 || true
}

main() {
  log_msg "AIME/HMMT exp: $AIME_EXP_DIR"
  log_msg "GSM8K exp: $GSM_EXP_DIR"
  log_msg "Model: $MODEL_PATH"
  log_msg "GPUs: $GPUS"
  log_msg "Fair eval: non-thinking, val_n=$VAL_N, temp=$EVAL_TEMPERATURE, top_p=$EVAL_TOP_P, top_k=$EVAL_TOP_K"

  verify_assets

  if [ "$RUN_AIME_BASE" = "1" ]; then
    eval_math_all "$AIME_EXP_DIR" "base" ""
    summarize_aime
  fi

  if [ "$RUN_AIME_GRPO" = "1" ]; then
    MAIN_PROCESS_PORT="${AIME_GRPO_PORT:-19240}" train_grpo \
      "$AIME_EXP_DIR" "$MATH_TRAIN_DATASET" "$GRPO_RUN" "$AIME_EXP_DIR/logs/train_grpo.log"
    eval_math_all "$AIME_EXP_DIR" "grpo" "$AIME_EXP_DIR/outputs/$GRPO_RUN"
    summarize_aime
  fi

  if [ "$RUN_GSM_FIXED_OPSD" = "1" ]; then
    MAIN_PROCESS_PORT="${GSM_FIXED_PORT:-19241}" train_opsd \
      "$GSM_EXP_DIR" "$GSM8K_DATASET" "GSM8K fixed-temperature OPSD" "$GSM_FIXED_RUN" \
      "$GSM_EXP_DIR/logs/train_opsd_fixed_t${TEMP_TAG}.log"
  fi

  if [ "$RUN_GSM_VAULT_OPSD" = "1" ]; then
    MAIN_PROCESS_PORT="${GSM_VAULT_PORT:-19242}" train_opsd \
      "$GSM_EXP_DIR" "$GSM8K_DATASET" "GSM8K VAULT" "$GSM_VAULT_RUN" \
      "$GSM_EXP_DIR/logs/train_opsd_vault_t${TEMP_TAG}.log" \
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
  fi

  if [ "$RUN_GSM_GRPO" = "1" ]; then
    MAIN_PROCESS_PORT="${GSM_GRPO_PORT:-19243}" train_grpo \
      "$GSM_EXP_DIR" "$GSM8K_DATASET" "$GRPO_RUN" "$GSM_EXP_DIR/logs/train_grpo.log"
  fi

  if [ "$RUN_GSM_BASE" = "1" ] || [ "$RUN_GSM_GRPO" = "1" ] || [ "$RUN_GSM_FIXED_OPSD" = "1" ] || [ "$RUN_GSM_VAULT_OPSD" = "1" ]; then
    eval_gsm_all
    summarize_gsm
  fi

  summarize_aime
  summarize_gsm
  log_msg "All requested fair comparison jobs finished."
}

main "$@"
