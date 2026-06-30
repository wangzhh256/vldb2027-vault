#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_DIR="${EXP_DIR:-$ROOT_DIR/experiments/aime_hmmt}"
OPSD_DIR="${OPSD_DIR:-$ROOT_DIR/OPSD}"
LOG_DIR="$EXP_DIR/logs"
OUTPUT_DIR="$EXP_DIR/outputs"
EVAL_DIR="$EXP_DIR/eval_results"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$EVAL_DIR"

PYTHON="${PYTHON:-python}"
ACCELERATE="${ACCELERATE:-accelerate}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-1.7B}"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/models/Qwen3-1.7B}"

GPUS="${GPUS:-4,5,6,7}"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
NUM_GPUS="${#GPU_LIST[@]}"
if [ "$NUM_GPUS" -lt 2 ]; then
  echo "Expected at least two GPUs in GPUS, got: $GPUS" >&2
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

RUN_FIXED_OPSD="${RUN_FIXED_OPSD:-0}"
RUN_VAULT_OPSD="${RUN_VAULT_OPSD:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
IMPORT_FIXED_BASELINE="${IMPORT_FIXED_BASELINE:-1}"
BASELINE_EXP_DIR="${BASELINE_EXP_DIR:-$ROOT_DIR/experiments/aime_hmmt_reference}"
AUTO_RESUME="${AUTO_RESUME:-1}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
GPU_WAIT_MAX_USED_MB="${GPU_WAIT_MAX_USED_MB:-1000}"
GPU_WAIT_POLL_SECONDS="${GPU_WAIT_POLL_SECONDS:-120}"

VAL_N="${VAL_N:-4}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-8192}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-12288}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
EVAL_TOP_P="${EVAL_TOP_P:-0.8}"
EVAL_TOP_K="${EVAL_TOP_K:--1}"
EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.55}"

MAX_STEPS="${MAX_STEPS:-600}"
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
VAULT_KAPPA_UPDATE_MODE="${VAULT_KAPPA_UPDATE_MODE:-all}"
VAULT_VALIDITY_MODE="${VAULT_VALIDITY_MODE:-boxed}"

TEMP_TAG="$(printf '%s' "$TRAIN_TEMPERATURE" | sed 's/\./p/g')"
SIGMA_TAG="$(printf '%s' "$VAULT_BELL_SIGMA" | sed 's/\./p/g')"
FIXED_RUN="opsd_fixed_t${TEMP_TAG}_steps${MAX_STEPS}_ctx${MAX_LENGTH}_gen${MAX_COMPLETION_LENGTH}"
VAULT_RUN="opsd_vault_t${TEMP_TAG}_steps${MAX_STEPS}_ctx${MAX_LENGTH}_gen${MAX_COMPLETION_LENGTH}_w${VAULT_WINDOW_SIZE}_s${SIGMA_TAG}"
FIXED_OUT="$OUTPUT_DIR/$FIXED_RUN"
VAULT_OUT="$OUTPUT_DIR/$VAULT_RUN"

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

prepare_assets() {
  log_msg "Preparing model and evaluation datasets."
  if [ ! -d "$MODEL_PATH" ]; then
    export HF_HUB_OFFLINE=0
    MODEL_PATH="$EXP_DIR/models/Qwen3-1.7B"
    export MODEL_PATH
    mkdir -p "$(dirname "$MODEL_PATH")"
    log_msg "Model path not found. Downloading $MODEL_REPO to $MODEL_PATH"
    "$PYTHON" - "$MODEL_REPO" "$MODEL_PATH" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo_id, local_dir = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
print(local_dir)
PY
    export HF_HUB_OFFLINE=1
  fi

  "$PYTHON" - "$MODEL_PATH" <<'PY'
import sys
from datasets import load_dataset
from transformers import AutoTokenizer

model_path = sys.argv[1]
AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
datasets = [
    ("siyanzhao/Openthoughts_math_30k_opsd", {}),
    ("HuggingFaceH4/aime_2024", {"split": "train"}),
    ("yentinglin/aime_2025", {"split": "train", "trust_remote_code": True}),
    ("MathArena/hmmt_feb_2025", {"split": "train", "trust_remote_code": True}),
]
for name, kwargs in datasets:
    ds = load_dataset(name, **kwargs)
    length = len(ds["train"]) if isinstance(ds, dict) and "train" in ds else len(ds)
    print(f"cached {name}: {length} rows")
PY
}

import_fixed_baseline() {
  if [ "$IMPORT_FIXED_BASELINE" != "1" ] || [ "$RUN_FIXED_OPSD" = "1" ]; then
    return 0
  fi
  log_msg "Importing fixed OPSD baseline eval JSONs from $BASELINE_EXP_DIR"
  for dataset in aime24 aime25 hmmt25; do
    mkdir -p "$EVAL_DIR/$dataset"
    local src="$BASELINE_EXP_DIR/eval_results/$dataset/opsd_fixed_t${TEMP_TAG}.json"
    local dst="$EVAL_DIR/$dataset/opsd_fixed_t${TEMP_TAG}.json"
    if [ ! -s "$src" ]; then
      echo "Missing fixed OPSD baseline result: $src" >&2
      exit 1
    fi
    cp "$src" "$dst"
  done
}

train_opsd() {
  local method="$1"
  local run_name="$2"
  local out_dir="$3"
  local log_file="$4"
  shift 4
  local extra_args=("$@")

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
  log_msg "Train config: steps=$MAX_STEPS, rollout_temp=$TRAIN_TEMPERATURE, loss_temp=$LOSS_TEMPERATURE, ctx=$MAX_LENGTH, gen=$MAX_COMPLETION_LENGTH, per_device_batch=$OPSD_TRAIN_BATCH_SIZE, grad_accum=$OPSD_GRAD_ACCUM"
  (
    cd "$OPSD_DIR"
    env CUDA_VISIBLE_DEVICES="$GPUS" "$ACCELERATE" launch \
      --config_file "$OPSD_DIR/accelerate.yaml" \
      --num_processes "$NUM_GPUS" \
      --gradient_accumulation_steps "$OPSD_GRAD_ACCUM" \
      --main_process_port "${MAIN_PROCESS_PORT:-19231}" \
      "$OPSD_DIR/opsd_train.py" \
      --model_name_or_path "$MODEL_PATH" \
      --learning_rate "$LEARNING_RATE" \
      --max_grad_norm 0.1 \
      --per_device_train_batch_size "$OPSD_TRAIN_BATCH_SIZE" \
      --gradient_checkpointing \
      --gradient_accumulation_steps "$OPSD_GRAD_ACCUM" \
      --output_dir "$OUTPUT_DIR" \
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
      --wandb_project "VAULT-Qwen3-1.7B-AIME-HMMT" \
      "${extra_args[@]}" \
      "${resume_args[@]}"
  ) > "$log_file" 2>&1
  log_msg "Finished $method training"
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
    log_msg "Skip eval $method on $dataset: $output_file already complete."
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
  if [ "$RUN_EVAL" != "1" ]; then
    return 0
  fi

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
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [ "$failed" -ne 0 ]; then
    echo "At least one evaluation failed for method $method. Check $LOG_DIR/eval_*_${method}.log" >&2
    exit 1
  fi

  "$PYTHON" "$EXP_DIR/summarize_results.py" --exp_dir "$EXP_DIR" > "$LOG_DIR/summary_after_${method}.log" 2>&1 || true
}

main() {
  log_msg "Experiment directory: $EXP_DIR"
  log_msg "OPSD source directory: $OPSD_DIR"
  log_msg "Model path: $MODEL_PATH"
  log_msg "GPUs: $GPUS"
  log_msg "Method: original fixed-temperature OPSD baseline vs VAULT bell-survival sequence weighting."
  log_msg "VAULT config: survival=$VAULT_SURVIVAL_MODE, sigma=$VAULT_BELL_SIGMA, clip=$VAULT_CLIP_MODE, kappa_update=$VAULT_KAPPA_UPDATE_MODE, validity=$VAULT_VALIDITY_MODE"

  prepare_assets
  import_fixed_baseline

  if [ "$RUN_FIXED_OPSD" = "1" ]; then
    MAIN_PROCESS_PORT="${FIXED_PORT:-19231}" train_opsd \
      "fixed-temperature OPSD" \
      "$FIXED_RUN" \
      "$FIXED_OUT" \
      "$LOG_DIR/train_opsd_fixed_t${TEMP_TAG}.log"
    eval_all "opsd_fixed_t${TEMP_TAG}" "$FIXED_OUT"
  fi

  if [ "$RUN_VAULT_OPSD" = "1" ]; then
    MAIN_PROCESS_PORT="${VAULT_PORT:-19232}" train_opsd \
      "VAULT" \
      "$VAULT_RUN" \
      "$VAULT_OUT" \
      "$LOG_DIR/train_opsd_vault_t${TEMP_TAG}.log" \
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
    eval_all "opsd_vault_t${TEMP_TAG}" "$VAULT_OUT"
  fi

  "$PYTHON" "$EXP_DIR/summarize_results.py" --exp_dir "$EXP_DIR" | tee "$LOG_DIR/final_summary.log"
  log_msg "All requested VAULT jobs finished."
}

main "$@"
