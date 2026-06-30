#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def extract_boxed_answer(text):
    search_text = str(text)
    think_end = search_text.rfind("</think>")
    if think_end != -1:
        search_text = search_text[think_end + len("</think>") :]

    idx = search_text.rfind(r"\boxed")
    if idx < 0:
        return None

    start = search_text.find("{", idx)
    if start < 0:
        return None

    depth = 0
    for pos in range(start, len(search_text)):
        char = search_text[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return search_text[start + 1 : pos].strip()
    return None


def extract_last_number(text):
    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*(?:\.\d+)?)?", str(text))
    return matches[-1] if matches else None


def normalize_answer(answer):
    if answer is None:
        return None
    answer = str(answer).strip()
    answer = answer.replace("$", "").replace(",", "")
    answer = re.sub(r"\\text\{([^}]*)\}", r"\1", answer)
    answer = answer.replace("\\", "").replace(" ", "")
    number = re.search(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", answer)
    return number.group(0) if number else answer.lower()


def extract_gsm8k_answer(solution):
    match = re.search(r"####\s*(.+)\s*$", str(solution))
    if match:
        return match.group(1).strip().replace(",", "")
    return extract_last_number(solution)


def grade_answer(predicted, ground_truth):
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    return pred_norm is not None and gt_norm is not None and pred_norm == gt_norm


def summarize_problem_outputs(problem_id, problem, gt_answer, generations, val_n):
    predicted_answers = []
    is_correct_list = []
    is_formatted_list = []

    for generated_text in generations:
        predicted_answer = extract_boxed_answer(generated_text)
        is_formatted = predicted_answer is not None
        is_correct = grade_answer(predicted_answer, gt_answer)

        predicted_answers.append(predicted_answer if predicted_answer else "[No boxed answer found]")
        is_correct_list.append(is_correct)
        is_formatted_list.append(is_formatted)

    num_correct = sum(is_correct_list)
    num_formatted = sum(is_formatted_list)
    has_correct = any(is_correct_list)

    majority_vote_correct = False
    formatted_predictions = [pred for pred, fmt in zip(predicted_answers, is_formatted_list) if fmt]
    if formatted_predictions:
        most_common_answer = Counter(formatted_predictions).most_common(1)[0][0]
        majority_vote_correct = grade_answer(most_common_answer, gt_answer)

    return {
        "problem_id": problem_id,
        "problem": problem,
        "ground_truth": gt_answer,
        "val_n": val_n,
        "generations": [
            {"predicted_answer": pred, "full_generation": gen, "correct": corr, "formatted": fmt}
            for pred, gen, corr, fmt in zip(
                predicted_answers, generations, is_correct_list, is_formatted_list
            )
        ],
        "num_correct": num_correct,
        "num_formatted": num_formatted,
        "pass_at_n": has_correct,
        "majority_vote_correct": majority_vote_correct,
        "predicted_answer": predicted_answers[0] if predicted_answers else None,
        "full_generation": generations[0] if generations else "",
        "correct": is_correct_list[0] if is_correct_list else False,
        "formatted": is_formatted_list[0] if is_formatted_list else False,
    }


def load_gsm8k(dataset_path, split):
    dataset = load_from_disk(dataset_path)
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise ValueError(f"Dataset at {dataset_path!r} does not contain split {split!r}.")
        dataset = dataset[split]
    return dataset


def apply_chat_template(tokenizer, messages, enable_thinking):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_prompt(tokenizer, problem, enable_thinking):
    user_message = (
        f"{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    )
    return apply_chat_template(tokenizer, [{"role": "user", "content": user_message}], enable_thinking)


def load_llm(args):
    config = {
        "model": args.base_model,
        "trust_remote_code": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "distributed_executor_backend": "mp",
        "enforce_eager": True,
    }

    if args.checkpoint_dir:
        checkpoint = Path(args.checkpoint_dir)
        if (checkpoint / "adapter_model.safetensors").exists() or (checkpoint / "adapter_model.bin").exists():
            config.update(
                {
                    "enable_lora": True,
                    "max_lora_rank": args.max_lora_rank,
                    "max_loras": 1,
                    "max_cpu_loras": 1,
                }
            )
        else:
            raise FileNotFoundError(f"No LoRA adapter weights found in {checkpoint}")

    llm = LLM(**config)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    return llm, tokenizer


def evaluate(args):
    dataset = load_gsm8k(args.dataset_path, args.split)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    llm, tokenizer = load_llm(args)

    from vllm.lora.request import LoRARequest

    lora_request = None
    if args.checkpoint_dir:
        lora_request = LoRARequest("gsm8k_lora", 1, args.checkpoint_dir)

    prompts = []
    records = []
    for idx, example in enumerate(dataset):
        problem = example["problem"] if "problem" in example else example["question"]
        solution = example["solution"] if "solution" in example else example["answer"]
        gt_answer = extract_gsm8k_answer(solution)
        prompts.append(build_prompt(tokenizer, problem, args.enable_thinking))
        records.append({"problem_id": idx, "problem": problem, "ground_truth": gt_answer})

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        n=args.val_n,
    )
    outputs = llm.generate(prompts, sampling, lora_request=lora_request, use_tqdm=True)

    results = []
    pass_at_n = 0
    total_correct = 0
    formatted_count = 0
    total_solutions = 0

    for record, output in zip(records, outputs):
        generations = [item.text for item in output.outputs]
        result = summarize_problem_outputs(
            problem_id=record["problem_id"],
            problem=record["problem"],
            gt_answer=record["ground_truth"],
            generations=generations,
            val_n=args.val_n,
        )
        results.append(result)
        pass_at_n += int(result["pass_at_n"])
        total_correct += result["num_correct"]
        formatted_count += result["num_formatted"]
        total_solutions += args.val_n

    num_problems = len(results)
    majority_vote_count = sum(1 for result in results if result["majority_vote_correct"])
    summary = {
        "run_label": args.run_label,
        "base_model": args.base_model,
        "checkpoint_dir": args.checkpoint_dir,
        "dataset": "gsm8k",
        "dataset_path": args.dataset_path,
        "split": args.split,
        "enable_thinking": args.enable_thinking,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "val_n": args.val_n,
        "num_problems": num_problems,
        "total_solutions": total_solutions,
        "pass_at_n": pass_at_n,
        "pass_at_n_pct": pass_at_n / num_problems * 100 if num_problems else 0.0,
        "average_at_n": total_correct,
        "average_at_n_pct": total_correct / total_solutions * 100 if total_solutions else 0.0,
        "majority_vote_at_n": majority_vote_count,
        "majority_vote_at_n_pct": majority_vote_count / num_problems * 100 if num_problems else 0.0,
        "formatted_count": formatted_count,
        "format_rate": formatted_count / total_solutions * 100 if total_solutions else 0.0,
        "results": results,
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.summary_csv:
        csv_path = Path(args.summary_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        exists = csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            fieldnames = [
                "run_label",
                "checkpoint_dir",
                "num_problems",
                "average_at_n_pct",
                "pass_at_n_pct",
                "majority_vote_at_n_pct",
                "format_rate",
                "temperature",
                "top_p",
                "top_k",
                "max_new_tokens",
                "val_n",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow({key: summary.get(key) for key in fieldnames})

    printable = {key: value for key, value in summary.items() if key != "results"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen LoRA adapters on GSM8K with val-n metrics.")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--dataset_path", default="data/gsm8k_opsd")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--summary_csv", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.55)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_lora_rank", type=int, default=64)
    parser.add_argument("--val_n", type=int, default=4)
    parser.add_argument("--enable_thinking", action="store_true", default=False)
    parser.add_argument("--no_thinking", action="store_false", dest="enable_thinking")
    parser.add_argument("--run_label", default=None)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
