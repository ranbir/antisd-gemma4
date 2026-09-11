"""
Held-out evaluation for base / GRPO / AntiSD checkpoints on GSM8K.

Reports pass@1 accuracy (greedy by default), average generated tokens, average
thought-channel tokens, and the average number of deliberation markers
("wait", "let me check", ...) per trace. Writes a JSON summary plus every
completion so the blog's tables and traces can be filled from real outputs.

Usage:
  python eval_antisd.py --out outputs/eval_base.json
  python eval_antisd.py --adapter_dir outputs/grpo   --out outputs/eval_grpo.json
  python eval_antisd.py --adapter_dir outputs/antisd --out outputs/eval_antisd.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import List

import torch

from antisd_common import (
    compute_verifiable_reward,
    count_deliberation_markers,
    encode_prompt,
    extract_answer_number,
    load_model_and_tokenizer,
    pick_device,
    render_prompt,
    sample_rollouts_batched,
    split_thought_and_answer,
    terminator_ids,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="google/gemma-4-E2B-it")
    p.add_argument("--adapter_dir", default=None, help="LoRA adapter directory (omit for the base model)")
    p.add_argument("--dataset_name", default="openai/gsm8k")
    p.add_argument("--dataset_config", default="main")
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=200, help="number of held-out problems")
    p.add_argument("--k", type=int, default=1, help="samples per problem (k>1 reports avg@k with sampling)")
    p.add_argument("--greedy", action="store_true", default=None,
                   help="greedy decoding (default when k=1)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=64)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=16, help="problems generated at once (A100: 16-32, T4 4-bit: 4)")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default="outputs/eval.json")
    args = p.parse_args()
    greedy = (args.k == 1) if args.greedy is None else args.greedy

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    model, tokenizer = load_model_and_tokenizer(args.model_name, device, args.load_in_4bit, args.adapter_dir)
    stop_ids = terminator_ids(tokenizer, model)

    from datasets import load_dataset
    ds = load_dataset(args.dataset_name, args.dataset_config, split=args.split).shuffle(seed=args.seed)
    ds = ds.select(range(min(args.n, len(ds))))

    records: List[dict] = []
    t0 = time.time()
    rows = list(ds)
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start:start + args.batch_size]
        prompts = [encode_prompt(tokenizer, render_prompt(tokenizer, r["question"])) for r in chunk]
        per_prompt = sample_rollouts_batched(model, tokenizer, prompts, args.k, args.max_new_tokens, device,
                                             temperature=args.temperature, top_p=args.top_p,
                                             top_k=args.top_k, greedy=greedy)
        for offset, (row, gens) in enumerate(zip(chunk, per_prompt)):
            idx = start + offset
            problem, gold = row["question"], row["answer"]
            for j, g in enumerate(gens):
                completion = tokenizer.decode([t for t in g if t not in stop_ids], skip_special_tokens=False)
                thought, answer = split_thought_and_answer(completion)
                correct = compute_verifiable_reward(completion, gold)
                records.append({
                    "idx": idx, "sample": j, "problem": problem, "gold": gold,
                    "gold_number": extract_answer_number(gold),
                    "pred_number": extract_answer_number(answer if answer else completion),
                    "correct": correct,
                    "gen_tokens": len(g),
                    "thought_tokens": len(tokenizer(thought, add_special_tokens=False).input_ids) if thought else 0,
                    "deliberation_markers": count_deliberation_markers(thought),
                    "finished": bool(g) and g[-1] in stop_ids,
                    "completion": completion,
                })
        done = min(start + args.batch_size, len(rows))
        acc = sum(r["correct"] for r in records) / len(records)
        print(f"[{done}/{len(rows)}] running acc={acc:.3f}  ({(time.time()-t0)/60:.1f} min)")

    n = len(records)
    summary = {
        "model_name": args.model_name,
        "adapter_dir": args.adapter_dir,
        "split": args.split,
        "n_problems": len(ds),
        "k": args.k,
        "greedy": greedy,
        "accuracy": sum(r["correct"] for r in records) / n,
        "avg_gen_tokens": sum(r["gen_tokens"] for r in records) / n,
        "avg_thought_tokens": sum(r["thought_tokens"] for r in records) / n,
        "avg_deliberation_markers": sum(r["deliberation_markers"] for r in records) / n,
        "frac_traces_with_deliberation": sum(r["deliberation_markers"] > 0 for r in records) / n,
        "frac_unfinished": sum(not r["finished"] for r in records) / n,
        "elapsed_min": (time.time() - t0) / 60,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=1)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
