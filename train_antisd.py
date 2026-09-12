"""
Anti-Self-Distillation (AntiSD) training for Gemma 4 with LoRA.

Per training step, on one GSM8K problem x:
  1. Sample G rollouts y ~ pi_S(. | x) with Gemma 4's native thinking mode on.
  2. Score each rollout with the verifiable 0/1 reward -> GRPO sequence advantage A_i^seq.
  3. Self-teacher pass (no grad): score the SAME tokens under pi_T(. | x, c), where c is the
     verified solution plus correctness feedback. Record per-token entropy for the gate.
  4. Student pass (grad): per-token log-probs under pi_S. u_t = log pi_T - log pi_S (= PMI).
  5. A_t = A_i^seq + lambda * gate * ( -1/2 (softplus(u_t) - log 2) ).
  6. REINFORCE update:  loss = - sum_t A_t log pi_S(y_t) / N_tokens.

Setting --lambda_asd 0 gives the plain GRPO baseline with the identical pipeline.

Reference: Shen et al. (2026), arXiv:2605.11609.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from typing import Dict, List, Optional

import torch
from transformers import get_cosine_schedule_with_warmup

from antisd_common import (
    antisd_advantage,
    apply_lora,
    compute_verifiable_reward,
    count_deliberation_markers,
    encode_prompt,
    load_model_and_tokenizer,
    pick_device,
    privileged_context,
    render_prompt,
    sample_rollouts_batched,
    score_rollout,
    split_thought_and_answer,
    terminator_ids,
)

SMOKE_PROBLEMS = [
    ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
     "How many clips did Natalia sell altogether in April and May?",
     "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether.\n#### 72"),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. "
     "How much did she earn?",
     "Weng earns 12/60 = $0.2 per minute.\nFor 50 minutes she earned 0.2 * 50 = $10.\n#### 10"),
    ("Betty is saving money for a new wallet which costs $100. Betty has only half of the money she "
     "needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much "
     "as her parents. How much more money does Betty need to buy the wallet?",
     "Betty has 100/2 = $50.\nGrandparents gave 15*2 = $30.\nShe needs 100 - 50 - 30 - 15 = $5 more.\n#### 5"),
]


class SchmittEntropyGate:
    """Auto-calibrated entropy gate from the paper.

    During the first ``warmup_steps`` (lambda = 0) it records the teacher entropy
    statistic H per step and sets H_warm = median of those. Afterwards the gate
    closes when H < tau_down = 0.93 * H_warm and only reopens once H >= H_warm,
    which stops it chattering when entropy hovers near the threshold.

    Degenerate case: if H_warm is already ~0 (Gemma 4 E2B's median per-token
    teacher entropy is below 1e-3 nats) there is no "collapse" left to detect
    and the thresholds would sit inside floating-point noise. The gate is then
    declared inert and stays open, which is the behaviour the paper's gate has
    on any model whose entropy never drops below its warmup level.
    """

    MIN_H_WARM = 1e-3  # nats

    def __init__(self, warmup_steps: int, down_factor: float = 0.93):
        self.warmup_steps = warmup_steps
        self.down_factor = down_factor
        self.warmup_entropies: List[float] = []
        self.h_warm: Optional[float] = None
        self.tau_down: Optional[float] = None
        self.is_open = True
        self.inert = False

    def update(self, step: int, teacher_entropy: float) -> Dict[str, object]:
        teacher_entropy = max(0.0, float(teacher_entropy))  # entropy is >= 0; kill fp noise
        if step < self.warmup_steps:
            self.warmup_entropies.append(teacher_entropy)
            return {"gate": "calibrating", "gate_open": False}
        if self.h_warm is None:
            self.h_warm = statistics.median(self.warmup_entropies) if self.warmup_entropies else teacher_entropy
            self.tau_down = self.down_factor * self.h_warm
            if self.h_warm < self.MIN_H_WARM:
                self.inert = True
                print(f"[gate] calibrated: H_warm={self.h_warm:.3g} < {self.MIN_H_WARM:g} nats -> "
                      f"gate is INERT (teacher entropy already ~0; AntiSD stays on)")
            else:
                print(f"[gate] calibrated: H_warm={self.h_warm:.4g}  tau_down={self.tau_down:.4g}")
        if self.inert:
            return {"gate": "inert", "gate_open": True}
        if self.is_open and teacher_entropy < self.tau_down:
            self.is_open = False
        elif not self.is_open and teacher_entropy >= self.h_warm:
            self.is_open = True
        return {"gate": "open" if self.is_open else "closed", "gate_open": self.is_open}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_problems(args) -> List[Dict[str, str]]:
    if args.smoke_test:
        return [{"question": q, "answer": a} for q, a in SMOKE_PROBLEMS]
    from datasets import load_dataset
    ds = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    ds = ds.shuffle(seed=args.seed)
    rows = []
    for row in ds.select(range(min(len(ds), args.total_steps))):
        rows.append({"question": row.get("question") or row.get("problem"),
                     "answer": row.get("answer") or row.get("solution")})
    return rows


def decode_completion(tokenizer, gen_ids: List[int], stop_ids: List[int]) -> str:
    """Decode keeping the thought-channel markers but dropping the end-of-turn token."""
    keep = [t for t in gen_ids if t not in stop_ids]
    return tokenizer.decode(keep, skip_special_tokens=False)


def train(args) -> None:
    set_seed(args.seed)
    device = pick_device()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Loading {args.model_name} on {device} (4-bit={args.load_in_4bit})")
    model, tokenizer = load_model_and_tokenizer(args.model_name, device, load_in_4bit=args.load_in_4bit)
    model = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha, load_in_4bit=args.load_in_4bit,
                       gradient_checkpointing=args.gradient_checkpointing and device.type == "cuda")
    model.print_trainable_parameters()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.lr_warmup_steps,
                                                num_training_steps=args.total_steps)
    gate = SchmittEntropyGate(args.warmup_steps)
    stop_ids = terminator_ids(tokenizer, model)
    problems = load_problems(args)

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    rollouts_path = os.path.join(args.output_dir, "rollouts.jsonl")
    metrics_f = open(metrics_path, "w")
    rollouts_f = open(rollouts_path, "w")

    print(f"Training {args.total_steps} steps, G={args.group_size}, B={args.problems_per_batch} problems/generate, "
          f"lambda={args.lambda_asd}, warmup={args.warmup_steps}, max_new_tokens={args.max_new_tokens}")
    t_start = time.time()
    buffer: List[tuple] = []  # (problem, gold, prompt_ids, gens) awaiting an update
    for step in range(args.total_steps):
        # ---- 1. sample G rollouts per problem, B problems per generate() call ---------
        # Decoding is memory-bound, so B prompts cost little more than one. Updates
        # still happen one problem at a time; problems 2..B in a batch are trained on
        # weights up to B-1 updates old (standard rollout batching, mild staleness).
        if not buffer:
            model.eval()
            batch_rows = [problems[(step + k) % len(problems)]
                          for k in range(min(args.problems_per_batch, args.total_steps - step))]
            prompt_ids = [encode_prompt(tokenizer, render_prompt(tokenizer, r["question"])) for r in batch_rows]
            per_prompt = sample_rollouts_batched(model, tokenizer, prompt_ids, args.group_size,
                                                 args.max_new_tokens, device, temperature=args.temperature,
                                                 top_p=args.top_p, top_k=args.top_k)
            buffer = [(r["question"], r["answer"], pid, gens)
                      for r, pid, gens in zip(batch_rows, prompt_ids, per_prompt)]
        problem, gold, s_prompt_ids, gens = buffer.pop(0)
        gens = [g for g in gens if len(g) > 0]
        if not gens:
            print(f"[step {step+1}] all rollouts empty, skipping")
            continue
        completions = [decode_completion(tokenizer, g, stop_ids) for g in gens]

        # ---- 2. verifiable reward -> GRPO sequence advantage ------------------------
        rewards = torch.tensor([compute_verifiable_reward(c, gold) for c in completions],
                               dtype=torch.float32, device=device)
        if len(gens) > 1 and rewards.std(unbiased=False) > 0:
            seq_adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
        else:
            seq_adv = torch.zeros_like(rewards)

        # ---- 3. self-teacher pass with privileged context (no grad) -----------------
        t_logps: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        with torch.no_grad():
            for g, r in zip(gens, rewards.tolist()):
                ctx = privileged_context(gold, is_correct=r > 0.5)
                t_prompt_ids = encode_prompt(tokenizer, render_prompt(tokenizer, problem, ctx))
                scored = score_rollout(model, t_prompt_ids, g, device, want_entropy=True)
                t_logps.append(scored.logp)
                entropies.append(scored.entropy)
        ent_all = torch.cat(entropies)
        teacher_entropy_median = ent_all.median().item()
        teacher_entropy_mean = ent_all.mean().item()
        teacher_entropy = teacher_entropy_mean if args.gate_stat == "mean" else teacher_entropy_median

        # ---- 4. gate + effective lambda ----------------------------------------------
        gate_info = gate.update(step, teacher_entropy)
        lam = args.lambda_asd if gate_info["gate_open"] else 0.0

        # ---- 5./6. student pass, AntiSD advantage, REINFORCE update -----------------
        model.train()
        optimizer.zero_grad(set_to_none=True)
        n_tokens = sum(len(g) for g in gens)
        total_loss = 0.0
        u_all: List[torch.Tensor] = []
        for i, g in enumerate(gens):
            scored = score_rollout(model, s_prompt_ids, g, device)
            s_logp = scored.logp
            u_t = (t_logps[i] - s_logp.detach())
            u_all.append(u_t)
            adv = seq_adv[i] + lam * antisd_advantage(u_t)
            loss = -(adv.detach() * s_logp).sum() / n_tokens
            loss.backward()
            total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        # ---- logging ------------------------------------------------------------------
        u_cat = torch.cat(u_all)
        thoughts = [split_thought_and_answer(c)[0] for c in completions]
        thought_tokens = [len(tokenizer(t, add_special_tokens=False).input_ids) for t in thoughts]
        delib = [count_deliberation_markers(t) for t in thoughts]
        finished = [int(g[-1] in stop_ids) for g in gens]
        record = {
            "step": step + 1,
            "loss": total_loss,
            "reward_mean": rewards.mean().item(),
            "teacher_entropy_median": teacher_entropy_median,
            "teacher_entropy_mean": teacher_entropy_mean,
            "gate_stat_value": teacher_entropy,
            "gate": gate_info["gate"],
            "lambda_eff": lam,
            "lr": scheduler.get_last_lr()[0],
            "avg_gen_tokens": n_tokens / len(gens),
            "avg_thought_tokens": sum(thought_tokens) / len(gens),
            "avg_deliberation_markers": sum(delib) / len(gens),
            "frac_finished": sum(finished) / len(gens),
            "u_mean": u_cat.mean().item(),
            "u_frac_negative": (u_cat < 0).float().mean().item(),
            "u_frac_below_-2": (u_cat < -2).float().mean().item(),
            "elapsed_s": time.time() - t_start,
        }
        metrics_f.write(json.dumps(record) + "\n")
        metrics_f.flush()
        for i, c in enumerate(completions):
            rollouts_f.write(json.dumps({"step": step + 1, "i": i, "reward": rewards[i].item(),
                                         "problem": problem, "completion": c}) + "\n")
        rollouts_f.flush()

        if args.save_every and (step + 1) % args.save_every == 0 and step + 1 < args.total_steps:
            ck = os.path.join(args.output_dir, f"checkpoint-{step+1}")
            model.save_pretrained(ck)
            print(f"[checkpoint] saved {ck}")

        if (step + 1) % args.log_every == 0 or step == 0:
            print(f"[step {step+1:3d}/{args.total_steps}] loss={total_loss:+.4f} "
                  f"reward={record['reward_mean']:.2f} H_T(med/mean)={teacher_entropy_median:.4g}/{teacher_entropy_mean:.4g} "
                  f"gate={gate_info['gate']} lam={lam:.2f} "
                  f"gen_tok={record['avg_gen_tokens']:.0f} think_tok={record['avg_thought_tokens']:.0f} "
                  f"delib={record['avg_deliberation_markers']:.2f} u_mean={record['u_mean']:+.3f} "
                  f"[{record['elapsed_s']/60:.1f} min]")

    metrics_f.close()
    rollouts_f.close()
    print(f"Saving LoRA adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Anti-Self-Distillation on Gemma 4")
    p.add_argument("--model_name", default="google/gemma-4-E2B-it")
    p.add_argument("--dataset_name", default="openai/gsm8k")
    p.add_argument("--dataset_config", default="main")
    p.add_argument("--dataset_split", default="train")
    p.add_argument("--output_dir", default="outputs/antisd")
    p.add_argument("--total_steps", type=int, default=50)
    p.add_argument("--warmup_steps", type=int, default=5, help="gate calibration steps at lambda=0 (paper: 5)")
    p.add_argument("--gate_stat", choices=["median", "mean"], default="median",
                   help="teacher-entropy statistic the gate watches (paper: median; mean is more informative when the median is ~0)")
    p.add_argument("--group_size", type=int, default=4, help="rollouts per prompt, G")
    p.add_argument("--problems_per_batch", type=int, default=4,
                   help="problems sampled per generate() call; updates stay one problem per step (1 = fully on-policy)")
    p.add_argument("--save_every", type=int, default=25, help="save an adapter checkpoint every N steps (0 = off)")
    p.add_argument("--lambda_asd", type=float, default=0.5, help="AntiSD mixing weight (paper: 0.5; 0 = GRPO baseline)")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr_warmup_steps", type=int, default=2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0, help="Gemma 4 recommended sampling: T=1.0")
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=64)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--load_in_4bit", action="store_true", help="QLoRA for 16 GB GPUs such as a Colab T4")
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--smoke_test", action="store_true",
                   help="use 3 built-in problems (no dataset download); pair with a tiny model to test the loop")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
