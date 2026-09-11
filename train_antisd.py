"""
Anti-Self-Distillation (AntiSD) Training Script for Gemma 4
Based on: "Anti-Self-Distillation for Reasoning RL via Pointwise Mutual Information" (Shen et al., 2026)

Key Components:
1. Student Policy (pi_S): Evaluates rollout y given prompt x.
2. Self-Teacher Policy (pi_T): Evaluates same rollout y given prompt x + privileged context c.
3. JSD-derived Softplus Advantage: A_t^{AntiSD} = -0.5 * (softplus(u_t) - log(2)), where u_t = log pi_T - log pi_S.
4. Auto-Calibrated Entropy Gate: Disables AntiSD if median teacher entropy collapses below tau_down = 0.93 * H_warm.
"""

import os
import re
import math
import argparse
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset


# ==========================================
# 1. Answer Parsing & Verifiable Reward
# ==========================================

def extract_answer_number(text: str) -> Optional[float]:
    """
    Extracts numerical answer from model output or GSM8K target.
    Handles '#### 42', 'The answer is 42', boxed answers, or trailing floats.
    """
    # 1. Check GSM8K explicit delimiter
    if "####" in text:
        ans_part = text.split("####")[-1].strip().replace(",", "")
        m = re.search(r"[-+]?\d*\.?\d+", ans_part)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                pass

    # 2. Check LaTeX \boxed{...}
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        m = re.search(r"[-+]?\d*\.?\d+", boxed[-1].replace(",", ""))
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                pass

    # 3. Check common verbal templates
    m = re.findall(r"(?:answer is|equals|result is)\s*[:=]?\s*(\$?\s*[-+]?\d*\.?\d+)", text, re.IGNORECASE)
    if m:
        cleaned = m[-1].replace("$", "").strip().replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            pass

    # 4. Fallback to last numerical token in string
    nums = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            pass

    return None


def compute_verifiable_reward(generated_text: str, ground_truth_text: str) -> float:
    """Returns 1.0 if answers match numerically within 1e-4 tolerance, else 0.0."""
    pred = extract_answer_number(generated_text)
    gold = extract_answer_number(ground_truth_text)
    if pred is not None and gold is not None:
        return 1.0 if abs(pred - gold) < 1e-4 else 0.0
    return 0.0


# ==========================================
# 2. AntiSD Math & Advantage Kernels
# ==========================================

def compute_token_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Gathers per-token log-probabilities for each token in input_ids.
    logits: [B, L, V]
    input_ids: [B, L]
    returns: [B, L - 1] matching shifted token positions.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logp = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_logp


def compute_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Computes per-token Shannon entropy H[P] = - sum P * log P in nats.
    logits: [B, L, V]
    returns: [B, L]
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def compute_antisd_token_advantage(
    student_logp: torch.Tensor,
    teacher_logp: torch.Tensor,
    gate_active: bool = True
) -> torch.Tensor:
    """
    Computes per-token Anti-Self-Distillation advantage via JSD ascent:
    u_t = t_t - s_t  (conditional Pointwise Mutual Information)
    A_t^{AntiSD} = -phi(u_t) = -0.5 * (softplus(u_t) - log(2))

    Properties:
    - For deliberation tokens (u_t < 0): bounded positive bonus <= 0.5 * log(2) ~ +0.3466.
    - For shortcut tokens (u_t > 0): proportional linear penalty ~ -0.5 * u_t.
    """
    u_t = teacher_logp - student_logp
    # Softplus is strictly monotonic and numerically stable
    antisd_adv = -0.5 * (F.softplus(u_t) - math.log(2.0))

    if not gate_active:
        antisd_adv = torch.zeros_like(antisd_adv)

    return antisd_adv


# ==========================================
# 3. Prompt Formatting for Gemma 4
# ==========================================

def format_student_prompt(problem: str) -> str:
    """Builds user prompt asking Gemma 4 to think inside <think> tags."""
    return (
        "<start_of_turn>user\n"
        "Solve the following math problem step-by-step. "
        "Show your intermediate thinking and reasoning inside <think> and </think> tags, "
        "and conclude with your final answer as 'The answer is: <number>'.\n\n"
        f"Problem: {problem}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
        "<think>\n"
    )


def format_teacher_prompt(problem: str, ground_truth: str, is_correct: bool) -> str:
    """
    Builds privileged teacher context c:
    Teacher sees the verified reference solution and correctness indicator.
    """
    assessment = "Your answer is correct." if is_correct else "Your answer is incorrect."
    return (
        "<start_of_turn>user\n"
        "Solve the following math problem step-by-step. "
        "Show your intermediate thinking and reasoning inside <think> and </think> tags, "
        "and conclude with your final answer as 'The answer is: <number>'.\n\n"
        f"Problem: {problem}\n\n"
        f"[Privileged Context: Verified Reference Solution]\n"
        f"{ground_truth}\n"
        f"Previous assessment on this rollout attempt: {assessment}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
        "<think>\n"
    )


# ==========================================
# 4. AntiSD Trainer Loop
# ==========================================

class AntiSDTrainer:
    def __init__(
        self,
        model_name: str = "google/gemma-4-e2b-it",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_4bit: bool = False,
        lr: float = 5e-6,
        lambda_asd: float = 0.1,
        warmup_steps: int = 5,
        total_steps: int = 50,
        group_size: int = 4,
        max_new_tokens: int = 512,
        output_dir: str = "./gemma4_antisd_output"
    ):
        self.device = device
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.lambda_asd = lambda_asd
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        print(f"Loading tokenizer & model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model Loading with optional QLoRA
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )

        # Apply LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # Entropy Gate State (Schmitt Trigger)
        self.h_warm: Optional[float] = None
        self.tau_down: Optional[float] = None
        self.gate_is_open: bool = True
        self.warmup_entropies: List[float] = []

    def sample_rollouts(self, prompt_text: str) -> Tuple[List[str], List[str]]:
        """Samples G candidate rollouts for a given prompt."""
        self.model.eval()
        enc = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        prompt_len = enc.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=self.group_size,
                pad_token_id=self.tokenizer.pad_token_id
            )

        full_texts = []
        gen_texts = []
        for i in range(self.group_size):
            full = self.tokenizer.decode(outputs[i], skip_special_tokens=False)
            gen = self.tokenizer.decode(outputs[i][prompt_len:], skip_special_tokens=True)
            full_texts.append(full)
            gen_texts.append(gen)

        return full_texts, gen_texts

    def evaluate_sequence_logprobs(
        self,
        prompts: List[str],
        rollouts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Tokenizes (prompt + rollout), computes forward pass, and returns:
        - token_logp: [B, max_len]
        - mask: [B, max_len] (1 for generated tokens, 0 for prompt/pad)
        - logits: [B, max_len, V]
        - median_entropy: scalar median entropy across generated tokens
        """
        combined_texts = [p + r for p, r in zip(prompts, rollouts)]
        enc = self.tokenizer(
            combined_texts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        ).to(self.device)

        input_ids = enc.input_ids
        attention_mask = enc.attention_mask

        # Forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Log-probs for all tokens
        token_logp = compute_token_logprobs(logits, input_ids)

        # Create mask that isolates only generated tokens (excluding prompt and padding)
        gen_mask = torch.zeros_like(token_logp)
        entropies = []

        for b in range(len(prompts)):
            p_len = len(self.tokenizer(prompts[b]).input_ids)
            total_len = attention_mask[b].sum().item()
            # gen tokens start at p_len - 1 in the shifted sequence
            start_idx = max(0, p_len - 1)
            end_idx = max(start_idx, total_len - 1)
            gen_mask[b, start_idx:end_idx] = 1.0

            # compute entropy on generated slice
            token_entropy = compute_entropy_from_logits(logits[b, start_idx:end_idx, :])
            entropies.append(token_entropy.detach())

        all_entropies = torch.cat(entropies) if entropies else torch.tensor([1.0], device=self.device)
        median_h = torch.median(all_entropies).item()

        return token_logp, gen_mask, logits, median_h

    def update_step(self, problem: str, ground_truth: str, step_idx: int) -> Dict[str, float]:
        """Executes a single AntiSD training step on 1 prompt with G rollouts."""
        self.model.train()

        # 1. Build Student Prompt and Sample G Rollouts
        s_prompt = format_student_prompt(problem)
        full_texts, gen_texts = self.sample_rollouts(s_prompt)

        # 2. Compute Verifiable Trajectory Rewards (Scalar bit)
        rewards = [compute_verifiable_reward(gen, ground_truth) for gen in gen_texts]
        r_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        r_mean = r_tensor.mean().item()
        r_std = r_tensor.std(unbiased=False).item() + 1e-6

        # Standard GRPO normalized advantage: (R - mean) / std
        seq_adv = (r_tensor - r_mean) / r_std

        # 3. Student Forward Pass (pi_S)
        s_prompts = [s_prompt] * self.group_size
        s_logp, s_mask, s_logits, _ = self.evaluate_sequence_logprobs(s_prompts, gen_texts)

        # 4. Teacher Forward Pass (pi_T with Privileged Context c)
        # Teacher context depends on whether rollout was correct
        t_prompts = [
            format_teacher_prompt(problem, ground_truth, r > 0.5)
            for r in rewards
        ]

        with torch.no_grad():
            # Stop gradient on self-teacher
            t_logp, t_mask, t_logits, teacher_median_h = self.evaluate_sequence_logprobs(t_prompts, gen_texts)

        # Ensure masks match
        active_mask = s_mask * t_mask

        # 5. Entropy Gate Calibration & Schmitt Trigger
        if step_idx < self.warmup_steps:
            self.warmup_entropies.append(teacher_median_h)
            current_lambda = 0.0  # Warmup at lambda = 0
            gate_status = "calibrating"
        else:
            if self.h_warm is None:
                self.h_warm = float(torch.median(torch.tensor(self.warmup_entropies)).item())
                self.tau_down = 0.93 * self.h_warm
                print(f"\n[Gate Calibrated] H_warm: {self.h_warm:.4f}, tau_down: {self.tau_down:.4f}\n")

            # Schmitt Trigger Logic
            if self.gate_is_open and teacher_median_h < self.tau_down:
                self.gate_is_open = False  # Teacher entropy collapsed
            elif not self.gate_is_open and teacher_median_h >= self.h_warm:
                self.gate_is_open = True   # Teacher recovered

            current_lambda = self.lambda_asd if self.gate_is_open else 0.0
            gate_status = "open" if self.gate_is_open else "closed"

        # 6. Compute AntiSD Advantage: -0.5 * (softplus(u_t) - log(2))
        antisd_adv = compute_antisd_token_advantage(
            student_logp=s_logp,
            teacher_logp=t_logp,
            gate_active=(current_lambda > 0)
        )

        # 7. Total Combined Advantage: A_i^{seq} + lambda * A_t^{AntiSD}
        total_adv = seq_adv.unsqueeze(1) + (current_lambda * antisd_adv)

        # 8. Policy Gradient Loss
        # - sum( A_t * log pi_S(y_t) ) over active tokens
        policy_loss = -(total_adv.detach() * s_logp * active_mask).sum() / (active_mask.sum() + 1e-8)

        # Backward & Step
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        return {
            "loss": policy_loss.item(),
            "reward_mean": r_mean,
            "teacher_entropy": teacher_median_h,
            "gate_status": gate_status,
            "lambda": current_lambda,
            "avg_tokens": float(active_mask.sum().item() / self.group_size)
        }


# ==========================================
# 5. Main Execution Entry Point
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Train Gemma 4 with Anti-Self-Distillation")
    parser.add_argument("--model_name", type=str, default="google/gemma-4-e2b-it", help="Model name or path")
    parser.add_argument("--dataset_name", type=str, default="gsm8k", help="Dataset name")
    parser.add_argument("--total_steps", type=int, default=50, help="Total training steps")
    parser.add_argument("--group_size", type=int, default=4, help="Rollouts per prompt (G)")
    parser.add_argument("--lambda_asd", type=float, default=0.1, help="AntiSD mixing coefficient")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit QLoRA for lower memory")
    parser.add_argument("--output_dir", type=str, default="./antisd_gemma4_checkpoints", help="Save dir")
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f" Anti-Self-Distillation (AntiSD) on Gemma 4")
    print(f" Model: {args.model_name} | Dataset: {args.dataset_name}")
    print(f" Total Steps: {args.total_steps} | Rollouts per prompt: {args.group_size}")
    print(f"=======================================================\n")

    # Load Dataset
    print(f"Loading dataset: {args.dataset_name}...")
    dataset = load_dataset(args.dataset_name, "main" if args.dataset_name == "gsm8k" else None, split="train")

    trainer = AntiSDTrainer(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        lambda_asd=args.lambda_asd,
        total_steps=args.total_steps,
        group_size=args.group_size,
        output_dir=args.output_dir
    )

    for step in range(args.total_steps):
        sample = dataset[step % len(dataset)]
        problem = sample.get("question") or sample.get("problem")
        solution = sample.get("answer") or sample.get("solution")

        metrics = trainer.update_step(problem=problem, ground_truth=solution, step_idx=step)

        if (step + 1) % 5 == 0 or step == 0:
            print(
                f"[Step {step+1:02d}/{args.total_steps}] "
                f"Loss: {metrics['loss']:.4f} | "
                f"Reward: {metrics['reward_mean']:.2f} | "
                f"Teacher H: {metrics['teacher_entropy']:.3f} | "
                f"Gate: {metrics['gate_status']} (λ={metrics['lambda']:.2f}) | "
                f"Tokens/trace: {metrics['avg_tokens']:.0f}"
            )

    print("\nTraining complete! Saving LoRA adapter...")
    trainer.model.save_pretrained(args.output_dir)
    trainer.tokenizer.save_pretrained(args.output_dir)
    print(f"Saved checkpoint to: {args.output_dir}")


if __name__ == "__main__":
    main()
