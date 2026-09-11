"""
Pointwise Mutual Information (PMI) Token Inspector for Reasoning Models
Visualizes Shortcut vs. Deliberation tokens along a reasoning rollout.

u_t = log pi_T(y_t) - log pi_S(y_t)
- u_t << 0: Deliberation tokens (Wait, Let, Maybe, backtracks) -> BLUE
- u_t >> 0: Shortcut tokens (formulas, therefore, answers)      -> RED
- u_t ~= 0: Neutral tokens                                      -> GRAY
"""

import sys
import math
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


def format_student_prompt(problem: str) -> str:
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


def format_teacher_prompt(problem: str, ground_truth: str, assessment: str = "correct") -> str:
    return (
        "<start_of_turn>user\n"
        "Solve the following math problem step-by-step. "
        "Show your intermediate thinking and reasoning inside <think> and </think> tags, "
        "and conclude with your final answer as 'The answer is: <number>'.\n\n"
        f"Problem: {problem}\n\n"
        f"[Privileged Context: Verified Reference Solution]\n"
        f"{ground_truth}\n"
        f"Previous assessment on this rollout attempt: Your answer is {assessment}.\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
        "<think>\n"
    )


def colorize_terminal(token: str, u_val: float) -> str:
    """Formats a token with ANSI color escape codes based on PMI value u_val."""
    # u_val < -1.5: Deep Blue / Cyan (Deliberation)
    # u_val > +1.5: Deep Red / Yellow (Shortcut)
    if u_val <= -2.0:
        return f"\033[1;36m{token}\033[0m"  # Bold Cyan
    elif u_val < -0.5:
        return f"\033[34m{token}\033[0m"    # Blue
    elif u_val >= 2.0:
        return f"\033[1;31m{token}\033[0m"  # Bold Red
    elif u_val > 0.5:
        return f"\033[33m{token}\033[0m"    # Yellow
    else:
        return f"\033[90m{token}\033[0m"    # Muted Gray


def generate_html_heatmap(tokens_and_pmi, output_html_path: str):
    """Generates a standalone, beautiful HTML visualization for blog posts."""
    html_spans = []
    for token, u_val in tokens_and_pmi:
        safe_token = token.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        if u_val <= -1.0:
            # Blue shade for deliberation
            intensity = min(1.0, abs(u_val) / 5.0)
            bg = f"rgba(0, 180, 255, {intensity * 0.7 + 0.15:.2f})"
            color = "#003366"
        elif u_val >= 1.0:
            # Red shade for shortcuts
            intensity = min(1.0, u_val / 5.0)
            bg = f"rgba(255, 60, 60, {intensity * 0.7 + 0.15:.2f})"
            color = "#660000"
        else:
            bg = "transparent"
            color = "#333333"

        html_spans.append(
            f'<span style="background-color: {bg}; color: {color}; padding: 1px 3px; '
            f'border-radius: 3px; margin: 1px; font-family: monospace;" '
            f'title="PMI u_t: {u_val:.3f}">{safe_token}</span>'
        )

    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>AntiSD Pointwise Mutual Information (PMI) Trace</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 30px; background: #fafafa; }}
        .card {{ background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 24px; max-width: 900px; margin: 0 auto; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }}
        .legend {{ display: flex; gap: 20px; margin-bottom: 20px; font-size: 14px; align-items: center; }}
        .box {{ width: 14px; height: 14px; border-radius: 3px; display: inline-block; vertical-align: middle; margin-right: 6px; }}
        .trace-box {{ line-height: 2.0; font-size: 15px; border-top: 1px solid #eee; padding-top: 16px; word-wrap: break-word; }}
    </style>
</head>
<body>
<div class="card">
    <h2>Anti-Self-Distillation: Pointwise Mutual Information (PMI) Token Trace</h2>
    <div class="legend">
        <div><span class="box" style="background: rgba(0, 180, 255, 0.6);"></span><b>Deliberation Tokens</b> (u_t &lt;&lt; 0, Rewarded by AntiSD)</div>
        <div><span class="box" style="background: rgba(255, 60, 60, 0.6);"></span><b>Shortcut Tokens</b> (u_t &gt;&gt; 0, Penalized by AntiSD)</div>
        <div><span class="box" style="background: #e0e0e0;"></span><b>Neutral Tokens</b> (u_t &asymp; 0)</div>
    </div>
    <div class="trace-box">
        {"".join(html_spans)}
    </div>
</div>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"\nSaved interactive HTML visualization to: {output_html_path}")


def main():
    parser = argparse.ArgumentParser(description="Inspect PMI tokens on reasoning traces")
    parser.add_argument("--model_name", type=str, default="google/gemma-4-e2b-it")
    parser.add_argument("--problem", type=str, default="Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell in total?")
    parser.add_argument("--ground_truth", type=str, default="Natalia sold 48/2 = 24 clips in May. In total she sold 48 + 24 = 72 clips. The answer is 72.")
    parser.add_argument("--html_out", type=str, default="pmi_trace_visualization.html")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    model.eval()

    s_prompt = format_student_prompt(args.problem)
    t_prompt = format_teacher_prompt(args.problem, args.ground_truth, assessment="correct")

    # 1. Sample Rollout from Student
    print("Generating student reasoning rollout...")
    s_enc = tokenizer(s_prompt, return_tensors="pt").to(device)
    prompt_len = s_enc.input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            input_ids=s_enc.input_ids,
            attention_mask=s_enc.attention_mask,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    gen_token_ids = out[0][prompt_len:]
    rollout_text = tokenizer.decode(gen_token_ids, skip_special_tokens=True)

    # 2. Re-evaluate student and teacher log-probs on the exact same rollout
    s_full_text = s_prompt + rollout_text
    t_full_text = t_prompt + rollout_text

    s_inputs = tokenizer(s_full_text, return_tensors="pt").to(device)
    t_inputs = tokenizer(t_full_text, return_tensors="pt").to(device)

    with torch.no_grad():
        s_logits = model(**s_inputs).logits[0]
        t_logits = model(**t_inputs).logits[0]

    s_logprobs = F.log_softmax(s_logits[:-1], dim=-1)
    t_logprobs = F.log_softmax(t_logits[:-1], dim=-1)

    s_labels = s_inputs.input_ids[0][1:]
    t_labels = t_inputs.input_ids[0][1:]

    s_token_logp = s_logprobs.gather(dim=-1, index=s_labels.unsqueeze(-1)).squeeze(-1)
    t_token_logp = t_logprobs.gather(dim=-1, index=t_labels.unsqueeze(-1)).squeeze(-1)

    # Align generated slice
    gen_len = len(gen_token_ids)
    s_gen_logp = s_token_logp[-gen_len:]
    t_gen_logp = t_token_logp[-gen_len:]

    u_t = (t_gen_logp - s_gen_logp).cpu().float().numpy()

    # 3. Print Colorized Trace to Terminal
    print("\n" + "=" * 60)
    print(" COLORIZED POINTWISE MUTUAL INFORMATION (PMI) TRACE")
    print(" CYAN/BLUE = Deliberation (u_t << 0) | RED/YELLOW = Shortcut (u_t >> 0)")
    print("=" * 60 + "\n<think>\n")

    token_strings = [tokenizer.decode([tid]) for tid in gen_token_ids]
    colored_text = []
    tokens_and_pmi = []

    for tok_str, u_val in zip(token_strings, u_t):
        colored_text.append(colorize_terminal(tok_str, float(u_val)))
        tokens_and_pmi.append((tok_str, float(u_val)))

    print("".join(colored_text))
    print("\n" + "=" * 60)

    # Generate HTML visualization
    generate_html_heatmap(tokens_and_pmi, args.html_out)


if __name__ == "__main__":
    main()
