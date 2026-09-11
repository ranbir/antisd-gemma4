"""
Pointwise Mutual Information (PMI) inspector for reasoning traces.

Samples one rollout from the student, then scores the *same tokens* under the
student (no context) and the self-teacher (verified solution + grading), and
prints / renders

    u_t = log pi_T(y_t | x, c, y_<t) - log pi_S(y_t | x, y_<t)

per token.  u_t << 0 (blue) are deliberation tokens the teacher dislikes and
AntiSD rewards; u_t >> 0 (red) are shortcut tokens the teacher loves and
AntiSD penalises.  Works for the base model or a trained adapter.

Usage:
  python inspect_pmi.py --gsm8k_index 3 --html_out assets/pmi_base.html
  python inspect_pmi.py --adapter_dir outputs/antisd --gsm8k_index 3 --html_out assets/pmi_antisd.html
"""

from __future__ import annotations

import argparse
import html
import json
import os
import random
from typing import List, Tuple

import torch

from antisd_common import (
    antisd_advantage,
    compute_verifiable_reward,
    encode_prompt,
    load_model_and_tokenizer,
    pick_device,
    privileged_context,
    render_prompt,
    sample_rollouts,
    score_rollout,
    terminator_ids,
)


def colorize_terminal(token: str, u: float) -> str:
    if u <= -2.0:
        return f"\033[1;36m{token}\033[0m"   # bold cyan: strong deliberation
    if u < -0.5:
        return f"\033[34m{token}\033[0m"     # blue
    if u >= 2.0:
        return f"\033[1;31m{token}\033[0m"   # bold red: strong shortcut
    if u > 0.5:
        return f"\033[33m{token}\033[0m"     # yellow
    return f"\033[90m{token}\033[0m"         # grey: neutral


def render_html(tokens_and_u: List[Tuple[str, float]], title: str, stats: dict, out_path: str) -> None:
    spans = []
    for tok, u in tokens_and_u:
        safe = html.escape(tok).replace("\n", "<br>")
        if u <= -1.0:
            a = min(1.0, abs(u) / 5.0) * 0.7 + 0.15
            style = f"background: rgba(0,150,255,{a:.2f}); color:#00294d;"
        elif u >= 1.0:
            a = min(1.0, u / 5.0) * 0.7 + 0.15
            style = f"background: rgba(255,60,60,{a:.2f}); color:#4d0000;"
        else:
            style = "color:#333;"
        spans.append(f'<span style="{style} padding:1px 2px; border-radius:3px;" '
                     f'title="u_t = {u:+.2f}">{safe}</span>')
    stat_rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in stats.items())
    doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
 body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 24px; background:#fafafa; }}
 .card {{ background:#fff; border:1px solid #e3e3e3; border-radius:8px; padding:24px; max-width:960px; margin:0 auto; }}
 .legend span.box {{ display:inline-block; width:14px; height:14px; border-radius:3px; vertical-align:middle; margin-right:6px; }}
 .legend {{ display:flex; gap:24px; font-size:14px; margin:12px 0 18px; }}
 .trace {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:14px; line-height:2.0; border-top:1px solid #eee; padding-top:14px; white-space:pre-wrap; word-wrap:break-word; }}
 table {{ font-size:13px; border-collapse:collapse; margin-top:16px; }} td {{ padding:2px 12px 2px 0; }}
</style></head><body><div class="card">
<h2>{html.escape(title)}</h2>
<div class="legend">
 <div><span class="box" style="background:rgba(0,150,255,.6)"></span><b>Deliberation</b> (u_t &lt;&lt; 0, rewarded by AntiSD)</div>
 <div><span class="box" style="background:rgba(255,60,60,.6)"></span><b>Shortcut</b> (u_t &gt;&gt; 0, penalised by AntiSD)</div>
 <div><span class="box" style="background:#e0e0e0"></span><b>Neutral</b> (u_t &asymp; 0)</div>
</div>
<div class="trace">{"".join(spans)}</div>
<table>{stat_rows}</table>
</div></body></html>"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="google/gemma-4-E2B-it")
    p.add_argument("--adapter_dir", default=None)
    p.add_argument("--problem", default=None)
    p.add_argument("--ground_truth", default=None)
    p.add_argument("--gsm8k_index", type=int, default=None, help="use this GSM8K test problem instead of --problem")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top_n", type=int, default=15, help="how many extreme tokens to list")
    p.add_argument("--html_out", default="outputs/pmi_trace.html")
    p.add_argument("--json_out", default=None, help="optional: dump (token, u_t) pairs")
    p.add_argument("--load_in_4bit", action="store_true")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()

    if args.gsm8k_index is not None:
        from datasets import load_dataset
        row = load_dataset("openai/gsm8k", "main", split="test")[args.gsm8k_index]
        problem, gold = row["question"], row["answer"]
    else:
        problem = args.problem or ("Natalia sold clips to 48 of her friends in April, and then she sold half "
                                   "as many clips in May. How many clips did Natalia sell altogether in April and May?")
        gold = args.ground_truth or "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips.\n#### 72"

    model, tokenizer = load_model_and_tokenizer(args.model_name, device, args.load_in_4bit, args.adapter_dir)
    stop_ids = terminator_ids(tokenizer, model)

    s_prompt_ids = encode_prompt(tokenizer, render_prompt(tokenizer, problem))
    gen = sample_rollouts(model, tokenizer, s_prompt_ids, 1, args.max_new_tokens, device, greedy=args.greedy)[0]
    completion = tokenizer.decode([t for t in gen if t not in stop_ids], skip_special_tokens=False)
    reward = compute_verifiable_reward(completion, gold)

    t_prompt_ids = encode_prompt(tokenizer, render_prompt(tokenizer, problem, privileged_context(gold, reward > 0.5)))
    with torch.no_grad():
        s = score_rollout(model, s_prompt_ids, gen, device)
        t = score_rollout(model, t_prompt_ids, gen, device)
    u = (t.logp - s.logp).cpu()
    adv = antisd_advantage(u)
    toks = [tokenizer.decode([tid]) for tid in gen]
    pairs = list(zip(toks, u.tolist()))

    print("\n" + "=" * 72)
    print(" PMI trace   cyan/blue = deliberation (u<<0)   red/yellow = shortcut (u>>0)")
    print("=" * 72)
    print("".join(colorize_terminal(tok, val) for tok, val in pairs))
    print("=" * 72)

    order = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
    print(f"\nMost NEGATIVE u_t (teacher dislikes; AntiSD rewards, capped at +{0.5*0.6931:.3f}):")
    for i in order[:args.top_n]:
        print(f"  {pairs[i][1]:+7.2f}  adv={adv[i].item():+.3f}  {pairs[i][0]!r}")
    print(f"\nMost POSITIVE u_t (teacher loves; AntiSD penalises linearly):")
    for i in order[::-1][:args.top_n]:
        print(f"  {pairs[i][1]:+7.2f}  adv={adv[i].item():+.3f}  {pairs[i][0]!r}")

    stats = {
        "adapter": args.adapter_dir or "(base model)",
        "answer correct": bool(reward),
        "generated tokens": len(gen),
        "mean u_t": f"{u.mean().item():+.3f}",
        "fraction u_t < 0": f"{(u < 0).float().mean().item():.2f}",
        "fraction u_t < -2": f"{(u < -2).float().mean().item():.2f}",
        "fraction u_t > +2": f"{(u > 2).float().mean().item():.2f}",
        "mean AntiSD advantage": f"{adv.mean().item():+.3f}",
    }
    print("\n" + json.dumps(stats, indent=2))
    render_html(pairs, "AntiSD: per-token PMI on a Gemma 4 reasoning trace", stats, args.html_out)
    print(f"Wrote {args.html_out}")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"problem": problem, "gold": gold, "completion": completion, "stats": stats,
                       "tokens": [{"token": tk, "u": val} for tk, val in pairs]}, f, indent=1)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
