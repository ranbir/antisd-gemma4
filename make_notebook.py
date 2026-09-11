"""Generate antisd_gemma4_colab.ipynb.

The notebook deliberately contains no training logic of its own: it clones the
repo and drives train_antisd.py / eval_antisd.py / inspect_pmi.py, so the
Colab path and the CLI path can never disagree.

    python make_notebook.py
"""

import json

REPO_URL = "https://github.com/ranbir/antisd-gemma4.git"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# Anti-Self-Distillation (AntiSD) on Gemma 4, in one Colab session

This notebook reproduces the experiment behind the blog post *Teaching Gemma 4 to Hesitate*.
It runs three things on `google/gemma-4-E2B-it` with Gemma 4's native thinking mode enabled:

1. **Base model** evaluation on held-out GSM8K.
2. **GRPO baseline**: 50 steps of on-policy RL with the verifiable reward only (`--lambda_asd 0`).
3. **AntiSD**: the same 50 steps plus the per-token AntiSD advantage (`--lambda_asd 0.5`).

Then it evaluates both adapters, plots the training curves, and renders per-token PMI heatmaps.

All logic lives in the repo's Python files; this notebook only calls them.
Reference: Shen et al. (2026), [arXiv:2605.11609](https://arxiv.org/abs/2605.11609).

**Hardware.** A T4 (16 GB) works with `--load_in_4bit`. An L4 or A100 runs in bfloat16 and is much faster.
Expect the full pipeline (3 evals + 2 trainings) to take several hours on a T4; see the timing printed by each step.
"""),
    code("""
# 1. GPU check
import torch, subprocess
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(props.name, f"{props.total_memory/1e9:.1f} GB")
    FOURBIT = "--load_in_4bit" if props.total_memory < 20e9 else ""
else:
    FOURBIT = ""
print("4-bit flag:", repr(FOURBIT))
"""),
    code("""
# 2. Dependencies (Gemma 4 needs a current transformers).
#    Colab preinstalls an old torchao that current PEFT refuses to coexist with; remove it.
!pip install -q -U transformers peft datasets accelerate bitsandbytes
!pip uninstall -q -y torchao 2>/dev/null || true
"""),
    code("""
# 3. (Optional) Hugging Face login. Gemma 4 E2B is ungated Apache 2.0, so this is only needed
#    for the final upload cell. Store a token under Colab -> Secrets -> HF_TOKEN to use it.
from huggingface_hub import login
try:
    from google.colab import userdata
    login(token=userdata.get("HF_TOKEN"))
except Exception:
    print("No HF_TOKEN secret found; skipping login (downloads still work).")
"""),
    code(f"""
# 4. Get the code
import os
if not os.path.exists("antisd-gemma4"):
    !git clone {REPO_URL}
%cd antisd-gemma4
!git pull -q
!ls
"""),
    code("""
# 5. Experiment settings (edit here, nowhere else)
STEPS = 50          # training steps per run
GROUP = 4           # rollouts per prompt
LAMBDA = 0.5        # AntiSD weight (paper default)
MAX_NEW = 1024      # generation budget for thinking + answer
N_EVAL = 100        # held-out GSM8K test problems (200 in the blog; lower it for a quick pass)
SEED = 0
EVAL_BS = 4 if FOURBIT else 16   # problems generated at once during evaluation
"""),
    code("""
# 6. Evaluate the untouched base model
!python eval_antisd.py --n {N_EVAL} --max_new_tokens {MAX_NEW} --batch_size {EVAL_BS} {FOURBIT} --out outputs/eval_base.json
"""),
    code("""
# 7. GRPO baseline: identical pipeline, AntiSD term switched off
!python train_antisd.py --lambda_asd 0 --total_steps {STEPS} --group_size {GROUP} \\
    --max_new_tokens {MAX_NEW} --seed {SEED} {FOURBIT} --output_dir outputs/grpo
"""),
    code("""
# 8. AntiSD
!python train_antisd.py --lambda_asd {LAMBDA} --total_steps {STEPS} --group_size {GROUP} \\
    --max_new_tokens {MAX_NEW} --seed {SEED} {FOURBIT} --output_dir outputs/antisd
"""),
    code("""
# 9. Evaluate both adapters on the same held-out problems
!python eval_antisd.py --adapter_dir outputs/grpo   --n {N_EVAL} --max_new_tokens {MAX_NEW} --batch_size {EVAL_BS} {FOURBIT} --out outputs/eval_grpo.json
!python eval_antisd.py --adapter_dir outputs/antisd --n {N_EVAL} --max_new_tokens {MAX_NEW} --batch_size {EVAL_BS} {FOURBIT} --out outputs/eval_antisd.json
"""),
    code("""
# 10. Results table (this is what goes into the blog post)
import json, pandas as pd
rows = []
for name, path in [("Gemma 4 E2B (base)", "outputs/eval_base.json"),
                   ("+ GRPO, 50 steps", "outputs/eval_grpo.json"),
                   ("+ AntiSD, 50 steps", "outputs/eval_antisd.json")]:
    s = json.load(open(path))["summary"]
    rows.append({"method": name,
                 "pass@1 (%)": round(100 * s["accuracy"], 1),
                 "thought tokens": round(s["avg_thought_tokens"]),
                 "deliberation markers / trace": round(s["avg_deliberation_markers"], 2),
                 "traces with any marker (%)": round(100 * s["frac_traces_with_deliberation"], 1),
                 "unfinished (%)": round(100 * s["frac_unfinished"], 1)})
df = pd.DataFrame(rows).set_index("method")
display(df)
print(df.to_markdown())
"""),
    code("""
# 11. Training curves: GRPO vs AntiSD
import matplotlib.pyplot as plt
def load_metrics(path):
    return pd.DataFrame([json.loads(l) for l in open(path)])
m = {"GRPO": load_metrics("outputs/grpo/metrics.jsonl"), "AntiSD": load_metrics("outputs/antisd/metrics.jsonl")}
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for name, d in m.items():
    axes[0,0].plot(d.step, d.reward_mean.rolling(5, min_periods=1).mean(), label=name)
    axes[0,1].plot(d.step, d.avg_thought_tokens, label=name)
    axes[1,0].plot(d.step, d.avg_deliberation_markers, label=name)
    axes[1,1].plot(d.step, d.teacher_entropy_median, label=name)
axes[0,0].set_title("train reward (rolling mean of 5)"); axes[0,1].set_title("thought tokens per rollout")
axes[1,0].set_title("deliberation markers per rollout"); axes[1,1].set_title("median teacher entropy (gate input)")
for ax in axes.flat: ax.legend(); ax.set_xlabel("step")
plt.tight_layout(); os.makedirs("assets", exist_ok=True); plt.savefig("assets/training_curves.png", dpi=150); plt.show()
"""),
    code("""
# 12. Per-token PMI heatmaps on the same held-out problem, base vs AntiSD
IDX = 3
!python inspect_pmi.py --gsm8k_index {IDX} --greedy {FOURBIT} --html_out assets/pmi_base.html --json_out outputs/pmi_base.json
!python inspect_pmi.py --gsm8k_index {IDX} --greedy {FOURBIT} --adapter_dir outputs/antisd --html_out assets/pmi_antisd.html --json_out outputs/pmi_antisd.json
from IPython.display import HTML, display
display(HTML(open("assets/pmi_base.html").read()))
display(HTML(open("assets/pmi_antisd.html").read()))
"""),
    code("""
# 13. Before / after traces on one held-out problem (paste real ones into the blog)
base = json.load(open("outputs/eval_base.json"))["records"]
anti = json.load(open("outputs/eval_antisd.json"))["records"]
# pick the first problem the base model got wrong and AntiSD got right, else the first problem
cands = [b["idx"] for b, a in zip(base, anti) if b["correct"] == 0 and a["correct"] == 1]
i = cands[0] if cands else 0
print("PROBLEM:", base[i]["problem"], "\\nGOLD:", base[i]["gold_number"])
print("\\n===== BASE (correct=%s) =====\\n" % base[i]["correct"], base[i]["completion"])
print("\\n===== ANTISD (correct=%s) =====\\n" % anti[i]["correct"], anti[i]["completion"])
"""),
    code("""
# 14. (Optional) push the AntiSD adapter to the Hub
# from huggingface_hub import HfApi
# HfApi().upload_folder(folder_path="outputs/antisd", repo_id="YOUR_USERNAME/gemma-4-e2b-it-antisd", repo_type="model")
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
with open("antisd_gemma4_colab.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote antisd_gemma4_colab.ipynb with", len(cells), "cells")
