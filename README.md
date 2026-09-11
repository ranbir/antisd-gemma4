# Anti-Self-Distillation (AntiSD) on Gemma 4

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ranbir/antisd-gemma4/blob/main/antisd_gemma4_colab.ipynb)
[![arXiv](https://img.shields.io/badge/arXiv-2605.11609-b31b1b.svg)](https://arxiv.org/abs/2605.11609)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A small, single-GPU reproduction of **Anti-Self-Distillation** (Shen et al., 2026) on
`google/gemma-4-E2B-it`, using Gemma 4's native thinking mode. It trains a LoRA adapter
for 50 steps on GSM8K, compares against a GRPO baseline that shares every line of the
pipeline, and ships the tooling to inspect *why* it works: a per-token PMI heatmap.

> **Status.** The code is complete and tested for correctness on CPU with a stand-in model.
> Numbers for Gemma 4 are produced by running the notebook; the blog post is filled in from
> those outputs and contains no hand-written results.

## The idea in three lines

Self-distillation scores a rollout under the *same* model given a cheat sheet (the verified
solution). The per-token log-ratio $u_t = \log \pi_T(y_t) - \log \pi_S(y_t)$ is the
conditional pointwise mutual information between the token and the cheat sheet. A teacher
that already knows the answer dislikes hesitation ("wait", "let me check"), so plain
self-distillation trains it away. AntiSD flips the sign and bounds it:

$$
A_t^{\text{AntiSD}} = -\tfrac{1}{2}\big(\operatorname{softplus}(u_t) - \log 2\big)
$$

Deliberation tokens ($u_t \ll 0$) get a bonus capped at $+\tfrac12\log 2$; shortcut tokens
($u_t \gg 0$) get a linear penalty. An entropy gate (Schmitt trigger at
$0.93\,H_{\text{warm}}$) switches the term off if the teacher's entropy collapses.

## Files

| File | What it does |
| --- | --- |
| `antisd_common.py` | Everything shared: Gemma 4 prompts via the chat template with `enable_thinking=True`, answer parsing, the 0/1 reward, the AntiSD kernel, token-aligned scoring, model/LoRA loading. |
| `train_antisd.py` | GRPO + AntiSD training loop. `--lambda_asd 0` is the GRPO baseline. Writes `metrics.jsonl`, `rollouts.jsonl`, and the adapter. |
| `eval_antisd.py` | Held-out GSM8K evaluation: pass@1, thought length, deliberation markers, per-example completions. |
| `inspect_pmi.py` | Samples a trace and colours every token by $u_t$. Terminal output plus a standalone HTML heatmap. |
| `antisd_gemma4_colab.ipynb` | Drives the three scripts end to end on a Colab GPU. Regenerate with `python make_notebook.py`. |
| `HUGGINGFACE_BLOG_POST.md` | The blog post draft. |

## Quickstart

### Colab (recommended)

Open the notebook badge above and run all cells. Gemma 4 E2B is an ungated Apache 2.0 model, so no
license click-through or token is needed to download it. A token (Colab secret `HF_TOKEN`) is only
needed if you want the last cell to upload your adapter to the Hub.

### CLI

```bash
pip install -r requirements.txt

# base model, held-out GSM8K test
python eval_antisd.py --n 200 --out outputs/eval_base.json

# GRPO baseline and AntiSD share the pipeline; only lambda differs
python train_antisd.py --lambda_asd 0   --output_dir outputs/grpo
python train_antisd.py --lambda_asd 0.5 --output_dir outputs/antisd

python eval_antisd.py --adapter_dir outputs/grpo   --n 200 --out outputs/eval_grpo.json
python eval_antisd.py --adapter_dir outputs/antisd --n 200 --out outputs/eval_antisd.json

# per-token PMI heatmap of one trace, before and after
python inspect_pmi.py --gsm8k_index 3 --greedy --html_out assets/pmi_base.html
python inspect_pmi.py --gsm8k_index 3 --greedy --adapter_dir outputs/antisd --html_out assets/pmi_antisd.html
```

Add `--load_in_4bit` to every command on a 16 GB GPU such as a Colab T4.

### Defaults

| Setting | Value | Note |
| --- | --- | --- |
| Model | `google/gemma-4-E2B-it` | thinking mode on via the chat template |
| LoRA | r=16, alpha=32, dropout 0.05 | q/k/v/o projections of the text model only |
| Steps / rollouts | 50 steps, G=4 | one GSM8K train problem per step, shuffled with the seed |
| Sampling | T=1.0, top-p 0.95, top-k 64 | Gemma 4's recommended settings |
| Generation budget | 1024 new tokens | thinking + answer |
| AntiSD weight | lambda = 0.5 | paper default |
| Gate | 5 warmup steps at lambda=0, tau_down = 0.93 H_warm | paper default |
| Optimiser | AdamW, lr 1e-5, cosine, grad-clip 1.0 | |
| Eval | 200 GSM8K test problems, greedy | `--k 4` for avg@4 with sampling |

## Correctness checks that run on CPU

The parts most likely to be silently wrong are covered by the smoke test:

```bash
python -m venv .venv && .venv/bin/pip install "numpy<2" torch transformers peft datasets accelerate
.venv/bin/python train_antisd.py --smoke_test --model_name hf-internal-testing/tiny-random-Gemma2ForCausalLM \
    --total_steps 8 --warmup_steps 3 --group_size 3 --max_new_tokens 24 --output_dir outputs/smoke
```

This exercises sampling, reward parsing, teacher/student scoring, the gate, and the update. The
scoring function is additionally checked against a brute-force log-softmax for both the short
student prompt and the longer teacher prompt, so $u_t$ is computed on the same generated
token in both.

## Citation

```bibtex
@misc{shen2026antisd,
  title  = {Anti-Self-Distillation for Reasoning RL via Pointwise Mutual Information},
  author = {Guobin Shen and Xiang Cheng and Chenxiao Zhao and Lei Huang and Jindong Li and Dongcheng Zhao and Xing Yu},
  year   = {2026},
  eprint = {2605.11609},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url    = {https://arxiv.org/abs/2605.11609}
}
```

## License

Apache 2.0 for this code. The `google/gemma-4-E2B-it` weights are also released under Apache 2.0,
so adapters trained on them can be shared under the same terms.
