# Anti-Self-Distillation (AntiSD) on Gemma 4

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](antisd_gemma4_colab.ipynb)
[![arXiv](https://img.shields.io/badge/arXiv-2605.11609-b31b1b.svg)](https://arxiv.org/abs/2605.11609)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

An open-source, lightweight reproduction and adaptation of **Anti-Self-Distillation (AntiSD)** for Google DeepMind's **Gemma 4** (`google/gemma-4-e2b-it`), based on the paper:
> **"Anti-Self-Distillation for Reasoning RL via Pointwise Mutual Information"**  
> *Guobin Shen, Xiang Cheng, Chenxiao Zhao, Lei Huang, Jindong Li, Dongcheng Zhao, Xing Yu (2026)*  
> [arXiv:2605.11609](https://arxiv.org/abs/2605.11609)

---

## 💡 Overview

Reasoning models with `<think>` modes often suffer from **performative thinking**—they output textbook templates and confident connectives without genuinely deliberating or catching arithmetic errors.

Standard on-policy self-distillation (where the model mimics an oracle version of itself conditioned on the answer key) inadvertently suppresses deliberation tokens like `Wait`, `Let`, and `Maybe` because an oracle never needs to doubt.

**AntiSD** fixes this by:
1. **Reversing the gradient direction (Ascent):** Rewards the model when the oracle is unsure, directly incentivizing exploration.
2. **Jensen-Shannon Divergence (JSD) Softplus Capping:**

$$
A_t^{\text{AntiSD}} = -\frac{1}{2}\big(\text{softplus}(u_t) - \log 2\big)
$$

   Capping the deliberation reward at \( +\frac{1}{2}\log 2 \approx +0.3466 \) while preserving a proportional linear penalty for answer-template copying.
3. **Schmitt Trigger Entropy Gate:** Disables the term if teacher entropy collapses below \( \tau_{\text{down}} = 0.93 \cdot H_{\text{warm}} \).

---

## 📁 Repository Structure

* **[`HUGGINGFACE_BLOG_POST.md`](HUGGINGFACE_BLOG_POST.md):** Complete draft formatted with KaTeX for the Hugging Face Community Blog.
* **[`antisd_gemma4_colab.ipynb`](antisd_gemma4_colab.ipynb):** 1-Click Google Colab notebook for fine-tuning `google/gemma-4-e2b-it` in 50 steps on a single GPU.
* **[`train_antisd.py`](train_antisd.py):** Standalone PyTorch / Hugging Face training script with LoRA and verifiable reward logic.
* **[`inspect_pmi.py`](inspect_pmi.py):** Diagnostic CLI tool that generates colorized terminal traces and interactive HTML heatmaps of deliberation vs. shortcut tokens.

---

## 🚀 Quickstart

### 1. Run in Google Colab (Recommended)
Open `antisd_gemma4_colab.ipynb` directly in [Google Colab](https://colab.research.google.com). Fits on a standard GPU (T4 with 4-bit QLoRA, or L4/A100 with bfloat16).

### 2. Standalone Training via CLI
```bash
pip install -r requirements.txt

# Run 50 steps of AntiSD on GSM8K
python train_antisd.py \
  --model_name google/gemma-4-e2b-it \
  --dataset_name gsm8k \
  --total_steps 50 \
  --group_size 4 \
  --lambda_asd 0.1 \
  --output_dir ./gemma4_antisd_adapter
```

### 3. Inspecting Deliberation Tokens (PMI Visualizer)
```bash
python inspect_pmi.py \
  --model_name google/gemma-4-e2b-it \
  --problem "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell in total?" \
  --html_out pmi_trace.html
```

---

## 📚 Citation

```bibtex
@misc{shen2026antiselfdistillationreasoningrlpointwise,
      title={Anti-Self-Distillation for Reasoning RL via Pointwise Mutual Information}, 
      author={Guobin Shen and Xiang Cheng and Chenxiao Zhao and Lei Huang and Jindong Li and Dongcheng Zhao and Xing Yu},
      year={2026},
      eprint={2605.11609},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.11609}
}
```
