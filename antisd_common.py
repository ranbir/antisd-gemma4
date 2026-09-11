"""
Shared building blocks for Anti-Self-Distillation (AntiSD) on Gemma 4.

Everything that must be *identical* between training, evaluation, and the PMI
inspector lives here, so the three entry points cannot drift apart:

  * prompt construction through the model's own chat template, with Gemma 4's
    native thinking mode enabled (``<|think|>`` in the system turn, thoughts
    emitted inside ``<|channel>thought ... <channel|>``);
  * answer parsing and the verifiable 0/1 reward;
  * the AntiSD per-token advantage kernel;
  * token-aligned scoring of a rollout under the student and the self-teacher.

Reference: Shen et al. (2026), "Anti-Self-Distillation for Reasoning RL via
Pointwise Mutual Information", arXiv:2605.11609.
"""

from __future__ import annotations

import inspect
import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Gemma 4 thinking-mode markers (from the google/gemma-4-E2B-it model card)
# ----------------------------------------------------------------------------
THOUGHT_OPEN = "<|channel>thought"
THOUGHT_CLOSE = "<channel|>"

SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step and check "
    "your work. Finish with the final numeric answer on its own line in the "
    "form 'The answer is: <number>'."
)

PRIVILEGED_HEADER = "[Reference solution and grading, visible only to the teacher]"

# Words and phrases that mark self-checking / backtracking in a thought trace.
# Used only as a *diagnostic* metric, never as a training signal.
DELIBERATION_PATTERNS = [
    r"\bwait\b",
    r"\bhmm+\b",
    r"\bhold on\b",
    r"\bactually\b",
    r"\balternatively\b",
    r"\blet me (?:double[- ]?check|check|verify|re-?check|recompute|re-?examine|reconsider)\b",
    r"\bdouble[- ]?check\b",
    r"\bre-?check\b",
    r"\bmistake\b",
    r"\bthat(?:'s| is) (?:not right|wrong|incorrect)\b",
    r"\bon second thought\b",
]
_DELIBERATION_RE = re.compile("|".join(DELIBERATION_PATTERNS), re.IGNORECASE)


# ----------------------------------------------------------------------------
# 1. Prompt construction
# ----------------------------------------------------------------------------
def build_messages(problem: str, privileged_context: Optional[str] = None) -> List[dict]:
    """Chat messages for the student (no context) or self-teacher (with context c)."""
    user = f"Problem: {problem.strip()}"
    if privileged_context is not None:
        user += f"\n\n{PRIVILEGED_HEADER}\n{privileged_context.strip()}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def privileged_context(ground_truth_solution: str, is_correct: bool) -> str:
    """The privileged context c: verified solution + correctness feedback on the rollout."""
    verdict = "correct" if is_correct else "incorrect"
    return (
        f"Verified solution:\n{ground_truth_solution.strip()}\n"
        f"Grading of the attempt below: the final answer is {verdict}."
    )


def _manual_gemma4_template(messages: List[dict], enable_thinking: bool) -> str:
    """Fallback that mirrors Gemma 4's Jinja template for a system+user exchange.

    Only used when the tokenizer's template does not accept ``enable_thinking``
    (for example in the CPU smoke test with a stand-in model).
    """
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next(m["content"] for m in messages if m["role"] == "user")
    think = "<|think|>\n" if enable_thinking else ""
    return (
        f"<|turn>system\n{think}{system}<turn|>\n"
        f"<|turn>user\n{user}<turn|>\n"
        f"<|turn>model\n"
    )


def render_prompt(tokenizer, problem: str, privileged: Optional[str] = None,
                  enable_thinking: bool = True) -> str:
    """Render a generation-ready prompt string via the tokenizer's chat template."""
    messages = build_messages(problem, privileged)
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except Exception:  # template without enable_thinking, or no template at all
        return _manual_gemma4_template(messages, enable_thinking)


def encode_prompt(tokenizer, prompt_text: str) -> List[int]:
    """Tokenize a rendered prompt exactly once, with exactly one BOS if the model uses one."""
    ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None and (len(ids) == 0 or ids[0] != bos):
        ids = [bos] + ids
    return ids


# ----------------------------------------------------------------------------
# 2. Parsing generated text: thought channel, final answer, reward
# ----------------------------------------------------------------------------
def split_thought_and_answer(text: str) -> Tuple[str, str]:
    """Split a Gemma 4 completion into (thought, answer).

    Handles the native ``<|channel>thought ... <channel|>`` format and, as a
    courtesy for other models, ``<think> ... </think>``.
    """
    for open_tag, close_tag in ((THOUGHT_OPEN, THOUGHT_CLOSE), ("<think>", "</think>")):
        if open_tag in text:
            _, _, rest = text.partition(open_tag)
            if close_tag in rest:
                thought, _, answer = rest.partition(close_tag)
            else:  # ran out of tokens while still thinking
                thought, answer = rest, ""
            return thought.strip(), answer.strip()
    return "", text.strip()


_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def extract_answer_number(text: str) -> Optional[float]:
    """Extract the final numeric answer from a completion or a GSM8K target.

    Priority: GSM8K ``#### 42`` > ``\\boxed{42}`` > ``The answer is: 42`` >
    last number in the text. Commas inside numbers are stripped first.
    """
    clean = text.replace(",", "")
    candidates: List[str] = []
    if "####" in clean:
        candidates = re.findall(_NUM, clean.split("####")[-1])
    if not candidates:
        boxed = re.findall(r"\\boxed\{([^}]*)\}", clean)
        if boxed:
            candidates = re.findall(_NUM, boxed[-1])
    if not candidates:
        m = re.findall(r"(?:answer is|answer:|equals|result is)\s*[:=]?\s*\$?\s*(" + _NUM + ")",
                       clean, re.IGNORECASE)
        candidates = m
    if not candidates:
        candidates = re.findall(_NUM, clean)
    for c in reversed(candidates):
        try:
            return float(c)
        except ValueError:
            continue
    return None


def compute_verifiable_reward(completion: str, ground_truth: str) -> float:
    """1.0 if the completion's final answer matches the gold answer, else 0.0.

    Only the text *after* the thought channel is graded, so a number that
    appears mid-deliberation cannot be credited as the answer.
    """
    _, answer_part = split_thought_and_answer(completion)
    pred = extract_answer_number(answer_part if answer_part else completion)
    gold = extract_answer_number(ground_truth)
    if pred is None or gold is None:
        return 0.0
    return 1.0 if abs(pred - gold) < 1e-4 else 0.0


def count_deliberation_markers(thought: str) -> int:
    return len(_DELIBERATION_RE.findall(thought))


# ----------------------------------------------------------------------------
# 3. The AntiSD advantage kernel
# ----------------------------------------------------------------------------
def antisd_advantage(u_t: torch.Tensor) -> torch.Tensor:
    """Per-token AntiSD advantage  A_t = -phi(u_t),  phi(u) = 1/2 (softplus(u) - log 2).

    u_t = log pi_T(y_t | x, c, y_<t) - log pi_S(y_t | x, y_<t)  (= conditional PMI)

    * u_t -> -inf (teacher dislikes the token; deliberation):  A_t -> +1/2 log 2 ~ +0.347
    * u_t -> +inf (teacher loves the token; shortcut):         A_t ~ -1/2 u_t
    """
    return -0.5 * (F.softplus(u_t) - math.log(2.0))


# ----------------------------------------------------------------------------
# 4. Token-aligned scoring
# ----------------------------------------------------------------------------
@dataclass
class ScoredRollout:
    logp: torch.Tensor      # [gen_len] log-prob of each generated token
    entropy: torch.Tensor   # [gen_len] entropy of the next-token distribution at each position


def _logits_kwarg_name(model) -> Optional[str]:
    """Name of the 'keep only the last k logits' kwarg, if the model supports one."""
    try:
        params = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return None
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            return name
    return None


def score_rollout(model, prompt_ids: Sequence[int], gen_ids: Sequence[int],
                  device: torch.device, want_entropy: bool = False) -> ScoredRollout:
    """Score one rollout: log-prob (and optionally entropy) of every generated token.

    The sequence is ``prompt_ids + gen_ids`` with no padding, so the logits
    that predict generated token j sit at position ``len(prompt_ids) - 1 + j``.
    Because each rollout is scored on its own, the student sequence and the
    (longer) teacher sequence are aligned by *generated-token index*, never by
    raw position. This is what makes u_t = t_logp - s_logp well defined.
    """
    p_len, g_len = len(prompt_ids), len(gen_ids)
    ids = torch.tensor([list(prompt_ids) + list(gen_ids)], dtype=torch.long, device=device)
    attn = torch.ones_like(ids)

    kwargs = {"input_ids": ids, "attention_mask": attn, "use_cache": False}
    keep_name = _logits_kwarg_name(model)
    if keep_name is not None:
        kwargs[keep_name] = g_len + 1  # only materialise the tail we need (vocab is ~262k)

    logits = model(**kwargs).logits[0]  # [T or g_len+1, V]
    # The g_len positions that predict the generated tokens are the last g_len+1 rows minus the final one.
    tail = logits[-(g_len + 1):-1].float()  # [g_len, V]
    targets = torch.tensor(list(gen_ids), dtype=torch.long, device=device)

    logsumexp = torch.logsumexp(tail, dim=-1)
    logp = tail.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - logsumexp

    entropy = None
    if want_entropy:
        log_probs = tail - logsumexp.unsqueeze(-1)
        entropy = -(log_probs.exp() * log_probs).sum(-1)
    return ScoredRollout(logp=logp, entropy=entropy)


# ----------------------------------------------------------------------------
# 5. Model loading and LoRA
# ----------------------------------------------------------------------------
LORA_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj")
NON_TEXT_MARKERS = ("vision", "audio", "image", "embed_tokens", "lm_head")


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_tokenizer(model_name: str, device: torch.device, load_in_4bit: bool = False,
                             adapter_dir: Optional[str] = None):
    """Load tokenizer + model. Tries the causal-LM class, then Gemma 4's multimodal class.

    ``adapter_dir`` loads a trained LoRA adapter on top (inference only).
    """
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    kwargs = {"dtype": dtype} if _accepts_dtype_kwarg(transformers) else {"torch_dtype": dtype}
    if device.type == "cuda":
        kwargs["device_map"] = "auto"
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    model = None
    errors = []
    for cls_name in ("AutoModelForCausalLM", "AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(model_name, **kwargs)
            break
        except Exception as e:  # try the next class
            errors.append(f"{cls_name}: {type(e).__name__}: {e}")
    if model is None:
        raise RuntimeError("Could not load model with any known class:\n" + "\n".join(errors))
    if device.type != "cuda":
        model.to(device)

    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def _accepts_dtype_kwarg(transformers_module) -> bool:
    try:
        major = int(transformers_module.__version__.split(".")[0])
    except Exception:
        return False
    return major >= 5


def lora_target_module_names(model) -> List[str]:
    """Full names of the text-model attention projections (skips vision/audio towers)."""
    import torch.nn as nn
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) and "Linear" not in type(module).__name__:
            continue
        leaf = name.split(".")[-1]
        if leaf in LORA_SUFFIXES and not any(m in name for m in NON_TEXT_MARKERS):
            names.append(name)
    return names


def apply_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05,
               load_in_4bit: bool = False, gradient_checkpointing: bool = True):
    from peft import LoraConfig, get_peft_model
    if load_in_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    targets = lora_target_module_names(model)
    if not targets:
        raise RuntimeError("No q/k/v/o projection layers found for LoRA; inspect model.named_modules().")
    config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                        target_modules=targets, task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    return model


# ----------------------------------------------------------------------------
# 6. Generation helpers
# ----------------------------------------------------------------------------
def terminator_ids(tokenizer, model=None) -> List[int]:
    """All token ids that end a model turn (EOS, end-of-turn, etc.)."""
    ids: List[int] = []
    for src in (getattr(tokenizer, "eos_token_id", None),
                getattr(getattr(model, "generation_config", None), "eos_token_id", None)):
        if src is None:
            continue
        for x in (src if isinstance(src, (list, tuple)) else [src]):
            if x is not None and x not in ids:
                ids.append(int(x))
    for tok in ("<turn|>", "<end_of_turn>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0 and tid != getattr(tokenizer, "unk_token_id", -1) and tid not in ids:
            ids.append(tid)
    return ids


def trim_generation(gen_ids: Sequence[int], stop_ids: Iterable[int], pad_id: Optional[int]) -> List[int]:
    """Cut a generated id sequence at its first terminator (kept) and drop padding."""
    stop = set(stop_ids)
    out: List[int] = []
    for t in gen_ids:
        if pad_id is not None and t == pad_id and t not in stop:
            break
        out.append(int(t))
        if t in stop:
            break
    return out


@torch.no_grad()
def sample_rollouts(model, tokenizer, prompt_ids: Sequence[int], n: int, max_new_tokens: int,
                    device: torch.device, temperature: float = 1.0, top_p: float = 0.95,
                    top_k: int = 64, greedy: bool = False) -> List[List[int]]:
    """Sample ``n`` completions for one prompt, returned as trimmed token-id lists."""
    ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    attn = torch.ones_like(ids)
    stops = terminator_ids(tokenizer, model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (stops[0] if stops else None)
    gen_kwargs = dict(
        input_ids=ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        num_return_sequences=n,
        pad_token_id=pad_id,
        eos_token_id=stops if stops else None,
        use_cache=True,
    )
    if greedy:
        gen_kwargs.update(do_sample=False)
    else:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)
    out = model.generate(**gen_kwargs)
    p_len = ids.shape[1]
    return [trim_generation(row[p_len:].tolist(), stops, pad_id) for row in out]


@torch.no_grad()
def sample_rollouts_batched(model, tokenizer, prompts_ids: Sequence[Sequence[int]], n: int,
                            max_new_tokens: int, device: torch.device, temperature: float = 1.0,
                            top_p: float = 0.95, top_k: int = 64, greedy: bool = False
                            ) -> List[List[List[int]]]:
    """Generate for several prompts at once (left-padded). Returns, per prompt, ``n`` trimmed rollouts.

    Used by evaluation, where prompts differ. Left padding keeps every prompt's
    last token adjacent to the first generated token, and the attention mask
    tells the model to ignore the pads.
    """
    stops = terminator_ids(tokenizer, model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (stops[0] if stops else 0)
    max_len = max(len(p) for p in prompts_ids)
    ids = torch.full((len(prompts_ids), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros_like(ids)
    for i, p in enumerate(prompts_ids):
        ids[i, max_len - len(p):] = torch.tensor(list(p), dtype=torch.long)
        attn[i, max_len - len(p):] = 1
    ids, attn = ids.to(device), attn.to(device)
    gen_kwargs = dict(
        input_ids=ids, attention_mask=attn, max_new_tokens=max_new_tokens,
        num_return_sequences=n, pad_token_id=pad_id, eos_token_id=stops if stops else None,
        use_cache=True,
    )
    if greedy:
        gen_kwargs.update(do_sample=False)
    else:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)
    out = model.generate(**gen_kwargs)  # rows are grouped by prompt: [p0s0, p0s1, ..., p1s0, ...]
    gens = [trim_generation(row[max_len:].tolist(), stops, pad_id) for row in out]
    return [gens[i * n:(i + 1) * n] for i in range(len(prompts_ids))]
