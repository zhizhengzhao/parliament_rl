#!/usr/bin/env python3
"""Offline RL trainer for Parliament-collected data.

Algorithm: token-level PPO surrogate with KL anchor to the frozen base
(= GRPO / DAPO family; the `π_old` for the ratio is recomputed on the
training side at the start of each iter, not grabbed from the vLLM
rollout).  The loss is:

    ratio_t = exp(logπ_θ − logπ_old)                         ← trust region
    pg      = − mean_token( min(                              ← clip surrogate
                 ratio_t · A_t,
                 clip(ratio_t, 1 − ε_lo, 1 + ε_hi) · A_t ))
    kl      = + mean_token( exp(log r) − log r − 1 )          ← k3 estimator
                 with log r = logπ_θ − logπ_ref                ← vs frozen base
    L       = pg + β_kl · kl

Defaults follow the 2025 DAPO-adjacent middle ground:
    ε_lo = 0.2    (standard; keeps low-probability tokens alive)
    ε_hi = 0.25   (asymmetric, mildly relaxed; DAPO uses 0.28)
    β_kl = 0.005  (between DeepSeek-R1-final 0.001 and GRPO 0.04)

Why recompute `π_old` on the training side instead of saving logprobs
from vLLM rollouts?  `rl/extract.py` re-renders every prompt from
`parliament.db` (anonymized names, augmented wrapper templates, vote
events injected from filtered cell-aware vote pools, etc.).  The
training prompt-token sequence therefore differs from what the
rollout actually saw, so any ratio against the rollout's old log-prob
would be against the wrong distribution.  Instead we pre-forward the
policy once at the start of the iter, freeze that as `π_old`, and use
it across all `ppo_epochs` passes.  This gives a clean trust region
for the multi-epoch SGD while keeping the rollout-side code unchanged.

Why token-level aggregation?  `sum / mask.sum()` over the whole batch
(DAPO-style) gives every response token equal weight.  Per-sequence
mean-then-batch-mean instead biases gradients toward short responses.

Naming (verl-aligned, see docs/00_naming.md):
    --ppo-epochs N        Passes through the same train.jsonl per iter
                          (`--num-epochs` kept as deprecated alias).
    --use-ppo-clip        Enable the PPO ratio+clip surrogate (default on;
                          --no-use-ppo-clip degenerates to vanilla RWR).
    --clip-ratio-low/high PPO clip bounds = [1 − low, 1 + high].

    Outer loops (`iter`, `total_epochs`) are owned by `scripts/iterate.py`;
    this script trains one sampling round's data once, repeated
    `ppo_epochs` times under a fresh π_old.

Usage (single-process smoke):
    python -m rl.train --data data/<run>/train.jsonl --output ckpts/run1

Multi-GPU DDP+LoRA via accelerate:
    accelerate launch --config_file rl/accelerate_ddp.yaml -m rl.train \
        --data data/<run>/train.jsonl --output ckpts/run1

Resume:
    python -m rl.train ... --resume ckpts/run1/step_200
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

# torch.compile + FSDP init on hybrid-attention models can block NCCL
# collectives for 10+ min on first step.  The default 480 s watchdog kills
# the process before it finishes.  Set once at import time so it works
# regardless of how the script is launched.
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "7200")

import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = os.environ.get("PRL_MODEL_PATH", "Qwen/Qwen3.5-9B")


# ── Config ────────────────────────────────────────────────

@dataclass
class TrainConfig:
    data: str
    output: str
    model: str = DEFAULT_MODEL
    ref_model: str = ""                 # empty → same as model (or unused under LoRA)

    per_device_batch_size: int = 1
    grad_accum_steps: int = 16          # effective batch 128 on 8 GPUs
    # Number of passes through the same train.jsonl per `iterate.py` iteration.
    # verl calls this `actor.ppo_epochs`; OpenRLHF `--max_epochs`.  With PPO
    # clip+ratio enabled, ppo_epochs=2 ≈ doubles sample efficiency per
    # rollout without destabilising the policy.  See docs/00_naming.md.
    ppo_epochs: int = 2
    learning_rate: float = 1e-4         # LoRA default; lower (~1e-6) for full FT
    warmup_steps: int = 10
    weight_decay: float = 0.0           # RL fine-tuning doesn't benefit
    grad_clip: float = 1.0              # LLM-RL standard

    # ── PPO surrogate (2025 DAPO-style asymmetric clip) ─────
    # `use_ppo_clip=False` degenerates to the older RWR objective (ratio=1,
    # no clip); only useful for ablating the clip's value.
    # `clip_ratio_low`  = 0.2 (standard)  — suppress side; keeping it tight
    #                     prevents probabilities of low-probability tokens
    #                     from being driven to zero (entropy collapse).
    # `clip_ratio_high` = 0.25 (relaxed)  — encourage side; allowing this to
    #                     be larger than the low bound gives low-probability
    #                     tokens room to rise when A>0 (DAPO uses 0.28, we
    #                     keep 0.25 as a safer midpoint for our setup).
    use_ppo_clip: bool = True
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.25

    # KL anchor to the frozen base model (LoRA disabled).  k3 estimator.
    # 0.02 was the initial conservative value; 0.005 (default here) gives
    # the policy ~4x more freedom and matches the DeepSeek-R1-final /
    # DAPO-adjacent range while still preventing long-term drift.
    # Set to 0 to run KL-free (DAPO style) when reward is fully verifiable.
    beta_kl: float = 0.005
    # Per-turn |A| clamp.  0 disables (default — main-line projects don't
    # explicitly clip advantages: zscore normalization + PPO ratio clip
    # already cap the gradient scale, and additional clipping mostly
    # squashes legitimate high-reward outliers (mid200: 3.82% of turns
    # had |A|>2, all of them genuine high-quality posts whose signal we
    # want the policy to learn from at full strength).  Set a positive
    # value as a safety net only when reward magnitudes can't be bounded.
    advantage_clip: float = 0.0
    # 0 = no truncation; >0 = clean truncation at the nearest user-turn
    # boundary so the cut never splits an assistant response.
    max_seq_len: int = 8192

    # LoRA (DDP-friendly; base doubles as π_ref via `disable_adapter()`).
    # r=64 / α=128 is the main-experiment configuration: it gives the
    # adapter enough capacity to absorb the full reasoning-style shift
    # the 2×2 ablation produces, while still keeping the perturbation
    # bounded by the KL anchor.  r=32 / α=64 (the previous default)
    # works for smoke / mid-sized runs but starves capacity at the
    # 1000-question / 2-epoch scale of the main study.
    # Targeting only attention (q,v,o) keeps base-model knowledge intact;
    # adding gate/up/down regresses MLP behaviour on every step.
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    lora_target_modules: str = "q_proj,v_proj,o_proj"

    # ``save_every = 0`` (default): no mid-training checkpoints — only the
    # final ``save_checkpoint`` at the bottom of ``main`` runs, giving
    # exactly one ckpt per iter (= per sample-train cycle in
    # ``scripts/iterate.py``).  Set a positive int to additionally save
    # every N optimizer steps inside the iter.
    save_every: int = 0
    keep_last: int = 1                  # one ckpt per iter (rotate count)
    log_every: int = 1
    seed: int = 42
    resume: str = ""

    def __post_init__(self):
        if not self.ref_model:
            self.ref_model = self.model


# ── Tokenization & dataset ───────────────────────────────

# Special tokens for Qwen3.5 — verified against the model tokenizer
# (see commit history); double-newline `\n\n` is a single merged token
# (271) and single newline `\n` is 198.  `make_response_mask` skips
# *any* number of trailing newlines so the masking stays correct even
# if a future tokenizer revision splits `\n\n` into two `\n`.
IM_START_ID = 248045
IM_END_ID = 248046
THINK_ID = 248068
END_THINK_ID = 248069
ASSISTANT_ID = 74455
USER_ID = 846
NEWLINE_IDS = (198, 271)   # `\n` and `\n\n` — both treated as whitespace
DEFAULT_PAD_ID = 248044    # <|endoftext|>


def truncate_to_turn_boundary(ids: list[int], max_len: int) -> list[int]:
    """Truncate a multi-turn chat from the front so the cut lands on a
    ``<|im_start|>user`` boundary, never in the middle of an assistant turn.

    We walk the token list and find all positions where a new user turn
    starts (`<|im_start|> user \n`).  The returned slice begins at the
    latest such boundary whose tail fits within ``max_len``; if every
    single user turn alone exceeds ``max_len`` we fall back to a plain
    tail truncation so at least the final assistant turn survives.
    """
    if len(ids) <= max_len:
        return ids
    # Find every <|im_start|>user position (not just assistant).
    starts: list[int] = []
    for i, t in enumerate(ids):
        if (t == IM_START_ID and i + 1 < len(ids)
                and ids[i + 1] == USER_ID):
            starts.append(i)
    # Pick the earliest start that keeps ≤ max_len at the tail.
    total = len(ids)
    for s in starts:
        if total - s <= max_len:
            return ids[s:]
    # Every user turn alone is longer than max_len — degenerate session.
    # Fall back to tail cut (may split a turn, mask_empty filter drops it).
    return ids[-max_len:]


def make_response_mask(input_ids: torch.Tensor
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Mark all assistant content tokens and assign turn indices.

    Each assistant turn gets a segment index (1, 2, 3, …).  The chat
    template puts ``<|im_start|>assistant\n[<think>…</think>\n\n]…<|im_end|>``
    around each assistant turn.  We mask **only** the actual response
    content — the role header, the optional think block, and all
    surrounding newline tokens are excluded.

    Whitespace handling is tokenizer-agnostic: after the role marker
    (and after `</think>` if present) we skip any number of consecutive
    newline tokens (``\\n`` = 198 or ``\\n\\n`` = 271 in Qwen3.5),
    so a tokenizer revision that splits or merges newlines won't
    break this code.
    """
    L = input_ids.size(0)
    mask = torch.zeros_like(input_ids)
    segments = torch.zeros_like(input_ids)

    im_starts = (input_ids == IM_START_ID).nonzero(as_tuple=True)[0]
    im_ends = (input_ids == IM_END_ID).nonzero(as_tuple=True)[0]

    def _skip_newlines(pos: int) -> int:
        """Advance past any \\n / \\n\\n tokens at `pos`."""
        while pos < L and input_ids[pos].item() in NEWLINE_IDS:
            pos += 1
        return pos

    turn = 0
    for s_pos in im_starts:
        s = s_pos.item()
        if s + 2 >= L:
            continue
        # Layout: <|im_start|>(s) assistant(s+1) \n(s+2) [<think>…</think>\n\n]content
        if input_ids[s + 1].item() != ASSISTANT_ID:
            continue

        # Content begins after the role newline; skip however many
        # newline tokens the tokenizer emitted there.
        response_start = _skip_newlines(s + 2)

        # Optional <think>…</think> block — skip the whole thing plus
        # any trailing newlines before the real content.
        if response_start < L and input_ids[response_start].item() == THINK_ID:
            think_end = (input_ids[response_start:] == END_THINK_ID
                         ).nonzero(as_tuple=True)[0]
            if len(think_end) > 0:
                # Land just past </think>, then skip trailing newlines.
                response_start = _skip_newlines(
                    response_start + think_end[0].item() + 1)

        # End at the next <|im_end|>.  Newlines immediately before
        # <|im_end|> belong to the content (rare — template doesn't
        # add trailing whitespace) and are kept; the <|im_end|> itself
        # is excluded by the half-open slice below.
        ie = im_ends[im_ends >= response_start]
        response_end = ie[0].item() if len(ie) > 0 else L
        if response_end <= response_start:
            continue

        turn += 1
        mask[response_start:response_end] = 1
        segments[response_start:response_end] = turn

    return mask, segments


class RLDataset(Dataset):
    """Loads per-actor trajectory JSONL and yields tokenized multi-turn samples.

    Each sample is one actor's full trajectory through a session:
      user → assistant (post 1) → user → assistant (post 2) → ...
    with per-turn advantages mapped to token-level advantage tensors.
    """

    def __init__(self, jsonl_path: str, tokenizer,
                 advantage_clip: float = 0.0,
                 max_seq_len: int = 8192):
        """Over-length trajectories are truncated at the nearest user-turn
        boundary (so the cut never splits an assistant response), with
        dropped assistant turns removed from turn_advantages accordingly."""
        self.tokenizer = tokenizer
        self.advantage_clip = advantage_clip
        self.max_seq_len = max_seq_len

        raw = [json.loads(l) for l in open(jsonl_path)]
        self.samples: list[dict] = []
        skipped_nomask = skipped_mismatch = n_truncated = 0
        max_len_seen = 0
        for s in raw:
            turn_advs = s.get("turn_advantages") or [s.get("advantage", 0)]
            if advantage_clip > 0:
                turn_advs = [max(-advantage_clip, min(advantage_clip, a))
                             for a in turn_advs]

            # Per-turn trainability — short posts are kept in the chat
            # context for distribution faithfulness but masked out of
            # the loss.  Old jsonl files (pre-flag) default to
            # all-trainable for backward compatibility.
            turn_trainable = s.get("turn_trainable",
                                   [True] * len(turn_advs))

            text = tokenizer.apply_chat_template(
                s["messages"], tokenize=False, add_generation_prompt=False)
            ids_list = tokenizer(text, add_special_tokens=False).input_ids

            if max_seq_len > 0 and len(ids_list) > max_seq_len:
                ids_list = truncate_to_turn_boundary(ids_list, max_seq_len)
                n_truncated += 1

            ids = torch.tensor(ids_list, dtype=torch.long)
            mask, segments = make_response_mask(ids)

            n_turns = int(segments.max().item())
            # Truncation drops leading assistant turns; keep the last n_turns.
            if n_turns > len(turn_advs) or n_turns == 0:
                skipped_mismatch += 1
                continue
            kept_advs = turn_advs[-n_turns:]
            kept_trainable = turn_trainable[-n_turns:]

            # Zero out the response mask for non-trainable turns: their
            # tokens still inform the model's input distribution but
            # contribute zero loss / zero KL.
            for t in range(n_turns):
                if not kept_trainable[t]:
                    mask[segments == (t + 1)] = 0

            # Re-check after masking — if every kept turn was non-trainable
            # the sample carries no learning signal at all.
            if mask.sum() == 0:
                skipped_nomask += 1
                continue

            adv_per_tok = torch.zeros(len(ids), dtype=torch.float32)
            for t in range(n_turns):
                adv_per_tok[segments == (t + 1)] = float(kept_advs[t])

            self.samples.append({
                "input_ids": ids,
                "response_mask": mask,
                "advantage": adv_per_tok,
                "session_id": s["session_id"],
                # Pre-computed per-token log-probs (filled by
                # `precompute_log_probs` just before the PPO epochs
                # start). `old_log_prob` is the policy's own log-prob
                # at that moment (used for the PPO ratio's denominator),
                # `ref_log_prob` is the frozen-base log-prob (used for
                # the KL anchor). Both shape: [T], float32, CPU-resident.
                "old_log_prob": None,
                "ref_log_prob": None,
            })
            max_len_seen = max(max_len_seen, len(ids))
        print(f"  Dataset: {len(self.samples)} samples ready "
              f"(truncated: {n_truncated}, "
              f"dropped: {skipped_nomask} mask-empty, "
              f"{skipped_mismatch} turn-mismatch, "
              f"max_tokens={max_len_seen})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        item = {
            "input_ids": s["input_ids"],
            "response_mask": s["response_mask"],
            "advantage": s["advantage"],
        }
        if s["old_log_prob"] is not None:
            item["old_log_prob"] = s["old_log_prob"]
        if s["ref_log_prob"] is not None:
            item["ref_log_prob"] = s["ref_log_prob"]
        return item


def collate(batch: list[dict], pad_id: int) -> dict:
    """Right-pad a batch to its max length.

    Cached `old_log_prob` / `ref_log_prob` tensors (present once the
    pre-compute pass has run) are padded with 0.0 — those positions are
    already masked out by `response_mask` so the pad value never
    reaches the loss.
    """
    L = max(b["input_ids"].size(0) for b in batch)
    out_ids, out_mask, out_attn, out_adv = [], [], [], []
    has_old = "old_log_prob" in batch[0]
    has_ref = "ref_log_prob" in batch[0]
    out_old: list[torch.Tensor] = [] if has_old else []
    out_ref: list[torch.Tensor] = [] if has_ref else []
    for b in batch:
        n = b["input_ids"].size(0)
        pad = L - n
        out_ids.append(F.pad(b["input_ids"], (0, pad), value=pad_id))
        out_mask.append(F.pad(b["response_mask"], (0, pad), value=0))
        attn = torch.cat([torch.ones(n, dtype=torch.long),
                          torch.zeros(pad, dtype=torch.long)])
        out_attn.append(attn)
        out_adv.append(F.pad(b["advantage"], (0, pad), value=0.0))
        if has_old:
            out_old.append(F.pad(b["old_log_prob"], (0, pad), value=0.0))
        if has_ref:
            out_ref.append(F.pad(b["ref_log_prob"], (0, pad), value=0.0))
    out: dict[str, torch.Tensor] = {
        "input_ids": torch.stack(out_ids),
        "response_mask": torch.stack(out_mask),
        "attention_mask": torch.stack(out_attn),
        "advantage": torch.stack(out_adv),
    }
    if has_old:
        out["old_log_prob"] = torch.stack(out_old)
    if has_ref:
        out["ref_log_prob"] = torch.stack(out_ref)
    return out


class LengthGroupedSampler(Sampler):
    """Sort by token length, batch nearby lengths together, shuffle batches.

    Within each batch, sequences have similar length → minimal padding.
    Batch order is shuffled each ppo epoch to avoid ordering bias.
    """

    def __init__(self, dataset: RLDataset, batch_size: int, seed: int = 42):
        self.lengths = [len(s["input_ids"]) for s in dataset.samples]
        self.sorted_indices = sorted(range(len(self.lengths)),
                                     key=lambda i: self.lengths[i])
        self.batch_size = batch_size
        self.seed = seed
        self.ppo_epoch = 0

    def __iter__(self):
        batches = [self.sorted_indices[i:i + self.batch_size]
                   for i in range(0, len(self.sorted_indices), self.batch_size)]
        g = torch.Generator()
        g.manual_seed(self.seed + self.ppo_epoch)
        order = torch.randperm(len(batches), generator=g).tolist()
        for idx in order:
            yield from batches[idx]
        self.ppo_epoch += 1

    def __len__(self):
        return len(self.sorted_indices)


# ── Loss ──────────────────────────────────────────────────

def per_token_log_p(logits: torch.Tensor, target_ids: torch.Tensor,
                    chunk: int = 1024) -> torch.Tensor:
    """Gather log p(token_t | <t) for each (B, t) in the response.

    Equivalent to ``F.log_softmax(logits, -1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)``
    but processes the time dimension in chunks of ``chunk`` to keep peak
    memory at ``O(B · chunk · V)`` instead of ``O(B · T · V)``.  For
    Qwen vocab (152k) and T=8k this is ~8× lower peak.  Same trick used
    by Verl / Megatron-LM for the LM-head softmax.
    """
    B, T, V = logits.shape
    out = torch.empty(B, T, dtype=logits.dtype, device=logits.device)
    for i in range(0, T, chunk):
        j = min(i + chunk, T)
        log_p = F.log_softmax(logits[:, i:j, :], dim=-1)
        out[:, i:j] = log_p.gather(-1, target_ids[:, i:j, None]).squeeze(-1)
    return out


def _masked_mean(x: torch.Tensor, mask: torch.Tensor,
                 denom: torch.Tensor) -> torch.Tensor:
    """Token-mean ``(x * mask).sum() / denom`` — DAPO-style batch-level
    averaging. Every response token contributes equally regardless of
    which sequence it belongs to; no length bias.
    """
    return (x * mask).sum() / denom


def compute_loss(model, batch: dict, cfg: TrainConfig) -> tuple[torch.Tensor, dict]:
    """Token-level PPO surrogate + KL anchor.

    ::

        ratio_t = exp(logπ_θ(y_t|·) − logπ_old(y_t|·))       [B, T]

        pg   = − mean_token( min(                        ← clip-surrogate
                  ratio_t · A_t,
                  clip(ratio_t, 1 − ε_lo, 1 + ε_hi) · A_t))
        kl   = + mean_token( exp(log r) − log r − 1 )    ← k3 estimator
                                 with log r = logπ_θ − logπ_ref
        L    = pg + β_kl · kl

    ``old_log_prob`` and ``ref_log_prob`` are pre-computed (detached)
    and attached to the batch by :func:`precompute_log_probs`.  If
    ``old_log_prob`` is absent the ratio degenerates to 1 and the
    surrogate reduces to vanilla RWR (``-A · logπ_θ``).  If
    ``ref_log_prob`` is absent or ``β_kl = 0`` the KL term is skipped.
    """
    input_ids = batch["input_ids"]
    mask = batch["response_mask"][:, 1:].float()
    advantages = batch["advantage"][:, 1:]                 # [B, T] per-token
    n_tok = mask.sum().clamp(min=1.0)

    out = model(input_ids=input_ids,
                attention_mask=batch["attention_mask"],
                use_cache=False)
    logp = per_token_log_p(out.logits[:, :-1], input_ids[:, 1:])  # [B, T]
    del out

    # ── Policy gradient ──
    if cfg.use_ppo_clip and "old_log_prob" in batch:
        old_logp = batch["old_log_prob"][:, 1:]             # detached
        log_ratio = (logp - old_logp).float().clamp(-20.0, 20.0)
        ratio = log_ratio.exp()
        pg_unclip = -advantages * ratio
        pg_clip = -advantages * ratio.clamp(
            1.0 - cfg.clip_ratio_low, 1.0 + cfg.clip_ratio_high)
        pg_per_tok = torch.maximum(pg_unclip, pg_clip)
        clipfrac = _masked_mean((pg_clip > pg_unclip).float(), mask, n_tok)
        # Split the clipfrac into upper / lower edges so we can tell
        # which side is more active.  Upper = ratio > 1+ε_hi (encourage
        # too-confident); lower = ratio < 1-ε_lo (suppress too-strong).
        # When upper » lower, advantages are dominated by positive
        # samples; when lower » upper, by negative samples.
        clipfrac_high = _masked_mean(
            (ratio > 1.0 + cfg.clip_ratio_high).float(), mask, n_tok)
        clipfrac_low = _masked_mean(
            (ratio < 1.0 - cfg.clip_ratio_low).float(), mask, n_tok)
        # Schulman's low-variance estimator of KL(π_θ‖π_old) — drift metric.
        approx_kl_old = _masked_mean(
            0.5 * (logp - old_logp).float().pow(2), mask, n_tok)
    else:                                                   # RWR fallback
        pg_per_tok = -advantages * logp
        clipfrac = torch.zeros((), device=input_ids.device)
        clipfrac_high = torch.zeros((), device=input_ids.device)
        clipfrac_low = torch.zeros((), device=input_ids.device)
        approx_kl_old = torch.zeros((), device=input_ids.device)
    pg_loss = _masked_mean(pg_per_tok, mask, n_tok)

    # ── KL anchor vs frozen base (k3 estimator, always ≥ 0) ──
    if cfg.beta_kl > 0 and "ref_log_prob" in batch:
        log_r_ref = (logp - batch["ref_log_prob"][:, 1:]
                     ).float().clamp(-10.0, 10.0)
        kl_per_tok = log_r_ref.exp() - log_r_ref - 1.0
        kl_loss = _masked_mean(kl_per_tok, mask, n_tok)
    else:
        kl_loss = torch.zeros((), device=input_ids.device)

    total = pg_loss + cfg.beta_kl * kl_loss
    if not torch.isfinite(total):
        # Rare: a single bf16 overflow in a long sequence.  Fall back to
        # the finite component so the optimizer step still happens with
        # a safe gradient instead of corrupting AdamW's moving averages.
        total = pg_loss if torch.isfinite(pg_loss) else torch.zeros(
            (), device=input_ids.device, requires_grad=True)
    # Extra observability — all cheap (single-batch reductions).
    n_resp_tok = float(n_tok.detach())
    n_total_tok = float(input_ids.numel())
    pad_frac = 1.0 - (float(batch["attention_mask"].sum().detach()) / n_total_tok)
    return total, {
        # Loss components
        "pg_loss": float(pg_loss.detach()),
        "kl_loss": float(kl_loss.detach()),
        "total": float(total.detach()),
        # Per-token statistics over the response mask
        "mean_logp_resp": float(_masked_mean(logp, mask, n_tok).detach()),
        "mean_adv": float(_masked_mean(advantages, mask, n_tok).detach()),
        "abs_adv": float(_masked_mean(advantages.abs(), mask, n_tok).detach()),
        # Trust-region diagnostics
        "clipfrac": float(clipfrac.detach()),
        "clipfrac_high": float(clipfrac_high.detach()),
        "clipfrac_low": float(clipfrac_low.detach()),
        "approx_kl_old": float(approx_kl_old.detach()),
        # Batch shape — useful for spotting padding waste / OOM precursors
        "n_response_tokens": n_resp_tok,
        "n_total_tokens": n_total_tok,
        "pad_frac": pad_frac,
    }


def _forward_log_prob(net, ids: torch.Tensor) -> torch.Tensor:
    """Single no-grad forward → length-T CPU float32 log-prob tensor.

    Convention: `logp[i]` is the log-prob of `ids[i]` (the token at
    position i). Position 0 has no predecessor and is padded to 0; the
    response mask ignores it anyway.
    """
    out = net(input_ids=ids,
              attention_mask=torch.ones_like(ids),
              use_cache=False)
    lp = per_token_log_p(out.logits[:, :-1], ids[:, 1:])[0].float().cpu()
    del out
    return F.pad(lp, (1, 0), value=0.0)


def _shard_indices(n: int, world: int, rank: int) -> list[int]:
    """Round-robin partition of `range(n)` into `world` shards.

    Round-robin (rather than contiguous split) keeps long-vs-short
    samples roughly balanced across ranks: ``LengthGroupedSampler``
    sorts ``dataset.samples`` by length, so a contiguous slice would
    give one rank all the long samples → that rank's forwards would
    take far longer than the others, wasting GPU-minutes on idle
    waiting at ``wait_for_everyone``.
    """
    return list(range(rank, n, world))


def _compute_local_log_probs(model, dataset: "RLDataset", device: torch.device,
                             indices: list[int],
                             lora_unwrap=None, ref_model=None,
                             need_old: bool = True,
                             need_ref: bool = True) -> dict[int, dict]:
    """Run forwards over a subset of `dataset.samples`; return a
    {idx: {"old"?: lp, "ref"?: lp}} dict (CPU float32 tensors).

    Pure logic — no distributed coordination, no disk I/O.  The two
    callers (single-rank fast path, multi-rank shard path) feed it
    different `indices` lists.
    """
    out: dict[int, dict] = {}
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for idx in indices:
                ids = dataset.samples[idx]["input_ids"].unsqueeze(0).to(device)
                entry: dict = {}
                if need_old:
                    entry["old"] = _forward_log_prob(model, ids)
                if need_ref:
                    if lora_unwrap is not None:
                        with lora_unwrap.disable_adapter():
                            entry["ref"] = _forward_log_prob(model, ids)
                    elif ref_model is not None:
                        entry["ref"] = _forward_log_prob(ref_model, ids)
                out[idx] = entry
    finally:
        if was_training:
            model.train()
    return out


def _attach_log_probs(dataset: "RLDataset", merged: dict[int, dict]) -> None:
    """Write the gathered cache into ``dataset.samples[idx]`` fields."""
    for idx, entry in merged.items():
        if "old" in entry:
            dataset.samples[idx]["old_log_prob"] = entry["old"]
        if "ref" in entry:
            dataset.samples[idx]["ref_log_prob"] = entry["ref"]


def precompute_log_probs(model, dataset: "RLDataset", accelerator,
                         lora_unwrap=None, ref_model=None,
                         need_old: bool = True, need_ref: bool = True,
                         cache_dir: "os.PathLike | None" = None) -> None:
    """Cache π_old and π_ref per-token log-probs into ``dataset.samples``.

    Fills ``dataset.samples[i]["old_log_prob"]`` and ``["ref_log_prob"]``
    with length-T CPU float32 tensors that the PPO-clip loss reads
    later.  Semantics:

    * ``old_log_prob`` = the current policy's log-prob at the moment
      this pre-compute runs (i.e. just before the PPO epochs begin).
      Frozen across all ``ppo_epochs`` → the π_old of the PPO ratio.
      Matches verl's ``actor.compute_log_prob``.
    * ``ref_log_prob`` = the frozen-base log-prob, used in the KL
      anchor.  LoRA path: same model with the adapter disabled.
      Full-FT path: a separate ``ref_model``.

    **DDP shard mode.**  Under ``accelerator.num_processes > 1`` we
    split the dataset round-robin across ranks so each forward is
    done exactly once total (rather than once per rank, as the
    legacy implementation did).  Every rank writes its local shard
    to ``cache_dir/rank_{i}.pt``, waits at ``wait_for_everyone``,
    then reads every other rank's shard so the full cache is
    available for the SGD pass that follows.  Disk-based gather
    keeps us out of NCCL variable-length-tensor land; the cache
    is small (~10 MB / 200 samples) so this is bandwidth-irrelevant.

    Single-rank stays on a fast in-memory path (no disk roundtrip).

    `cache_dir` defaults to a unique tempdir on the main process and
    is broadcast to all ranks via ``accelerator.wait_for_everyone``
    + filesystem visibility.  The directory is removed by the main
    process at the end of the call.
    """
    if not (need_old or need_ref):
        return
    world = accelerator.num_processes
    rank = accelerator.process_index
    n = len(dataset.samples)
    my_indices = _shard_indices(n, world, rank)

    local = _compute_local_log_probs(
        model, dataset, accelerator.device, my_indices,
        lora_unwrap=lora_unwrap, ref_model=ref_model,
        need_old=need_old, need_ref=need_ref,
    )

    # Single-rank: skip the disk roundtrip entirely.
    if world == 1:
        _attach_log_probs(dataset, local)
        return

    # Multi-rank: agree on a shared cache dir.  Caller-supplied path
    # takes precedence (e.g. inside the run dir); otherwise the main
    # process picks a fresh tempdir and the path is recovered on each
    # rank via the same deterministic naming convention.
    if cache_dir is None:
        # We need a path all ranks can derive without communication.
        # ``os.environ.get("ACCELERATE_*")`` is not stable enough; use
        # the model's id() — same on all ranks since the model object
        # is identical (DDP-wrapped weights, but same Python object id
        # within a process is rank-specific, so this would diverge).
        # Cleanest: caller MUST pass cache_dir for multi-rank.  Fail
        # loudly so the bug is obvious instead of silently wrong.
        raise ValueError(
            "precompute_log_probs: cache_dir must be supplied when "
            "num_processes > 1 (no portable path-derivation across ranks)"
        )
    cache_dir = Path(cache_dir)
    if accelerator.is_main_process:
        cache_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    # Each rank dumps its own shard.
    shard_path = cache_dir / f"rank_{rank}.pt"
    torch.save(local, shard_path)
    accelerator.wait_for_everyone()

    # All ranks read every shard; cache size is ~10 MB / 200 samples
    # so duplicating across ranks costs nothing.
    merged: dict[int, dict] = {}
    for r in range(world):
        merged.update(torch.load(cache_dir / f"rank_{r}.pt",
                                 map_location="cpu"))
    _attach_log_probs(dataset, merged)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        shutil.rmtree(cache_dir, ignore_errors=True)


# ── Checkpointing ────────────────────────────────────────

def save_checkpoint(acc: Accelerator, model, optimizer, scheduler,
                    step: int, ppo_epoch: int, cfg: TrainConfig,
                    output_dir: Path) -> None:
    """Save a fully-resumable checkpoint.

    LoRA path (default): only adapter (≈0.7 GB) + optimizer + scheduler
    are persisted. Skipping `acc.save_state` saves the wasteful 18 GB
    duplicate of the frozen base each step.

    Full-FT fallback: relies on `acc.save_state` (sharded model + optim).

    If `cfg.keep_last > 0`, prunes older `step_*` directories, keeping
    only the most recent `keep_last`.
    """
    ckpt_dir = output_dir / f"step_{step}"
    if cfg.use_lora:
        if acc.is_main_process:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            acc.unwrap_model(model).save_pretrained(ckpt_dir / "adapter")
            torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
            torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")
        acc.wait_for_everyone()
    else:
        acc.save_state(str(ckpt_dir))
    if acc.is_main_process:
        meta = {"step": step, "ppo_epoch": ppo_epoch, "config": asdict(cfg)}
        (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        acc.print(f"  saved checkpoint → {ckpt_dir}")
        if cfg.keep_last > 0:
            existing = sorted(
                output_dir.glob("step_*"),
                key=lambda p: int(p.name.split("_")[1]))
            for old in existing[:-cfg.keep_last]:
                shutil.rmtree(old, ignore_errors=True)
                acc.print(f"  pruned old checkpoint → {old}")


def load_checkpoint(acc: Accelerator, model, optimizer, scheduler,
                    resume_dir: str, cfg: TrainConfig) -> tuple[int, int]:
    """Restore from a checkpoint produced by `save_checkpoint`.

    Returns ``(step, ppo_epoch)``.  Falls back to the legacy ``"epoch"``
    meta key if a checkpoint was written before the rename.
    """
    rd = Path(resume_dir)
    meta = json.loads((rd / "meta.json").read_text())
    if cfg.use_lora:
        from peft import PeftModel
        unwrapped = acc.unwrap_model(model)
        # Replace the live adapter weights in-place with the saved ones.
        # `load_adapter` overwrites the existing "default" adapter so the
        # DDP-wrapped param refs stay valid (no need to re-prepare).
        unwrapped.load_adapter(str(rd / "adapter"), adapter_name="default",
                               is_trainable=True)
        optimizer.load_state_dict(
            torch.load(rd / "optimizer.pt", map_location="cpu"))
        scheduler.load_state_dict(
            torch.load(rd / "scheduler.pt", map_location="cpu"))
    else:
        acc.load_state(resume_dir)
    acc.print(f"  resumed from {resume_dir} at step {meta['step']}")
    return meta["step"], meta.get("ppo_epoch", meta.get("epoch", 0))


# ── Model loading ────────────────────────────────────────

def load_model(model_path: str, dtype: torch.dtype = torch.bfloat16):
    """Load text-only causal LM from a checkpoint.

    Handles multimodal-preview checkpoints (e.g. Qwen3.5
    `Qwen3_5ForConditionalGeneration` whose root config nests `text_config`
    + `vision_config`) by routing to the model-specific text-only
    `*ForCausalLM` class instead of `AutoModelForCausalLM` (which would
    instantiate the multimodal wrapper and trip on the missing root-level
    `vocab_size`).
    """
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    is_nested_multimodal = hasattr(cfg, "text_config") and not hasattr(cfg, "vocab_size")
    if is_nested_multimodal:
        import importlib
        mod_name = f"transformers.models.{cfg.model_type}"
        try:
            mod = importlib.import_module(mod_name)
            text_only_cls = next(
                getattr(mod, n) for n in dir(mod)
                if n.endswith("ForCausalLM") and not n.startswith("_")
            )
            return text_only_cls.from_pretrained(
                model_path, torch_dtype=dtype,
                attn_implementation="sdpa", trust_remote_code=True,
            )
        except (ImportError, StopIteration):
            pass
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )


def attach_lora(model, cfg: "TrainConfig"):
    """Wrap base model with a LoRA adapter and freeze the base.

    Default `cfg.lora_target_modules = "q_proj,v_proj,o_proj"` patches
    only attention projections — keeps the base MLP and Mamba-style
    `in_proj_*` / `out_proj` weights frozen so general-domain knowledge
    isn't perturbed by the RL signal.

    Set the flag to `""` (empty) to fall back to PEFT's `"all-linear"`,
    which covers attention proj, MLP gate/up/down, and the linear-
    attention `in_proj_*` / `out_proj` layers — both Qwen3.5 hybrid
    branches get adapted but with much higher KL drift risk.
    """
    targets = cfg.lora_target_modules.strip() or "all-linear"
    if "," in targets:
        targets = [t.strip() for t in targets.split(",") if t.strip()]
    lora_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=targets,
    )
    return get_peft_model(model, lora_cfg)


def freeze(model) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


# ── Main ─────────────────────────────────────────────────

def parse_args() -> TrainConfig:
    """CLI defaults mirror the TrainConfig dataclass above."""
    d = TrainConfig.__dataclass_fields__
    p = argparse.ArgumentParser(
        description="Offline RL trainer — token-level PPO clip + KL-to-base.")
    p.add_argument("--data", required=True, help="JSONL from rl.extract")
    p.add_argument("--output", required=True, help="Checkpoint directory")
    p.add_argument("--model", default=d["model"].default)
    p.add_argument("--ref-model", default="")
    p.add_argument("--per-device-batch-size", type=int,
                   default=d["per_device_batch_size"].default)
    p.add_argument("--grad-accum-steps", type=int,
                   default=d["grad_accum_steps"].default)
    # `ppo_epochs` follows verl's `actor.ppo_epochs`: number of passes
    # through the same train.jsonl per iter.  `--num-epochs` kept as a
    # deprecated alias so in-flight runs can still resume.
    p.add_argument("--ppo-epochs", "--num-epochs", dest="ppo_epochs",
                   type=int, default=d["ppo_epochs"].default,
                   help="Passes through train.jsonl per iter (verl: ppo_epochs)")
    p.add_argument("--learning-rate", type=float,
                   default=d["learning_rate"].default)
    p.add_argument("--warmup-steps", type=int,
                   default=d["warmup_steps"].default)
    p.add_argument("--weight-decay", type=float,
                   default=d["weight_decay"].default)
    p.add_argument("--grad-clip", type=float, default=d["grad_clip"].default)
    p.add_argument("--beta-kl", type=float, default=d["beta_kl"].default,
                   help="KL(π_θ‖π_base) coefficient. 0 disables the anchor.")
    p.add_argument("--use-ppo-clip", action=argparse.BooleanOptionalAction,
                   default=d["use_ppo_clip"].default,
                   help="Enable PPO ratio + asymmetric clip. "
                        "--no-use-ppo-clip degenerates to RWR (ratio=1).")
    p.add_argument("--clip-ratio-low", type=float,
                   default=d["clip_ratio_low"].default,
                   help="PPO clip lower bound = 1 - clip_ratio_low (default 0.2)")
    p.add_argument("--clip-ratio-high", type=float,
                   default=d["clip_ratio_high"].default,
                   help="PPO clip upper bound = 1 + clip_ratio_high "
                        "(default 0.25; DAPO-style asymmetric)")
    p.add_argument("--advantage-clip", type=float,
                   default=d["advantage_clip"].default,
                   help="Clamp |advantage| ≤ this; 0 (default) disables. "
                        "Mainline projects (TRL/OpenRLHF/Verl/DeepSeek-R1/"
                        "DAPO) don't explicitly clip — zscore + PPO ratio "
                        "clip already control gradient scale.")
    p.add_argument("--max-seq-len", type=int, default=d["max_seq_len"].default,
                   help="Truncate over-length samples at the nearest "
                        "user-turn boundary (0 = keep full length). "
                        "Safety valve against OOM on extreme outliers.")
    p.add_argument("--use-lora", action=argparse.BooleanOptionalAction,
                   default=d["use_lora"].default,
                   help="LoRA + DDP (default). Use --no-use-lora for full FT.")
    p.add_argument("--lora-r", type=int, default=d["lora_r"].default)
    p.add_argument("--lora-alpha", type=int, default=d["lora_alpha"].default)
    p.add_argument("--lora-dropout", type=float, default=d["lora_dropout"].default)
    p.add_argument("--lora-target-modules", default=d["lora_target_modules"].default,
                   help="Comma-separated module names; empty → \"all-linear\".")
    p.add_argument("--save-every", type=int, default=d["save_every"].default,
                   help="Mid-training save cadence (in optimizer steps). "
                        "Default 0 = save only once at the end of training "
                        "(= one ckpt per iter under iterate.py).")
    p.add_argument("--keep-last", type=int, default=d["keep_last"].default,
                   help="Keep at most this many step_* checkpoints (0 = all)")
    p.add_argument("--log-every", type=int, default=d["log_every"].default)
    p.add_argument("--seed", type=int, default=d["seed"].default)
    p.add_argument("--resume", default="")
    args = p.parse_args()
    return TrainConfig(**vars(args))


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    output_dir = Path(cfg.output)

    # No `mixed_precision` arg: the model is loaded BF16 (`load_model`
    # below) and that's the dtype we keep end-to-end.  Setting it would
    # wrap forward with autocast that casts logits BF16→FP32 at the
    # boundary — for vocab=152k and seq=8k that's ~6 GB extra per
    # forward, and we run two forwards (policy + ref).  Same approach
    # as TRL/OpenRLHF/axolotl LoRA paths.
    acc = Accelerator(
        gradient_accumulation_steps=cfg.grad_accum_steps,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=2))],
    )

    if acc.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
        acc.print(f"\nDevices: {acc.num_processes}, "
                  f"effective batch = {cfg.per_device_batch_size} × "
                  f"{cfg.grad_accum_steps} × {acc.num_processes} = "
                  f"{cfg.per_device_batch_size * cfg.grad_accum_steps * acc.num_processes}\n")

    # Tokenizer + dataset
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id or DEFAULT_PAD_ID
    dataset = RLDataset(cfg.data, tokenizer,
                        cfg.advantage_clip, cfg.max_seq_len)
    sampler = LengthGroupedSampler(dataset, cfg.per_device_batch_size,
                                    seed=cfg.seed)
    loader = DataLoader(
        dataset, batch_size=cfg.per_device_batch_size, shuffle=False,
        sampler=sampler,
        collate_fn=lambda b: collate(b, pad_id),
        num_workers=2, pin_memory=True,
    )

    # Models
    acc.print(f"Loading policy from {cfg.model}")
    model = load_model(cfg.model)
    if cfg.use_lora:
        model = attach_lora(model, cfg)
        if acc.is_main_process:
            model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    if cfg.use_lora:
        # LoRA + grad-checkpointing requires inputs to require grads so the
        # frozen embedding-output activations propagate through the adapter.
        model.enable_input_require_grads()

    if cfg.use_lora:
        ref_model = None                            # disable_adapter() doubles as ref
    elif cfg.beta_kl > 0:
        acc.print(f"Loading reference from {cfg.ref_model}")
        ref_model = load_model(cfg.ref_model)
        freeze(ref_model)
    else:
        ref_model = None

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    steps_per_ppo_epoch = math.ceil(len(loader) / cfg.grad_accum_steps)
    total_steps = steps_per_ppo_epoch * cfg.ppo_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps)

    model, optimizer, loader, scheduler = acc.prepare(
        model, optimizer, loader, scheduler)
    if ref_model is not None:
        # Keep ref unsharded: full copy per GPU, no optimizer, no grad.
        # 9B BF16 ≈ 18 GB; only used when use_lora=False (full FT path).
        ref_model = ref_model.to(acc.device)

    # Cache the unwrapped (peft) model once for cheap `disable_adapter()`
    # access inside the hot loop — avoids per-step `acc.unwrap_model()`.
    lora_unwrap = acc.unwrap_model(model) if cfg.use_lora else None

    start_step = start_ppo_epoch = 0
    if cfg.resume:
        start_step, start_ppo_epoch = load_checkpoint(
            acc, model, optimizer, scheduler, cfg.resume, cfg)

    # ── Pre-compute π_old and π_ref log-probs ──
    # Runs once, just before the PPO epochs.  On resume, the current
    # checkpoint's weights become the new π_old — trust region is
    # effectively re-anchored from that ckpt.  Every rank fills its
    # own copy; tensors are tiny so no distributed broadcast needed.
    need_old = cfg.use_ppo_clip
    need_ref = cfg.beta_kl > 0
    if need_old or need_ref:
        pc_t0 = time.time()
        acc.print(f"Pre-computing log-probs "
                  f"(old={need_old}, ref={need_ref}) on {len(dataset)} samples...")
        precompute_log_probs(
            model, dataset, acc,
            lora_unwrap=lora_unwrap, ref_model=ref_model,
            need_old=need_old, need_ref=need_ref,
            # Cache lives inside the run dir so accidentally re-running
            # mid-iter doesn't collide with another iter's tempdir.
            cache_dir=output_dir / "log_prob_cache")
        acc.print(f"  done in {time.time() - pc_t0:.0f}s\n")

    # Training loop
    metrics_log_path = output_dir / "metrics.jsonl"
    metrics_log = open(metrics_log_path, "a") if acc.is_main_process else None
    step = start_step
    t0 = time.time()
    micro_count = 0
    micro_metrics: dict[str, float] = {}

    for ppo_epoch in range(start_ppo_epoch, cfg.ppo_epochs):
        for batch in loader:
            with acc.accumulate(model):
                loss, m = compute_loss(model, batch, cfg)
                acc.backward(loss)
                if acc.sync_gradients and cfg.grad_clip > 0:
                    acc.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            for k, v in m.items():
                micro_metrics[k] = micro_metrics.get(k, 0.0) + v
            micro_count += 1

            if acc.sync_gradients:
                step += 1
                if step % cfg.log_every == 0 and acc.is_main_process:
                    avg = {k: v / micro_count for k, v in micro_metrics.items()}
                    elapsed = time.time() - t0
                    # Compact one-liner. Layout grouped: loss → policy
                    # signal → trust region → batch shape → schedule.
                    line = (f"step={step:5d} ppo_ep={ppo_epoch} "
                            f"pg={avg['pg_loss']:+.4f} "
                            f"kl={avg['kl_loss']:+.4f} "
                            f"L={avg['total']:+.4f} "
                            f"logp={avg['mean_logp_resp']:+.3f} "
                            f"adv={avg['mean_adv']:+.3f}±{avg.get('abs_adv', 0.0):.2f} "
                            f"clip%={avg.get('clipfrac', 0.0) * 100:.1f}"
                            f"(↑{avg.get('clipfrac_high', 0.0) * 100:.1f}"
                            f"/↓{avg.get('clipfrac_low', 0.0) * 100:.1f}) "
                            f"kl_old={avg.get('approx_kl_old', 0.0):.3f} "
                            f"resp_tok={avg.get('n_response_tokens', 0.0):.0f} "
                            f"pad={avg.get('pad_frac', 0.0) * 100:.0f}% "
                            f"lr={scheduler.get_last_lr()[0]:.2e} "
                            f"{elapsed:.0f}s")
                    acc.print(line)
                    metrics_log.write(json.dumps({
                        "step": step, "ppo_epoch": ppo_epoch,
                        **avg, "lr": scheduler.get_last_lr()[0],
                        "elapsed_s": round(elapsed, 1),
                    }) + "\n")
                    metrics_log.flush()
                if cfg.save_every > 0 and step % cfg.save_every == 0:
                    save_checkpoint(acc, model, optimizer, scheduler,
                                    step, ppo_epoch, cfg, output_dir)
                micro_metrics = {}
                micro_count = 0

    save_checkpoint(acc, model, optimizer, scheduler, step, ppo_epoch, cfg, output_dir)
    if metrics_log:
        metrics_log.close()
    acc.print(f"\nTraining done. {step} optimizer steps in "
              f"{time.time() - t0:.0f}s.")

    # Hard-stop to avoid NCCL teardown deadlocks during interpreter shutdown.
    acc.wait_for_everyone()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    os._exit(0)


if __name__ == "__main__":
    main()
