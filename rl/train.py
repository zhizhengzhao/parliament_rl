#!/usr/bin/env python3
"""Offline RL trainer for Parliament-collected data.

Algorithm: token-level Reward-Weighted Regression with KL-anchor to base
(ReST / RAFT family; AWR's linear cousin). The loss is

    L = -mean_token( clip(A,±C) · logπ_θ(y_t|x,y_<t) )      ← policy
        + β · mean_token( exp(log r) − log r − 1 )            ← KL anchor
            where  log r = logπ_θ − logπ_base ,  base = LoRA disabled.

Why no PPO ratio? `rl/extract.py` re-renders every prompt from `parliament.db`
(anonymized names, custom headers, `[you]` markers, score-visibility
gating). The training prompt-token sequence therefore differs from what
the rollout actually saw, so any importance ratio against a logged
"old log-prob" would be against the wrong distribution. Treat the data
as off-policy and use the off-policy-correct objective: weighted
regression. Multi-iteration safety comes from the KL anchor to base
plus ReST-style data refresh across iters/total_epochs in `scripts/iterate.py`.

Why token-level aggregation? `sum / mask.sum()` over the whole batch
(DAPO-style) gives every response token equal weight. Per-sequence
mean-then-batch-mean instead biases gradients toward short responses.

Naming (verl-aligned, see docs/00_naming.md):
    --ppo-epochs N   Passes through the same train.jsonl per iter.
                     `--num-epochs` kept as deprecated alias.
    Outer loops (`iter`, `total_epochs`) are owned by `scripts/iterate.py`;
    this script trains one shard's data once, repeated `ppo_epochs` times.

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
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

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
    # verl calls this `actor.ppo_epochs`; OpenRLHF `--max_epochs`.  1-2 is
    # plenty for RWR (no IS to keep "fresh"); see docs/00_naming.md.
    ppo_epochs: int = 2
    learning_rate: float = 1e-4         # LoRA default; lower (~1e-6) for full FT
    warmup_steps: int = 10
    weight_decay: float = 0.0           # RL fine-tuning doesn't benefit
    grad_clip: float = 1.0              # LLM-RL standard

    # RWR + KL anchor.
    # `beta_kl` 0.02 gives KL(π_θ‖π_base) ≈ O(1) on Sciencepedia data;
    # raise to 0.05 if KL drifts past 5, lower to 0.005 if PG signal is
    # being suffocated.  Monitor `kl` column in metrics.jsonl.
    beta_kl: float = 0.02
    advantage_clip: float = 2.0         # clamp |A| per turn to ±this
    max_seq_len: int = 8192             # 0 = no limit; >0 drops longer samples

    # LoRA (DDP-friendly; base doubles as π_ref via `disable_adapter()`).
    # r=32 / α=64 on 9B is the sweet spot — larger r over-parameterizes
    # the perturbation and lets KL drift; smaller starves capacity.
    # Targeting only attention (q,v,o) keeps base-model knowledge intact;
    # adding gate/up/down regresses MLP behaviour on every step.
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_target_modules: str = "q_proj,v_proj,o_proj"

    save_every: int = 50
    keep_last: int = 3                  # rotate checkpoints (0 keeps all)
    log_every: int = 1
    seed: int = 42
    resume: str = ""

    def __post_init__(self):
        if not self.ref_model:
            self.ref_model = self.model


# ── Tokenization & dataset ───────────────────────────────

# Special tokens for Qwen3.5
IM_START_ID = 248045
IM_END_ID = 248046
THINK_ID = 248068
END_THINK_ID = 248069
ASSISTANT_ID = 74455
USER_ID = 846
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
    template puts ``<|im_start|>assistant\n…<|im_end|>`` for every turn;
    intermediate turns contain just the response, and the final turn
    additionally has a ``<think>\n\n</think>\n\n`` prefix.  We mask the
    actual content tokens only, skipping the role header and any think
    block.
    """
    L = input_ids.size(0)
    mask = torch.zeros_like(input_ids)
    segments = torch.zeros_like(input_ids)

    im_starts = (input_ids == IM_START_ID).nonzero(as_tuple=True)[0]
    im_ends = (input_ids == IM_END_ID).nonzero(as_tuple=True)[0]

    turn = 0
    for s_pos in im_starts:
        s = s_pos.item()
        if s + 2 >= L:
            continue
        # Identify role: <|im_start|> <role_id> \n …
        if input_ids[s + 1].item() != ASSISTANT_ID:
            continue

        # Content begins after `\n` (s+2) by default.
        response_start = s + 3

        # If the immediately-following content is `<think>` block, skip it.
        # Layout: <|im_start|>(s) assistant(s+1) \n(s+2) <think>(s+3) ...
        #         </think>(k) \n\n(k+1) content...
        if response_start < L and input_ids[response_start].item() == THINK_ID:
            think_end = (input_ids[response_start:] == END_THINK_ID).nonzero(
                as_tuple=True)[0]
            if len(think_end) > 0:
                # Skip </think> + whitespace token (\n\n)
                response_start = response_start + think_end[0].item() + 2

        # Find the next <|im_end|> after response_start.
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
                 advantage_clip: float = 2.0,
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

            text = tokenizer.apply_chat_template(
                s["messages"], tokenize=False, add_generation_prompt=False)
            ids_list = tokenizer(text, add_special_tokens=False).input_ids

            if max_seq_len > 0 and len(ids_list) > max_seq_len:
                ids_list = truncate_to_turn_boundary(ids_list, max_seq_len)
                n_truncated += 1

            ids = torch.tensor(ids_list, dtype=torch.long)
            mask, segments = make_response_mask(ids)
            if mask.sum() == 0:
                skipped_nomask += 1
                continue

            n_turns = int(segments.max().item())
            # Truncation drops leading assistant turns; keep the last n_turns.
            if n_turns > len(turn_advs) or n_turns == 0:
                skipped_mismatch += 1
                continue
            kept_advs = turn_advs[-n_turns:]

            adv_per_tok = torch.zeros(len(ids), dtype=torch.float32)
            for t in range(n_turns):
                adv_per_tok[segments == (t + 1)] = float(kept_advs[t])

            self.samples.append({
                "input_ids": ids,
                "response_mask": mask,
                "advantage": adv_per_tok,
                "session_id": s["session_id"],
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
        return {
            "input_ids": s["input_ids"],
            "response_mask": s["response_mask"],
            "advantage": s["advantage"],
        }


def collate(batch: list[dict], pad_id: int) -> dict:
    """Right-pad a batch to its max length."""
    L = max(b["input_ids"].size(0) for b in batch)
    out_ids, out_mask, out_attn, out_adv = [], [], [], []
    for b in batch:
        n = b["input_ids"].size(0)
        pad = L - n
        out_ids.append(F.pad(b["input_ids"], (0, pad), value=pad_id))
        out_mask.append(F.pad(b["response_mask"], (0, pad), value=0))
        attn = torch.cat([torch.ones(n, dtype=torch.long),
                          torch.zeros(pad, dtype=torch.long)])
        out_attn.append(attn)
        out_adv.append(F.pad(b["advantage"], (0, pad), value=0.0))
    return {
        "input_ids": torch.stack(out_ids),
        "response_mask": torch.stack(out_mask),
        "attention_mask": torch.stack(out_attn),
        "advantage": torch.stack(out_adv),
    }


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


def compute_loss(model, ref_model, batch: dict, beta_kl: float,
                 lora_unwrap=None) -> tuple[torch.Tensor, dict]:
    """Token-level RWR + KL anchor.

        pg = - mean_token( A · logπ_θ(y_t|·) )            [B, T] → scalar
        kl = + mean_token( exp(log r) − log r − 1 )       k3 estimator
        L  = pg + β · kl

    `mean_token` = `(x * mask).sum() / mask.sum()` over the whole batch
    (DAPO-style) so every response token contributes equally — no length
    bias, no per-sequence renormalization that would over-weight short
    rollouts.

    `lora_unwrap` (PeftModel | None): when non-None, base = `model` with
    adapter disabled — saves 18 GB/GPU vs a duplicated π_ref. When None
    (full FT path), use the explicit `ref_model`.
    """
    input_ids = batch["input_ids"]
    response_mask = batch["response_mask"][:, 1:].float()
    target_ids = input_ids[:, 1:]
    advantages = batch["advantage"][:, 1:]                # [B, T] per-token
    attn = batch["attention_mask"]
    n_tok = response_mask.sum().clamp(min=1.0)

    out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    logp = per_token_log_p(out.logits[:, :-1], target_ids)         # [B, T]
    del out

    # Reference forward — needed when β_kl > 0.  No grad, bf16, single pass.
    if beta_kl > 0:
        with torch.no_grad():
            if lora_unwrap is not None:
                with lora_unwrap.disable_adapter():
                    ref_out = model(input_ids=input_ids,
                                    attention_mask=attn,
                                    use_cache=False)
            else:
                ref_out = ref_model(input_ids=input_ids,
                                    attention_mask=attn,
                                    use_cache=False)
            logp_ref = per_token_log_p(ref_out.logits[:, :-1], target_ids)
            del ref_out
    else:
        logp_ref = None

    # Policy gradient: per-token RWR.
    pg_per_tok = -advantages * logp                        # [B, T]
    pg_loss = (pg_per_tok * response_mask).sum() / n_tok

    # KL(π_θ‖π_ref) k3 estimator (Schulman 2020): always ≥ 0, low variance.
    if beta_kl > 0:
        log_r = (logp - logp_ref).float().clamp(-10.0, 10.0)
        kl_per_tok = log_r.exp() - log_r - 1.0
        kl_loss = (kl_per_tok * response_mask).sum() / n_tok
    else:
        kl_loss = torch.zeros((), device=input_ids.device)

    total = pg_loss + beta_kl * kl_loss
    if not torch.isfinite(total):
        # Rare: a single bf16 overflow in a long sequence.  Fall back to
        # the finite component so the optimizer step still happens with a
        # safe gradient instead of corrupting AdamW's moving averages.
        total = pg_loss if torch.isfinite(pg_loss) else torch.zeros(
            (), device=input_ids.device, requires_grad=True)
    return total, {
        "pg_loss": float(pg_loss.detach()),
        "kl_loss": float(kl_loss.detach()),
        "total": float(total.detach()),
        "mean_logp_resp": float(((logp * response_mask).sum() / n_tok).detach()),
        "mean_adv": float((advantages * response_mask).sum().detach() / n_tok),
    }


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
        description="Offline RL trainer — token-level RWR + KL-to-base.")
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
    p.add_argument("--advantage-clip", type=float,
                   default=d["advantage_clip"].default,
                   help="Clamp |advantage| ≤ this (set 0 to disable)")
    p.add_argument("--max-seq-len", type=int, default=d["max_seq_len"].default,
                   help="Drop samples longer than this (0 = keep all). "
                        "Safety valve against OOM on extreme outliers.")
    p.add_argument("--use-lora", action=argparse.BooleanOptionalAction,
                   default=d["use_lora"].default,
                   help="LoRA + DDP (default). Use --no-use-lora for full FT.")
    p.add_argument("--lora-r", type=int, default=d["lora_r"].default)
    p.add_argument("--lora-alpha", type=int, default=d["lora_alpha"].default)
    p.add_argument("--lora-dropout", type=float, default=d["lora_dropout"].default)
    p.add_argument("--lora-target-modules", default=d["lora_target_modules"].default,
                   help="Comma-separated module names; empty → \"all-linear\".")
    p.add_argument("--save-every", type=int, default=d["save_every"].default)
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
                loss, m = compute_loss(model, ref_model, batch,
                                       cfg.beta_kl, lora_unwrap=lora_unwrap)
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
                    line = (f"step={step:5d} ppo_ep={ppo_epoch} "
                            f"pg={avg['pg_loss']:+.4f} "
                            f"kl={avg['kl_loss']:+.4f} "
                            f"L={avg['total']:+.4f} "
                            f"logp={avg['mean_logp_resp']:+.3f} "
                            f"adv={avg['mean_adv']:+.3f} "
                            f"lr={scheduler.get_last_lr()[0]:.2e} "
                            f"{elapsed:.0f}s")
                    acc.print(line)
                    metrics_log.write(json.dumps({
                        "step": step, "ppo_epoch": ppo_epoch,
                        **avg, "lr": scheduler.get_last_lr()[0],
                        "elapsed_s": round(elapsed, 1),
                    }) + "\n")
                    metrics_log.flush()
                if step % cfg.save_every == 0:
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
