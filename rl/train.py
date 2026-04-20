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
plus 4-iteration ReST-style data refresh in `scripts/iterate.py`.

Why token-level aggregation? `sum / mask.sum()` over the whole batch
(DAPO-style) gives every response token equal weight. Per-sequence
mean-then-batch-mean instead biases gradients toward short responses.

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
    num_epochs: int = 2                 # 1-2 is plenty for RWR (no IS to keep "fresh")
    learning_rate: float = 1e-4         # LoRA default; lower (~1e-6) for full FT
    warmup_steps: int = 10
    weight_decay: float = 0.0           # RL fine-tuning doesn't benefit
    grad_clip: float = 1.0              # LLM-RL standard

    # RWR + KL anchor.
    # `beta_kl` 0.02 gives KL(π_θ‖π_base) ≈ O(1) on Sciencepedia data;
    # raise to 0.05 if KL drifts past 5, lower to 0.005 if PG signal is
    # being suffocated.  Monitor `kl` column in metrics.jsonl.
    beta_kl: float = 0.02
    advantage_clip: float = 2.0         # clamp |A| to 2 (covers ~95% of N(0,1) tail)
    advantage_min_abs: float = 0.1      # drop |A|<0.1 samples (no signal, dilute grad)

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
END_THINK_ID = 248069
DEFAULT_PAD_ID = 248044    # <|endoftext|>


def make_response_mask(input_ids: torch.Tensor) -> torch.Tensor:
    """Mark the assistant's actual content tokens (post-`</think>`, pre-`<|im_end|>`).

    Robust to user content containing literal `</think>`: locates the
    last `<|im_start|>` (assistant section start), then the `</think>`
    that closes the assistant's empty/real reasoning block, then the
    next `<|im_end|>`.
    """
    im_starts = (input_ids == IM_START_ID).nonzero(as_tuple=True)[0]
    if len(im_starts) == 0:
        return torch.zeros_like(input_ids)
    asst_start = im_starts[-1].item()

    end_think = (input_ids == END_THINK_ID).nonzero(as_tuple=True)[0]
    end_think_after = end_think[end_think > asst_start]
    if len(end_think_after) == 0:
        return torch.zeros_like(input_ids)
    response_start = end_think_after[0].item() + 2  # skip `</think>` + `\n\n`

    im_ends = (input_ids == IM_END_ID).nonzero(as_tuple=True)[0]
    eot_after = im_ends[im_ends >= response_start]
    response_end = eot_after[0].item() if len(eot_after) else input_ids.size(0)

    mask = torch.zeros_like(input_ids)
    mask[response_start:response_end] = 1
    return mask


class RLDataset(Dataset):
    """Loads a JSONL of (messages, advantage) and yields tokenized samples.

    All samples are kept regardless of length (no truncation, no filtering
    by token count). Length-grouped batching minimizes padding overhead.
    """

    def __init__(self, jsonl_path: str, tokenizer,
                 advantage_clip: float = 2.0,
                 advantage_min_abs: float = 0.1):
        self.tokenizer = tokenizer
        self.advantage_clip = advantage_clip
        self.advantage_min_abs = advantage_min_abs

        raw = [json.loads(l) for l in open(jsonl_path)]
        self.samples: list[dict] = []
        skipped_lowsig = skipped_nomask = 0
        max_len_seen = 0
        for s in raw:
            adv = float(s["advantage"])
            if abs(adv) < advantage_min_abs:
                skipped_lowsig += 1
                continue
            text = tokenizer.apply_chat_template(
                s["messages"], tokenize=False, add_generation_prompt=False)
            ids_list = tokenizer(text, add_special_tokens=False).input_ids
            ids = torch.tensor(ids_list, dtype=torch.long)
            mask = make_response_mask(ids)
            if mask.sum() == 0:
                skipped_nomask += 1
                continue
            if advantage_clip > 0:
                adv = max(-advantage_clip, min(advantage_clip, adv))
            self.samples.append({
                "input_ids": ids,
                "response_mask": mask,
                "advantage": adv,
                "session_id": s["session_id"],
            })
            max_len_seen = max(max_len_seen, len(ids_list))
        print(f"  Dataset: {len(self.samples)} samples ready "
              f"(dropped: {skipped_lowsig} low-|A|<{advantage_min_abs}, "
              f"{skipped_nomask} mask-empty, "
              f"max_tokens={max_len_seen})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            "input_ids": s["input_ids"],
            "response_mask": s["response_mask"],
            "advantage": torch.tensor(s["advantage"], dtype=torch.float32),
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
        out_adv.append(b["advantage"])
    return {
        "input_ids": torch.stack(out_ids),
        "response_mask": torch.stack(out_mask),
        "attention_mask": torch.stack(out_attn),
        "advantage": torch.stack(out_adv),
    }


class LengthGroupedSampler(Sampler):
    """Sort by token length, batch nearby lengths together, shuffle batches.

    Within each batch, sequences have similar length → minimal padding.
    Batch order is shuffled each epoch to avoid ordering bias.
    """

    def __init__(self, dataset: RLDataset, batch_size: int, seed: int = 42):
        self.lengths = [len(s["input_ids"]) for s in dataset.samples]
        self.sorted_indices = sorted(range(len(self.lengths)),
                                     key=lambda i: self.lengths[i])
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        batches = [self.sorted_indices[i:i + self.batch_size]
                   for i in range(0, len(self.sorted_indices), self.batch_size)]
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(batches), generator=g).tolist()
        for idx in order:
            yield from batches[idx]
        self.epoch += 1

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
    advantages = batch["advantage"]                       # [B], already clipped
    attn = batch["attention_mask"]
    n_tok = response_mask.sum().clamp(min=1.0)

    out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    logp = per_token_log_p(out.logits[:, :-1], target_ids)         # [B, T]

    # Reference forward — needed when β_kl > 0.  No grad, bf16, single pass.
    if beta_kl > 0:
        with torch.no_grad():
            if lora_unwrap is not None:
                with lora_unwrap.disable_adapter():
                    ref_logits = model(input_ids=input_ids,
                                       attention_mask=attn,
                                       use_cache=False).logits
            else:
                ref_logits = ref_model(input_ids=input_ids,
                                       attention_mask=attn,
                                       use_cache=False).logits
            logp_ref = per_token_log_p(ref_logits[:, :-1], target_ids)
    else:
        logp_ref = None

    # Policy gradient: linear RWR.  A is per-sequence, broadcast to [B, T].
    A_b = advantages.unsqueeze(1)                        # [B, 1]
    pg_per_tok = -A_b * logp                              # [B, T]
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
        "mean_adv": float(advantages.mean().detach()),
    }


# ── Checkpointing ────────────────────────────────────────

def save_checkpoint(acc: Accelerator, model, optimizer, scheduler,
                    step: int, epoch: int, cfg: TrainConfig,
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
        meta = {"step": step, "epoch": epoch, "config": asdict(cfg)}
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
    """Restore from a checkpoint produced by `save_checkpoint`."""
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
    return meta["step"], meta["epoch"]


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

    `target_modules="all-linear"` covers attention proj, MLP gate/up/down
    and the linear-attention `in_proj_*` / `out_proj` layers so both the
    self-attn and Mamba-style branches of Qwen3.5 hybrid get adapted.
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
    p.add_argument("--max-seq-len", type=int, default=0,
                   help="Deprecated, ignored. All samples kept regardless of length.")
    p.add_argument("--per-device-batch-size", type=int,
                   default=d["per_device_batch_size"].default)
    p.add_argument("--grad-accum-steps", type=int,
                   default=d["grad_accum_steps"].default)
    p.add_argument("--num-epochs", type=int, default=d["num_epochs"].default)
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
    p.add_argument("--advantage-min-abs", type=float,
                   default=d["advantage_min_abs"].default,
                   help="Drop samples with |advantage| < this (set 0 to keep all)")
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
                        cfg.advantage_clip, cfg.advantage_min_abs)
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
    steps_per_epoch = math.ceil(len(loader) / cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.num_epochs
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

    start_step = start_epoch = 0
    if cfg.resume:
        start_step, start_epoch = load_checkpoint(
            acc, model, optimizer, scheduler, cfg.resume, cfg)

    # Training loop
    metrics_log_path = output_dir / "metrics.jsonl"
    metrics_log = open(metrics_log_path, "a") if acc.is_main_process else None
    step = start_step
    t0 = time.time()
    micro_count = 0
    micro_metrics: dict[str, float] = {}

    for epoch in range(start_epoch, cfg.num_epochs):
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
                    line = (f"step={step:5d} ep={epoch} "
                            f"pg={avg['pg_loss']:+.4f} "
                            f"kl={avg['kl_loss']:+.4f} "
                            f"L={avg['total']:+.4f} "
                            f"logp={avg['mean_logp_resp']:+.3f} "
                            f"adv={avg['mean_adv']:+.3f} "
                            f"lr={scheduler.get_last_lr()[0]:.2e} "
                            f"{elapsed:.0f}s")
                    acc.print(line)
                    metrics_log.write(json.dumps({
                        "step": step, "epoch": epoch,
                        **avg, "lr": scheduler.get_last_lr()[0],
                        "elapsed_s": round(elapsed, 1),
                    }) + "\n")
                    metrics_log.flush()
                if step % cfg.save_every == 0:
                    save_checkpoint(acc, model, optimizer, scheduler,
                                    step, epoch, cfg, output_dir)
                micro_metrics = {}
                micro_count = 0

    save_checkpoint(acc, model, optimizer, scheduler, step, epoch, cfg, output_dir)
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
