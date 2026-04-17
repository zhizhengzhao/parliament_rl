#!/usr/bin/env python3
"""Offline RL trainer for Parliament-collected data.

Single-file, multi-GPU FSDP, checkpoint-resumable. Loss is

    L = -E[A · log π(a|s)] / |a|  +  β · KL(π || π_ref)

i.e. the GRPO objective with ratio = 1 (we always train from the same
checkpoint that produced the rollouts; second iteration onwards uses a
proper PPO ratio — controlled by `--use-ratio`). The advantages are
already precomputed by `rl.extract`, so this trainer never needs to do
any rollout — pure offline policy improvement on a JSONL.

Usage (single GPU, smoke test):
    python -m rl.train --data data/<run>/train.jsonl --output ckpts/run1

Multi-GPU FSDP via accelerate:
    accelerate launch --config_file rl/accelerate_fsdp.yaml -m rl.train \
        --data data/<run>/train.jsonl --output ckpts/run1

Resume:
    python -m rl.train ... --resume ckpts/run1/step_200
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "/root/zhizheng/models/Qwen3.5-9B"


# ── Config ────────────────────────────────────────────────

@dataclass
class TrainConfig:
    data: str
    output: str
    model: str = DEFAULT_MODEL
    ref_model: str = ""                 # empty → same as model

    max_seq_len: int = 8192
    per_device_batch_size: int = 1
    grad_accum_steps: int = 16          # effective batch 128 on 8 GPUs
    num_epochs: int = 2                 # multi-epoch squeezes the rollouts
    learning_rate: float = 1e-6         # DeepSeek-R1 default for 9B
    warmup_steps: int = 10
    weight_decay: float = 0.0           # RL fine-tuning doesn't benefit
    grad_clip: float = 1.0              # LLM-RL standard

    beta_kl: float = 0.001              # gentle anchor to base model
    use_ratio: bool = True              # PPO ratio+clip, required for >1 epoch
    advantage_clip: float = 0.0         # 0 = off; advantages already bounded

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

    Drops samples whose tokenized length exceeds `max_seq_len` so we
    never train on a truncated response (which would corrupt the loss).
    """

    def __init__(self, jsonl_path: str, tokenizer, max_seq_len: int,
                 advantage_clip: float = 0.0):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.advantage_clip = advantage_clip

        raw = [json.loads(l) for l in open(jsonl_path)]
        self.samples: list[dict] = []
        skipped = 0
        for s in raw:
            text = tokenizer.apply_chat_template(
                s["messages"], tokenize=False, add_generation_prompt=False)
            ids = tokenizer(text, add_special_tokens=False).input_ids
            if len(ids) > max_seq_len:
                skipped += 1
                continue
            self.samples.append({
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "advantage": float(s["advantage"]),
                "session_id": s["session_id"],
            })
        if skipped:
            print(f"  Dataset: skipped {skipped} oversized samples "
                  f"(>{max_seq_len} tokens)")
        print(f"  Dataset: {len(self.samples)} samples ready")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        adv = s["advantage"]
        if self.advantage_clip > 0:
            adv = max(-self.advantage_clip, min(self.advantage_clip, adv))
        ids = s["input_ids"]
        return {
            "input_ids": ids,
            "response_mask": make_response_mask(ids),
            "advantage": torch.tensor(adv, dtype=torch.float32),
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


# ── Loss ──────────────────────────────────────────────────

def per_token_log_p(logits: torch.Tensor, target_ids: torch.Tensor
                    ) -> torch.Tensor:
    """Gather log p(token_t | <t) for each (B, t) in the response."""
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)


def compute_loss(model, ref_model, batch: dict, beta_kl: float,
                 use_ratio: bool) -> tuple[torch.Tensor, dict]:
    input_ids = batch["input_ids"]
    response_mask = batch["response_mask"][:, 1:]      # align with shifted targets
    target_ids = input_ids[:, 1:]
    advantages = batch["advantage"]
    attn = batch["attention_mask"]
    n_resp = response_mask.sum(dim=-1).clamp(min=1)

    out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    log_p = per_token_log_p(out.logits[:, :-1], target_ids)            # [B, L-1]

    # Sequence-level mean log-prob over response tokens.
    seq_log_p = (log_p * response_mask).sum(dim=-1) / n_resp

    if use_ratio:
        with torch.no_grad():
            ref_out = ref_model(input_ids=input_ids, attention_mask=attn,
                                use_cache=False)
            ref_log_p_tok = per_token_log_p(ref_out.logits[:, :-1], target_ids)
        ratio = torch.exp(seq_log_p - (ref_log_p_tok * response_mask).sum(dim=-1)
                          / n_resp)
        clipped = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
        pg_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    else:
        pg_loss = -(advantages * seq_log_p).mean()

    if beta_kl > 0:
        with torch.no_grad():
            ref_out = ref_model(input_ids=input_ids, attention_mask=attn,
                                use_cache=False)
            ref_log_p = per_token_log_p(ref_out.logits[:, :-1], target_ids)
        # k3 estimator in fp32 with clipped log_ratio: prevents bf16 exp() overflow
        # at the cost of a tiny memory bump that's negligible vs the model itself.
        log_ratio = (log_p - ref_log_p).float()
        log_ratio = torch.clamp(log_ratio, min=-10.0, max=10.0) * response_mask
        kl_per_tok = (torch.exp(log_ratio) - 1 - log_ratio) * response_mask
        kl_loss = (kl_per_tok.sum(dim=-1) / n_resp).mean()
    else:
        kl_loss = torch.zeros((), device=input_ids.device)

    total = pg_loss + beta_kl * kl_loss
    if not torch.isfinite(total):
        # Skip non-finite step: zero the loss so backward is a no-op,
        # but keep gradients flowing through pg_loss so we still update on real signal.
        total = pg_loss
        if not torch.isfinite(total):
            total = torch.zeros((), device=input_ids.device, requires_grad=True)
    return total, {
        "pg_loss": float(pg_loss.detach()),
        "kl_loss": float(kl_loss.detach()),
        "total": float(total.detach()),
        "mean_logp": float(seq_log_p.mean().detach()),
        "mean_adv": float(advantages.mean().detach()),
    }


# ── Checkpointing ────────────────────────────────────────

def save_checkpoint(acc: Accelerator, model, optimizer, scheduler,
                    step: int, epoch: int, cfg: TrainConfig,
                    output_dir: Path) -> None:
    """Save a fully-resumable checkpoint via accelerate's sharded saver.

    If `cfg.keep_last > 0`, prunes older `step_*` directories, keeping
    only the most recent `keep_last` (so a long training run does not
    fill the disk with 50 GB-each shards).
    """
    ckpt_dir = output_dir / f"step_{step}"
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
                import shutil
                shutil.rmtree(old, ignore_errors=True)
                acc.print(f"  pruned old checkpoint → {old}")


def load_checkpoint(acc: Accelerator, resume_dir: str) -> tuple[int, int]:
    acc.load_state(resume_dir)
    meta = json.loads((Path(resume_dir) / "meta.json").read_text())
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


def freeze(model) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


# ── Main ─────────────────────────────────────────────────

def parse_args() -> TrainConfig:
    """CLI defaults mirror the TrainConfig dataclass above."""
    d = TrainConfig.__dataclass_fields__
    p = argparse.ArgumentParser(description="Offline RL trainer (RWR/GRPO).")
    p.add_argument("--data", required=True, help="JSONL from rl.extract")
    p.add_argument("--output", required=True, help="Checkpoint directory")
    p.add_argument("--model", default=d["model"].default)
    p.add_argument("--ref-model", default="")
    p.add_argument("--max-seq-len", type=int, default=d["max_seq_len"].default)
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
    p.add_argument("--beta-kl", type=float, default=d["beta_kl"].default)
    p.add_argument("--use-ratio", action=argparse.BooleanOptionalAction,
                   default=d["use_ratio"].default,
                   help="PPO-style ratio + clip (default on; disable with --no-use-ratio)")
    p.add_argument("--advantage-clip", type=float,
                   default=d["advantage_clip"].default)
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

    acc = Accelerator(gradient_accumulation_steps=cfg.grad_accum_steps,
                      mixed_precision="bf16")

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
    dataset = RLDataset(cfg.data, tokenizer, cfg.max_seq_len, cfg.advantage_clip)
    loader = DataLoader(
        dataset, batch_size=cfg.per_device_batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, pad_id),
        num_workers=2, pin_memory=True,
    )

    # Models
    acc.print(f"Loading policy from {cfg.model}")
    model = load_model(cfg.model)
    model.gradient_checkpointing_enable()
    if cfg.beta_kl > 0 or cfg.use_ratio:
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
        # 9B BF16 ≈ 18 GB; comfortable on 80 GB cards alongside the
        # FSDP-sharded policy (≈ 1 GB/GPU). Avoids FSDP2's "model+optim
        # must prepare together" constraint and dodges synchronization
        # overhead for a model that never updates.
        ref_model = ref_model.to(acc.device)

    start_step = start_epoch = 0
    if cfg.resume:
        start_step, start_epoch = load_checkpoint(acc, cfg.resume)

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
                                       cfg.beta_kl, cfg.use_ratio)
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
                            f"logp={avg['mean_logp']:+.3f} "
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
