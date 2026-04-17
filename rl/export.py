#!/usr/bin/env python3
"""Export an FSDP-sharded RL checkpoint as a single HF model directory.

Loads the sharded `pytorch_model_fsdp_0/` written by `accelerate.save_state`,
gathers the full unsharded BF16 state dict on rank 0, and saves it next to a
copy of the tokenizer + config so the result is ready for `from_pretrained`
(and thus for vLLM rollout in the next Parliament iteration).

Usage (multi-process — must match the world size that wrote the checkpoint):
    accelerate launch --config_file rl/accelerate_fsdp.yaml --num_processes 8 \
        -m rl.export --ckpt data/<run>/ckpt_run1/step_12 \
                     --output data/<run>/ckpt_run1/step_12_merged
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_model_state_dict)
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser(description="Export sharded RL checkpoint.")
    p.add_argument("--ckpt", required=True,
                   help="Path to step_<N> directory")
    p.add_argument("--output", required=True,
                   help="Output directory (HF format)")
    args = p.parse_args()

    ckpt_dir = Path(args.ckpt)
    out_dir = Path(args.output)
    meta = json.loads((ckpt_dir / "meta.json").read_text())
    model_path = meta["config"]["model"]

    acc = Accelerator(mixed_precision="bf16")
    if acc.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        acc.print(f"Loading shell from {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=True)
    # FSDP2 demands prepare(model, optim) as a pair, so we build a dummy
    # AdamW with no learning so it never touches anything during the load.
    dummy_optim = torch.optim.AdamW(model.parameters(), lr=0.0)
    model, _ = acc.prepare(model, dummy_optim)

    acc.print(f"Loading sharded weights from {ckpt_dir}")
    acc.load_state(str(ckpt_dir))

    acc.print("Gathering full state dict on rank 0")
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    full_sd = get_model_state_dict(model, options=options)

    acc.wait_for_everyone()
    if acc.is_main_process:
        unwrapped = acc.unwrap_model(model)
        unwrapped.save_pretrained(
            out_dir, state_dict=full_sd, safe_serialization=True)
        # Copy tokenizer + chat template + any auxiliary files.
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tok.save_pretrained(out_dir)
        for extra in ("chat_template.jinja",):
            src = Path(model_path) / extra
            if src.exists():
                shutil.copy(src, out_dir / extra)
        (out_dir / "rl_meta.json").write_text(json.dumps({
            "exported_from": str(ckpt_dir),
            "rl_step": meta["step"],
            "rl_epoch": meta["epoch"],
        }, indent=2))
        acc.print(f"Exported → {out_dir}")

    acc.wait_for_everyone()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
