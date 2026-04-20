#!/usr/bin/env python3
"""Export an RL checkpoint as a single HF model directory.

LoRA path (default in `rl.train`): loads the base model + adapter saved
at `<ckpt>/adapter`, merges via `merge_and_unload`, and writes the
result. Single-process (no accelerate / FSDP needed) since LoRA params
are tiny and the merge is a pure CPU/GPU `weight += alpha·BA` pass.

Full-FT fallback: if `<ckpt>/adapter` does not exist, falls back to the
old FSDP-sharded reload path so the script keeps working for non-LoRA
runs.

Usage:
    python -m rl.export \\
        --ckpt data/<run>/ckpt/step_K \\
        --output data/<run>/merged
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import glob

import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def copy_aux(model_path: str, out_dir: Path) -> None:
    """Copy tokenizer, configs, and preprocessor files from base model."""
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tok.save_pretrained(out_dir)
    for extra in ("chat_template.jinja", "config.json",
                  "preprocessor_config.json", "video_preprocessor_config.json"):
        src = Path(model_path) / extra
        if src.exists():
            shutil.copy(src, out_dir / extra)


def _patch_missing_weights(base_path: str, out_dir: Path) -> None:
    """Copy weights that exist in base but not in merged (visual, mtp).

    AutoModelForCausalLM only loads the text backbone, so visual encoder
    and multi-token-prediction head weights are missing from the merged
    safetensors.  We load them from the base model and append them to a
    new shard so vLLM finds the full model.
    """
    base_keys: dict[str, str] = {}
    for f in sorted(glob.glob(os.path.join(base_path, "*.safetensors"))):
        with open(f, "rb"):
            for k in load_file(f, device="cpu").keys():
                base_keys[k] = f

    merged_keys: set[str] = set()
    for f in sorted(glob.glob(os.path.join(str(out_dir), "*.safetensors"))):
        merged_keys.update(load_file(f, device="cpu").keys())

    missing = {k: v for k, v in base_keys.items() if k not in merged_keys}
    if not missing:
        print("  No missing weights to patch")
        return

    tensors: dict[str, torch.Tensor] = {}
    files_needed = set(missing.values())
    for f in files_needed:
        shard = load_file(f, device="cpu")
        for k in missing:
            if missing[k] == f and k in shard:
                tensors[k] = shard[k]

    patch_path = out_dir / "model_patch.safetensors"
    save_file(tensors, str(patch_path))

    idx_path = out_dir / "model.safetensors.index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        for k in tensors:
            idx["weight_map"][k] = "model_patch.safetensors"
        idx_path.write_text(json.dumps(idx, indent=2))

    print(f"  Patched {len(tensors)} missing weights → {patch_path.name}")


def export_lora(ckpt_dir: Path, out_dir: Path, model_path: str,
                meta: dict) -> None:
    print(f"Loading base from {model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=True)
    print(f"Loading adapter from {ckpt_dir / 'adapter'}")
    model = PeftModel.from_pretrained(base, str(ckpt_dir / "adapter"))
    print("Merging adapter into base")
    merged = model.merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    copy_aux(model_path, out_dir)
    _patch_missing_weights(model_path, out_dir)

    (out_dir / "rl_meta.json").write_text(json.dumps({
        "exported_from": str(ckpt_dir),
        "rl_step": meta["step"],
        "rl_epoch": meta["epoch"],
        "mode": "lora_merged",
    }, indent=2))
    print(f"Exported → {out_dir}")


def export_fsdp(ckpt_dir: Path, out_dir: Path, model_path: str,
                meta: dict) -> None:
    """Legacy FSDP-sharded export (kept for `--no-use-lora` runs)."""
    from accelerate import Accelerator
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions, get_model_state_dict)

    acc = Accelerator(mixed_precision="bf16")
    if acc.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        acc.print(f"Loading shell from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=True)
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
        unwrapped.save_pretrained(out_dir, state_dict=full_sd,
                                  safe_serialization=True)
        copy_aux(model_path, out_dir)
        (out_dir / "rl_meta.json").write_text(json.dumps({
            "exported_from": str(ckpt_dir),
            "rl_step": meta["step"],
            "rl_epoch": meta["epoch"],
            "mode": "fsdp_full",
        }, indent=2))
        acc.print(f"Exported → {out_dir}")
    acc.wait_for_everyone()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    os._exit(0)


def main() -> None:
    p = argparse.ArgumentParser(description="Export RL checkpoint.")
    p.add_argument("--ckpt", required=True, help="Path to step_<N> directory")
    p.add_argument("--output", required=True, help="Output directory")
    args = p.parse_args()

    ckpt_dir = Path(args.ckpt)
    out_dir = Path(args.output)
    meta = json.loads((ckpt_dir / "meta.json").read_text())
    model_path = meta["config"]["model"]

    if (ckpt_dir / "adapter").exists():
        export_lora(ckpt_dir, out_dir, model_path, meta)
    else:
        export_fsdp(ckpt_dir, out_dir, model_path, meta)


if __name__ == "__main__":
    main()
