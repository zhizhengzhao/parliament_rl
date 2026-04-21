#!/usr/bin/env python3
"""Export an RL checkpoint as a single vLLM-loadable HF model folder.

LoRA path (default): load base + adapter, `merge_and_unload`, save.
Single-process — LoRA merge is tiny (just `W += alpha · BA`).

Full-FT fallback: if `<ckpt>/adapter` is missing, gather FSDP-sharded
weights on rank 0 and save (legacy, rarely used).

Qwen3.5 is a multimodal model: ``AutoModelForCausalLM`` only loads the
text backbone, so we additionally copy the base's ``config.json`` +
preprocessor configs and patch in any missing visual/mtp weights so the
merged folder stays drop-in replaceable for the original base.

Usage:
    python -m rl.export --ckpt data/<run>/ckpt/step_K \\
        --output data/<run>/merged
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

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
    """Copy weights present in base but missing from merged (visual, mtp)."""
    base_keys: dict[str, str] = {}
    for f in sorted(glob.glob(os.path.join(base_path, "*.safetensors"))):
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
    for f in set(missing.values()):
        shard = load_file(f, device="cpu")
        for k, src in missing.items():
            if src == f and k in shard:
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
