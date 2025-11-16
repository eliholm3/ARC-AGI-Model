#!/usr/bin/env python3
"""
copy_and_append_predictions.py

- Recursively reads ARC JSON files from --in-dir (unchanged in place).
- For each file, runs the generator to produce predictions for every test example.
- Writes a copy of the JSON to --out-dir, preserving folder structure, with:
    test[i]["prediction"] = <model grid>
    If test[i] lacks "output", create it and set to the same prediction.

Notes
- No PIL/matplotlib; JSON only.
- Supports .pt checkpoints saved as TorchScript or state_dict (auto-detected).
"""

import argparse
import json
import os
import struct
import zlib
import binascii
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import torch

# Your repo modules
from src.training.evaluate_final import build_generator  # builds generator + submodules


# -------------------- checkpoint helpers --------------------
def _try_load_jit(path: str, device: torch.device) -> Optional[torch.nn.Module]:
    try:
        m = torch.jit.load(path, map_location=device)
        return m.eval()
    except Exception:
        return None

def _load_state_dict_into(module: torch.nn.Module, path: str, strict: bool = False) -> Tuple[bool, int]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if isinstance(obj, dict):
        own = module.state_dict()
        filtered = {k: v for k, v in obj.items() if k in own and own[k].shape == obj[k].shape}
        own.update(filtered)
        module.load_state_dict(own, strict=False)
        return True, len(filtered)
    try:
        module.load_state_dict(obj, strict=strict)
        return True, sum(1 for _ in obj)
    except Exception:
        return False, 0

def _autodetect_ckpt_type(path: str, device: torch.device) -> str:
    m = _try_load_jit(path, device)
    if m is not None:
        return "jit"
    try:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            return "state_dict"
    except Exception:
        pass
    return "state_dict"


# -------------------- tensor helpers --------------------
def _plus_one(v: int) -> int:
    # Map ARC {0..9} -> network domain where pad=0, real colors=1..10
    # (keep 0 as 0; shift nonzero by +1)
    return 0 if v == 0 else v + 1

def _grid_to_tensor_plus1(grid: List[List[int]]) -> torch.Tensor:
    H = len(grid)
    W = len(grid[0]) if H else 0
    T = torch.zeros((H, W), dtype=torch.long)
    for r in range(H):
        row = grid[r]
        for c in range(W):
            T[r, c] = _plus_one(int(row[c]))
    return T

def _pad_to_square(t: torch.Tensor, size: int) -> torch.Tensor:
    H, W = t.shape[-2], t.shape[-1]
    if H == size and W == size:
        return t
    P = torch.zeros((size, size), dtype=t.dtype, device=t.device)  # pad value 0
    P[:H, :W] = t
    return P

def _stack_and_pad(grids: List[torch.Tensor], size: int) -> torch.Tensor:
    if not grids:
        # Return empty with shape (0, size, size) so downstream can handle
        return torch.zeros((0, size, size), dtype=torch.long)
    padded = [_pad_to_square(g, size) for g in grids]
    return torch.stack(padded, dim=0)  # (K, size, size)

def _orig_from_plus1(t: torch.Tensor) -> List[List[int]]:
    """Convert HxW tensor where values in {0..10} back to ARC {0..9} (nonzero -> -1)."""
    H, W = int(t.size(0)), int(t.size(1))
    out: List[List[int]] = []
    for r in range(H):
        row: List[int] = []
        for c in range(W):
            v = int(t[r, c])
            row.append(0 if v == 0 else v - 1)
        out.append(row)
    return out


# -------------------- generator caller --------------------
def _call_generator(gen: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Expected forward signature:
      (train_inputs, train_outputs, train_input_masks, train_output_masks, test_inputs, test_input_masks)
    Some TorchScript models may not accept kwargs, so we try kwargs then positionals.
    """
    try:
        return gen(
            train_inputs=batch["train_inputs"],
            train_outputs=batch["train_outputs"],
            train_input_masks=batch["train_input_masks"].bool(),
            train_output_masks=batch["train_output_masks"].bool(),
            test_inputs=batch["test_inputs"],
            test_input_masks=batch["test_input_masks"].bool(),
        )
    except TypeError:
        return gen(
            batch["train_inputs"],
            batch["train_outputs"],
            batch["train_input_masks"].bool(),
            batch["train_output_masks"].bool(),
            batch["test_inputs"],
            batch["test_input_masks"].bool(),
        )


# -------------------- main per-file processing --------------------
def process_one_file(
    in_path: Path,
    out_path: Path,
    generator: torch.nn.Module,
    device: torch.device,
) -> None:
    with in_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    train = obj.get("train", [])
    test = obj.get("test", [])

    # Must have at least one train & test sample to run a meaningful forward
    if len(train) == 0 or len(test) == 0:
        # Just copy the file unchanged
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        return

    # Determine per-sample max square across train inputs/outputs and test inputs
    max_h = 0
    max_w = 0

    for p in train:
        ti = p["input"]
        to = p["output"]
        max_h = max(max_h, len(ti), len(to))
        if len(ti) > 0:
            max_w = max(max_w, len(ti[0]))
        if len(to) > 0:
            max_w = max(max_w, len(to[0]))

    for p in test:
        ti = p["input"]
        max_h = max(max_h, len(ti))
        if len(ti) > 0:
            max_w = max(max_w, len(ti[0]))

    max_size = max(max_h, max_w)

    # Build tensors (+1 encoding for non-zero so pad stays 0)
    train_inputs: List[torch.Tensor] = []
    train_outputs: List[torch.Tensor] = []
    train_input_masks: List[torch.Tensor] = []
    train_output_masks: List[torch.Tensor] = []

    for p in train:
        tinp = _grid_to_tensor_plus1(p["input"])
        tout = _grid_to_tensor_plus1(p["output"])
        train_inputs.append(tinp)
        train_outputs.append(tout)
        train_input_masks.append((tinp != 0).long())
        train_output_masks.append((tout != 0).long())

    test_inputs: List[torch.Tensor] = []
    test_input_masks: List[torch.Tensor] = []
    test_sizes: List[Tuple[int, int]] = []
    for p in test:
        tinp = _grid_to_tensor_plus1(p["input"])
        test_inputs.append(tinp)
        test_input_masks.append((tinp != 0).long())
        H = len(p["input"])
        W = len(p["input"][0]) if H else 0
        test_sizes.append((H, W))

    # Stack + pad to (K, S, S)
    ti_T = _stack_and_pad(train_inputs, max_size)
    to_T = _stack_and_pad(train_outputs, max_size)
    tim_T = _stack_and_pad(train_input_masks, max_size)
    tom_T = _stack_and_pad(train_output_masks, max_size)
    Tei_T = _stack_and_pad(test_inputs, max_size)
    Teim_T = _stack_and_pad(test_input_masks, max_size)

    # Add batch dimension (B=1) -> (1, K, S, S)
    def add_batch(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0) if x.dim() == 3 else x

    batch = {
        "train_inputs": add_batch(ti_T).to(device),
        "train_outputs": add_batch(to_T).to(device),
        "train_input_masks": add_batch(tim_T).to(device),
        "train_output_masks": add_batch(tom_T).to(device),
        "test_inputs": add_batch(Tei_T).to(device),
        "test_input_masks": add_batch(Teim_T).to(device),
    }

    # Forward
    with torch.no_grad():
        out = _call_generator(generator, batch)

    logits = out["logits"]  # (1, Kt, C, S, S) or (Kt, C, S, S)
    if logits.dim() == 5 and logits.size(0) == 1:
        logits = logits.squeeze(0)
    preds = logits.argmax(dim=1)  # (Kt, S, S)

    # Insert predictions into JSON (and create output if missing)
    Kt = preds.size(0)
    for j in range(min(Kt, len(test))):
        H, W = test_sizes[j]
        pred_hw = preds[j, :H, :W].cpu()
        grid = _orig_from_plus1(pred_hw)  # back to ARC {0..9}
        # Append prediction
        test[j]["prediction"] = grid
        # Create output if missing/empty
        if "output" not in test[j] or not test[j]["output"]:
            test[j]["output"] = grid

    # Write modified JSON to out path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, type=str, help="Root folder with ARC JSON files.")
    ap.add_argument("--out-dir", required=True, type=str, help="Where to write copied JSONs with predictions.")
    ap.add_argument("--gen-ckpt", required=True, type=str, help="Path to generator checkpoint (.pt).")
    ap.add_argument("--gen-ckpt-type", type=str, default="auto", choices=["auto", "jit", "state_dict"])
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--limit", type=int, default=0, help="Process only the first N files (0 = all).")
    args = ap.parse_args()

    device = torch.device("cuda") if (args.device == "cuda" and torch.cuda.is_available()) else torch.device("cpu")

    in_root = Path(args.in_dir).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    if not in_root.exists():
        raise FileNotFoundError(f"--in-dir not found: {in_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    # Build/load generator
    gen_type = args.gen_ckpt_type
    if gen_type == "auto":
        gen_type = _autodetect_ckpt_type(args.gen_ckpt, device)

    if gen_type == "jit":
        generator = _try_load_jit(args.gen_ckpt, device)
        if generator is None:
            raise RuntimeError(f"Failed to load TorchScript from {args.gen_ckpt}")
        print(f"[ok] Using weights: jit:{args.gen_ckpt}")
    else:
        generator, *_ = build_generator()
        ok, n = _load_state_dict_into(generator, args.gen_ckpt, strict=False)
        if ok:
            print(f"[ok] Using weights: state_dict:{args.gen_ckpt} (loaded {n} tensors)")
        else:
            print("[warn] Could not load checkpoint weights; using random init.")
    generator.to(device).eval()

    # Walk input tree; mirror structure under out_dir
    files = sorted(p for p in in_root.rglob("*.json") if p.is_file())
    if args.limit and args.limit > 0:
        files = files[:args.limit]

    print(f"[info] Found {len(files)} json file(s) under {in_root}")
    for i, src in enumerate(files, 1):
        rel = src.relative_to(in_root)
        dst = out_root / rel
        try:
            process_one_file(src, dst, generator, device)
            print(f"[{i}/{len(files)}] wrote {dst}")
        except Exception as e:
            print(f"[{i}/{len(files)}] ERROR on {src}: {e}")

if __name__ == "__main__":
    main()
