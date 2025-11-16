#!/usr/bin/env python3
"""
predict_and_viz.py

Same name as before, but this version ONLY writes predictions back into
the ARC JSON files (no images). Each test pair gains:
  - "prediction": the predicted grid (list[list[int]])
  - "is_exact": True/False (only if ground-truth "output" exists)
  - "prediction_meta": small info block

Usage:
  PYTHONPATH=. python -u predict_and_viz.py \
    --dir ./Curve-BallDatasetTasks \
    --gen-ckpt ./src/generator.pt \
    --device auto \
    --limit 0
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import torch

# Your repo entry point that builds an uninitialized generator module
from src.training.evaluate_final import build_generator


# ---------------- checkpoint helpers ----------------
def _try_load_jit(path: str, device: torch.device) -> Optional[torch.nn.Module]:
    try:
        m = torch.jit.load(path, map_location=device)
        return m.eval()
    except Exception:
        return None

def _load_state_dict_into(module: torch.nn.Module, path: str, strict: bool = False) -> Tuple[bool, int]:
    """
    Loads weights from a state_dict-like checkpoint into 'module'.
    Returns (ok, num_tensors_loaded).
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if isinstance(obj, dict):
        own = module.state_dict()
        # greedy-filter by name & shape to avoid strict mismatches
        filt = {k: v for k, v in obj.items() if k in own and own[k].shape == v.shape}
        own.update(filt)
        module.load_state_dict(own, strict=False)
        return True, len(filt)
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


# ---------------- tensor helpers (+1 encoding) ----------------
def _plus_one(v: int) -> int:
    # shift original values up by +1 so pad=0 is reserved
    return v + 1

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
    P = torch.zeros((size, size), dtype=t.dtype, device=t.device)  # pad = 0
    P[:H, :W] = t
    return P

def _stack_and_pad(grids: List[torch.Tensor], size: int, device: Optional[torch.device]=None) -> torch.Tensor:
    if not grids:
        return torch.zeros((0, size, size), dtype=torch.long, device=device)
    padded = [_pad_to_square(g.to(device) if device else g, size) for g in grids]
    return torch.stack(padded, dim=0)

def _orig_from_plus1(t: torch.Tensor) -> List[List[int]]:
    # Convert tensor (pad=0) back to original ARC ints.
    H, W = int(t.size(0)), int(t.size(1))
    out: List[List[int]] = []
    for r in range(H):
        row: List[int] = []
        for c in range(W):
            v = int(t[r, c])
            row.append(0 if v == 0 else (v - 1))
        row = row[:W]
        out.append(row)
    return out


# ---------------- generator caller ----------------
def _call_generator(gen: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Expected forward signature:
      (train_inputs, train_outputs, train_input_masks, train_output_masks, test_inputs, test_input_masks)
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
        # TorchScript / positional-only fallback
        return gen(
            batch["train_inputs"],
            batch["train_outputs"],
            batch["train_input_masks"].bool(),
            batch["train_output_masks"].bool(),
            batch["test_inputs"],
            batch["test_input_masks"].bool(),
        )


# ---------------- per-file processing ----------------
def _grid_hw(grid: List[List[int]]) -> Tuple[int,int]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w

def _square_size_from_sample(sample: Dict[str, Any]) -> int:
    """
    Compute a single square size S for the file, from all present grids.
    """
    max_h = 0
    max_w = 0

    for p in sample.get("train", []):
        if "input" in p:
            h, w = _grid_hw(p["input"])
            max_h, max_w = max(max_h, h), max(max_w, w)
        if "output" in p:
            h, w = _grid_hw(p["output"])
            max_h, max_w = max(max_h, h), max(max_w, w)

    for p in sample.get("test", []):
        if "input" in p:
            h, w = _grid_hw(p["input"])
            max_h, max_w = max(max_h, h), max(max_w, w)
        if "output" in p:
            h, w = _grid_hw(p["output"])
            max_h, max_w = max(max_h, h), max(max_w, w)

    return max(max_h, max_w)

def _build_batch_from_json(sample: Dict[str, Any], device: torch.device):
    """
    Builds a B=1 batch dict with +1 encoding and square padding.
    Handles missing 'output' in train/test gracefully.
    """
    S = _square_size_from_sample(sample)

    train_inputs: List[torch.Tensor] = []
    train_outputs: List[torch.Tensor] = []
    train_input_masks: List[torch.Tensor] = []
    train_output_masks: List[torch.Tensor] = []

    for p in sample.get("train", []):
        ti = _grid_to_tensor_plus1(p["input"])
        train_inputs.append(ti)
        train_input_masks.append((ti != 0).long())

        if "output" in p:
            to = _grid_to_tensor_plus1(p["output"])
            tom = (to != 0).long()
        else:
            to = torch.zeros_like(ti)
            tom = torch.zeros_like(ti)
        train_outputs.append(to)
        train_output_masks.append(tom)

    test_inputs: List[torch.Tensor] = []
    test_input_masks: List[torch.Tensor] = []
    test_sizes: List[Tuple[int,int]] = []
    test_outputs_opt: List[Optional[torch.Tensor]] = []

    for p in sample.get("test", []):
        ti = _grid_to_tensor_plus1(p["input"])
        test_inputs.append(ti)
        test_input_masks.append((ti != 0).long())

        if "output" in p:
            H, W = _grid_hw(p["output"])
            test_outputs_opt.append(_grid_to_tensor_plus1(p["output"]))
        else:
            H, W = _grid_hw(p["input"])  # use input dims if no output provided
            test_outputs_opt.append(None)
        test_sizes.append((H, W))

    # pad/stack and add batch dim
    def add_batch(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0) if x.dim() == 3 else x

    ti_T   = _stack_and_pad(train_inputs, S, device)
    to_T   = _stack_and_pad(train_outputs, S, device)
    tim_T  = _stack_and_pad(train_input_masks, S, device)
    tom_T  = _stack_and_pad(train_output_masks, S, device)
    Tei_T  = _stack_and_pad(test_inputs, S, device)
    Teim_T = _stack_and_pad(test_input_masks, S, device)

    batch = {
        "train_inputs": add_batch(ti_T),
        "train_outputs": add_batch(to_T),
        "train_input_masks": add_batch(tim_T),
        "train_output_masks": add_batch(tom_T),
        "test_inputs": add_batch(Tei_T),
        "test_input_masks": add_batch(Teim_T),
    }
    return batch, test_sizes, test_outputs_opt


def _write_predictions_back(
    json_path: Path,
    sample: Dict[str, Any],
    preds: torch.Tensor,
    test_sizes: List[Tuple[int,int]],
    test_outputs_opt: List[Optional[torch.Tensor]],
    used_fallback: bool,
    model_id: str,
):
    """
    Updates sample['test'][i] with 'prediction' (+ 'is_exact' if gt exists),
    then writes back atomically to json_path.
    """
    for j, p in enumerate(sample.get("test", [])):
        H, W = test_sizes[j]
        pred_j = preds[j, :H, :W].cpu()
        pred_grid = _orig_from_plus1(pred_j)
        p["prediction"] = pred_grid

        if test_outputs_opt[j] is not None:
            gt = test_outputs_opt[j][:H, :W].cpu()
            p["is_exact"] = bool(torch.equal(pred_j, gt))
        else:
            if "is_exact" in p:
                del p["is_exact"]

        meta = {"source": "fallback_input_copy" if used_fallback else "model_logits_argmax"}
        if model_id:
            meta["model_id"] = model_id
        p["prediction_meta"] = meta

    tmp = json_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2)
    tmp.replace(json_path)


def process_one_file(
    json_path: Path,
    generator: torch.nn.Module,
    device: torch.device,
    model_id: str,
) -> Tuple[bool, str]:
    try:
        with json_path.open("r", encoding="utf-8") as f:
            sample = json.load(f)
    except Exception as e:
        return False, f"[skip] failed to read {json_path}: {e}"

    if not isinstance(sample, dict) or "test" not in sample or not sample["test"]:
        return False, f"[skip] no test pairs: {json_path.name}"

    batch, test_sizes, test_outputs_opt = _build_batch_from_json(sample, device)

    # forward
    try:
        with torch.no_grad():
            out = _call_generator(generator, batch)
        logits = out["logits"]  # (1,K,C,S,S) or (K,C,S,S)
        if logits.dim() == 5 and logits.size(0) == 1:
            logits = logits.squeeze(0)
        preds = logits.argmax(dim=1)  # (K,S,S)
        used_fallback = False
    except Exception as e:
        # fallback: copy test inputs
        preds = batch["test_inputs"].squeeze(0).clone()
        used_fallback = True

    _write_predictions_back(json_path, sample, preds, test_sizes, test_outputs_opt, used_fallback, model_id)
    tag = "fallback" if used_fallback else "ok"
    return True, f"[{tag}] {json_path.name}"


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, type=str, help="Root folder containing ARC JSON files (recursively).")
    ap.add_argument("--gen-ckpt", required=True, type=str, help="Path to generator checkpoint (.pt).")
    ap.add_argument("--gen-ckpt-type", type=str, default="auto", choices=["auto","jit","state_dict"])
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--limit", type=int, default=0, help="Only process first N files (0 = all).")
    ap.add_argument("--model-id", type=str, default="", help="Optional label to embed in prediction_meta.")
    args = ap.parse_args()

    device = torch.device("cuda") if (args.device == "cuda" and torch.cuda.is_available()) else torch.device("cpu")

    root = Path(args.dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"--dir not found: {root}")

    # Build/load generator
    print(f"[detect] checkpoint: {args.gen_ckpt}")
    ctype = args.gen_ckpt_type if args.gen_ckpt_type != "auto" else _autodetect_ckpt_type(args.gen_ckpt, device)
    print(f"[detect] interpreted type: {ctype}")

    if ctype == "jit":
        gen = _try_load_jit(args.gen_ckpt, device)
        if gen is None:
            raise RuntimeError(f"Failed to load TorchScript from {args.gen_ckpt}")
        print(f"[ok] Using weights: jit:{args.gen_ckpt}")
    else:
        gen, *_ = build_generator()
        ok, n = _load_state_dict_into(gen, args.gen_ckpt, strict=False)
        if ok:
            print(f"[info] Loaded a filtered subset of weights ({n} tensors) from state_dict.")
            print(f"[ok] Using weights: state_dict:{args.gen_ckpt}")
        else:
            print("[warn] Could not load checkpoint strictly; continuing without weights.")
    gen.to(device).eval()

    # Find JSONs
    files = sorted(p for p in root.rglob("*.json") if p.is_file())
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    processed = 0
    fallback_used = 0
    for idx, p in enumerate(files, 1):
        ok, msg = process_one_file(p, gen, device, args.model_id)
        print(msg)
        if ok:
            processed += 1
            if msg.startswith("[fallback]"):
                fallback_used += 1

    print(f"\n[done] updated {processed} JSON file(s). fallbacks used: {fallback_used}\n")


if __name__ == "__main__":
    main()
