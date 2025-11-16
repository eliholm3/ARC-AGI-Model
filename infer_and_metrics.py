#!/usr/bin/env python3
"""
infer_and_metrics.py

Runs inference with your ARC generator over a dataset and prints:
  • number of examples that are fully correct (exact match)
  • % of pixels correct (global)
  • % of pixels correct averaged equally per example (weighted to each example)
Optionally saves PNG visualizations of INPUT / PRED / OUTPUT.

Examples:
  PYTHONPATH=. python -u infer_and_metrics.py \
    --data-dir ./src/data_pipeline/ARC_data/data/training \
    --batch-size 1 --shuffle False --limit 0 --device auto \
    --gen-ckpt ./src/generator.pt \
    --viz-out-dir ./viz --viz-scale 20

Notes:
- If a test pair has no real ground-truth (GT) output (i.e., all pad/zeros),
  we save a 2-panel PNG (Input | Pred) and suffix filename with _nogt.
- If GT exists and prediction is exact, filename gets _correct.
"""

import argparse
import os
import zlib
import struct
import binascii
from pathlib import Path
from typing import Tuple, Optional, List

import torch

# ==== Import your repo modules ====
from src.training.evaluate_final import build_generator  # builds generator + submodules
from src.data_pipeline.dataloader import ARCDataModule

DEVICE_DEFAULT = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- PNG writer (no PIL/matplotlib required) ----------------
def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data)) +
        tag +
        data +
        struct.pack(">I", binascii.crc32(tag + data) & 0xFFFFFFFF)
    )

def write_png_rgb(path: Path, rgb: List[List[Tuple[int, int, int]]]) -> None:
    """
    rgb: H x W list of rows, each row is list of (R,G,B) uint8 tuples.
    """
    height = len(rgb)
    width = len(rgb[0]) if height > 0 else 0

    # Build raw scanlines (filter type 0 per row)
    raw = bytearray()
    for r in range(height):
        raw.append(0)  # no filter
        row = rgb[r]
        for (R, G, B) in row:
            raw.extend(bytes((R, G, B)))

    # PNG file
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB",
                       width, height, 8, 2, 0, 0, 0)  # 8-bit, truecolor
    idat = zlib.compress(bytes(raw), level=9)
    with path.open("wb") as f:
        f.write(sig)
        f.write(_png_chunk(b"IHDR", ihdr))
        f.write(_png_chunk(b"IDAT", idat))
        f.write(_png_chunk(b"IEND", b""))


# ---------------- ARC color mapping (canonical) ----------------
# ARC color indices (0..9) -> RGB
# 0: black, 1: blue, 2: red, 3: green, 4: yellow,
# 5: gray, 6: magenta/pink, 7: orange, 8: cyan, 9: brown
ARC_RGB = [
    (0, 0, 0),        # 0 black
    (0, 0, 255),      # 1 blue
    (255, 0, 0),      # 2 red
    (0, 255, 0),      # 3 green
    (255, 255, 0),    # 4 yellow
    (128, 128, 128),  # 5 gray
    (255, 0, 255),    # 6 magenta/pink
    (255, 165, 0),    # 7 orange
    (0, 255, 255),    # 8 cyan
    (165, 42, 42),    # 9 brown
]

def _fallback_color(v: int) -> Tuple[int, int, int]:
    # Deterministic hash -> RGB for out-of-range values
    x = (int(v) * 2654435761) & 0xFFFFFFFF
    return ((x >> 16) & 0xFF, (x >> 8) & 0xFF, x & 0xFF)

def _label_to_rgb(v: int, shifted: bool) -> Tuple[int, int, int]:
    """
    Map a grid label to RGB.

    shifted=True  -> labels are {0=pad, 1..10 = ARC colors 0..9}
    shifted=False -> labels are {0..9 = ARC colors 0..9}
    """
    iv = int(v)
    if shifted:
        if iv <= 0:
            return ARC_RGB[0]  # pad or 0 -> black
        idx = iv - 1  # 1..10 -> 0..9
    else:
        idx = iv      # 0..9 -> 0..9

    if 0 <= idx <= 9:
        return ARC_RGB[idx]
    return _fallback_color(iv)

def _grid_to_rgb(grid: torch.Tensor, scale: int, shifted: bool) -> List[List[Tuple[int, int, int]]]:
    """
    grid: (H, W) long tensor of ints
    returns scaled RGB rows (H*scale x W*scale)
    """
    H, W = int(grid.size(0)), int(grid.size(1))
    rows: List[List[Tuple[int, int, int]]] = []
    for r in range(H):
        row_colors = [_label_to_rgb(int(grid[r, c]), shifted) for c in range(W)]
        # horizontal scale
        scaled_row = []
        for (R, G, B) in row_colors:
            scaled_row.extend([(R, G, B)] * scale)
        # vertical scale
        for _ in range(scale):
            rows.append(list(scaled_row))
    return rows


# ---------------- Checkpoint helpers ----------------
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


# ---------------- Helpers ----------------
def _get_orig_size_for_index(orig_size_tensor: torch.Tensor, j: int) -> Tuple[int, int]:
    """
    Handles both shapes:
      - (2,) => same H,W for all test outputs
      - (K_test, 2) => per-example H,W
    """
    if orig_size_tensor.dim() == 1:
        h, w = map(int, orig_size_tensor.tolist())
        return h, w
    elif orig_size_tensor.dim() == 2:
        h, w = map(int, orig_size_tensor[j].tolist())
        return h, w
    else:
        raise ValueError(f"Unexpected test_original_size shape: {tuple(orig_size_tensor.shape)}")

def _clean_name(s: str) -> str:
    s = s.replace("\\", "/")
    s = s.strip("/").replace("/", "__")
    if not s:
        s = "task"
    return s

def _batch_task_name(batch: dict, fallback_index: int) -> str:
    for key in ["id", "task_id", "name", "file_name", "source", "task"]:
        if key in batch and isinstance(batch[key], str) and batch[key].strip():
            return _clean_name(batch[key])
    return f"task_{fallback_index:05d}"


def run(args):
    # ---- device ----
    device = (
        torch.device("cuda") if (args.device == "cuda" and torch.cuda.is_available())
        else torch.device("cpu")
    )

    # ---- build / load model ----
    loaded_from = None
    gen_type = args.gen_ckpt_type
    if args.gen_ckpt and gen_type == "auto":
        gen_type = _autodetect_ckpt_type(args.gen_ckpt, device)

    if gen_type == "jit":
        if not args.gen_ckpt or not os.path.exists(args.gen_ckpt):
            raise FileNotFoundError(f"--gen-ckpt '{args.gen_ckpt}' not found for jit")
        generator = _try_load_jit(args.gen_ckpt, device)
        if generator is None:
            raise RuntimeError(f"Failed to load TorchScript from {args.gen_ckpt}")
        loaded_from = f"jit:{args.gen_ckpt}"
    else:
        generator, *_rest = build_generator()
        if args.gen_ckpt:
            if os.path.exists(args.gen_ckpt):
                ok, n = _load_state_dict_into(generator, args.gen_ckpt, strict=False)
                if ok:
                    print(f"[info] Loaded a filtered subset of weights ({n} tensors) from state_dict.")
                    loaded_from = f"state_dict:{args.gen_ckpt}"
                else:
                    print(f"[warn] Could not load checkpoint strictly; continuing without weights.")
            else:
                print(f"[warn] Checkpoint not found: {args.gen_ckpt} (continuing without)")

    generator.to(device).eval()
    if loaded_from:
        print(f"[ok] Using weights: {loaded_from}")

    # ---- data ----
    dm = ARCDataModule(
        dir_path=Path(args.data_dir),
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=0,
        pin_memory=False,
        pad_value=0,
    ).prepare()
    loader = dm.get_loader()

    if args.limit and args.limit > 0:
        dm.dataset.data = dm.dataset.data[: args.limit]

    # ---- metrics ----
    total_examples = 0               # includes only those with GT
    exact_match_count = 0
    total_pixels_global = 0
    total_pixels_correct_global = 0
    per_example_accs = []

    # ---- viz dir ----
    viz_dir = None
    if args.viz_out_dir:
        viz_dir = Path(args.viz_out_dir).expanduser().resolve()
        viz_dir.mkdir(parents=True, exist_ok=True)
        print(f"[viz] saving to: {viz_dir}")

    running_task_idx = 0

    with torch.no_grad():
        for batch in loader:
            # move to device
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)

            # forward
            try:
                out = generator(
                    train_inputs=batch["train_inputs"],
                    train_outputs=batch["train_outputs"],
                    train_input_masks=batch["train_input_masks"].bool(),
                    train_output_masks=batch["train_output_masks"].bool(),
                    test_inputs=batch["test_inputs"],
                    test_input_masks=batch["test_input_masks"].bool(),
                )
            except TypeError:
                # TorchScript positional fallback
                out = generator(
                    batch["train_inputs"],
                    batch["train_outputs"],
                    batch["train_input_masks"].bool(),
                    batch["train_output_masks"].bool(),
                    batch["test_inputs"],
                    batch["test_input_masks"].bool(),
                )

            logits = out["logits"]  # (B, K, C, H, W) or (K, C, H, W)
            if logits.dim() == 5 and logits.size(0) == 1:
                logits = logits.squeeze(0)

            preds = logits.argmax(dim=1)  # (K, H, W)
            targets = batch["test_outputs"]  # (K, H, W)

            task_name = _batch_task_name(batch, running_task_idx)

            K_test = targets.size(0)
            for j in range(K_test):
                orig_h, orig_w = _get_orig_size_for_index(batch["test_original_size"], j)
                pred_j = preds[j, :orig_h, :orig_w]
                targ_j = targets[j, :orig_h, :orig_w]
                inp_j  = batch["test_inputs"][j, :orig_h, :orig_w]

                # Determine if GT exists: any non-zero (non-pad) pixel in target crop
                has_gt = bool((targ_j != 0).any().item())

                # Only score metrics when GT exists
                if has_gt:
                    is_exact = bool((pred_j == targ_j).all().item())
                    correct = int((pred_j == targ_j).sum().item())
                    total   = int(orig_h * orig_w)
                    acc     = (correct / total) if total > 0 else 0.0

                    per_example_accs.append(acc)
                    total_pixels_correct_global += correct
                    total_pixels_global += total
                    total_examples += 1
                    if is_exact:
                        exact_match_count += 1
                else:
                    is_exact = None  # for naming only

                # ---- Visualization ----
                if viz_dir is not None:
                    # Build panels: Input | Pred | (Output if GT exists)
                    panels: List[torch.Tensor] = [inp_j, pred_j]
                    if has_gt:
                        panels.append(targ_j)

                    # Convert each panel to scaled RGB grid using correct palette + shift
                    scaled_panels = [_grid_to_rgb(p, args.viz_scale, args.labels_are_shifted)
                                     for p in panels]

                    # All panels share same H,W after crop; concatenate horizontally
                    Hs = [len(sp) for sp in scaled_panels]
                    Ws = [len(sp[0]) if sp else 0 for sp in scaled_panels]
                    H = Hs[0] if Hs else 0
                    total_W = sum(Ws)

                    # Build empty canvas
                    canvas: List[List[Tuple[int, int, int]]] = []
                    for _r in range(H):
                        canvas.append([(0, 0, 0)] * total_W)

                    # Blit each panel
                    x0 = 0
                    for sp in scaled_panels:
                        h = len(sp)
                        w = len(sp[0]) if sp else 0
                        for r in range(h):
                            canvas[r][x0:x0+w] = sp[r]
                        x0 += w

                    # File name
                    suffix = "_correct" if (has_gt and is_exact) else ("_nogt" if not has_gt else "")
                    fname = f"{task_name}_test{j:02d}{suffix}.png"

                    # Write PNG
                    write_png_rgb(viz_dir / fname, canvas)

            running_task_idx += 1

    pct_pixels_global = (
        (total_pixels_correct_global / total_pixels_global) if total_pixels_global > 0 else 0.0
    )
    pct_pixels_weighted_by_example = (
        (sum(per_example_accs) / len(per_example_accs)) if per_example_accs else 0.0
    )

    print("\n================ Inference Metrics ================")
    print(f"Fully-correct (exact matches): {exact_match_count} / {total_examples}")
    print(f"% pixels correct (global):     {100.0 * pct_pixels_global:.2f}%")
    print(f"% pixels correct (weighted):   {100.0 * pct_pixels_weighted_by_example:.2f}%")
    if args.model_id:
        print(f"Model ID: {args.model_id}")
    print("  - Only examples with real GT are scored in the metrics above.")
    print("  - Files with no GT are saved with only Input|Pred panels and '_nogt' suffix.")
    print("==================================================\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="./src/data_pipeline/ARC_data/data/training",
                    help="Path to ARC 'training' folder (root of *.json tasks)")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--shuffle", type=lambda x: str(x).lower() in ['true','1','yes'], default=False)
    ap.add_argument("--limit", type=int, default=0, help="Limit number of ARC tasks (0 = no limit)")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--gen-ckpt", type=str, default="",
                    help="Path to a generator checkpoint (.pt). TorchScript or state_dict.")
    ap.add_argument("--gen-ckpt-type", type=str, default="auto",
                    choices=["auto", "jit", "state_dict"],
                    help="How to interpret --gen-ckpt. 'auto' tries JIT first, then state_dict.")
    ap.add_argument("--model-id", type=str, default="", help="Optional label printed with metrics.")
    # Visualization
    ap.add_argument("--viz-out-dir", type=str, default="",
                    help="Directory to save PNG visualizations (empty = disable).")
    ap.add_argument("--viz-scale", type=int, default=20, help="Pixel scaling for PNGs.")
    # Label shift control (default True for your +1 pipeline)
    ap.add_argument("--labels-are-shifted",
                    type=lambda x: str(x).lower() in ['true', '1', 'yes'],
                    default=True,
                    help="If True, labels are {0=pad, 1..10=ARC colors 0..9}. If False, labels are 0..9.")
    args = ap.parse_args()
    run(args)
