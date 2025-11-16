
## PLEASE READ THE COMMENTS BELOW TO UNDERSTAND HOW TO USE THIS SCRIPT

## Example: visualize a directory of ARC-style task JSONs (optionally with predictions)
## change to index_base 0 if your data is 0-indexed
## task_dir should be a directory of ARC-style task JSONs
## pred_dir should be a directory of prediction JSONs
## out_dir should be a directory to save the images
# python visualize_groups.py \
#   --task_dir Curve-BallDatasetTasks \
#   --out_dir Curve-BallDatasetTasks_Images \
#   --index_base 1 \
#   --pred_dir Curve-BallDatasetTasks_Preds

## Example: visualize a single ARC-style task JSON (optionally with predictions)
## change to index_base 0 if your data is 0-indexed
## task_file should be a path to an ARC-style task JSON
## pred_file should be a path to a prediction JSON
## out_dir should be a directory to save the images
# python visualize_groups.py \
#   --task_file Curve-BallDatasetTasks/example06.json \
#   --out_dir Curve-BallDatasetTasks_Images \
#   --index_base 1 \
#   --pred_file Curve-BallDatasetTasks_Preds/example06.json


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ========= EDIT THESE IF YOU WANT DIFFERENT FILENAMES =========
TASK_FILE = "Curve-BallDatasetTasks/example06.json"
OUT_DIR   = "Curve-BallDatasetTasks_Images"
# ==============================================================

# Rendering params
CELL_SIZE  = 32    # pixels per ARC cell
LABEL_H    = 30    # label strip height in pixels
COL_GAP    = 24    # horizontal gap between Guess|Real|Diff
ROW_GAP    = 18    # vertical gap between rows (pairs) inside a group image
BG_COLOR   = (255, 255, 255)
GRID_COLOR = (0, 0, 0)

# ARC palette 0..9
ARC_COLORS = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
]
def _hex_to_rgb(h: str) -> Tuple[int,int,int]:
    h = h.lstrip("#"); return tuple(int(h[i:i+2],16) for i in (0,2,4))
PALETTE_RGB = np.array([_hex_to_rgb(h) for h in ARC_COLORS], dtype=np.uint8)

# Index base for grids: 0 (values 0..9) or 1 (values 1..10).
# This is configured via the --index_base CLI flag in main().
INDEX_BASE = 1

# -------- helpers: grids & validation --------
def _is_grid(x: Any) -> bool:
    return (
        isinstance(x, list) and x and
        all(isinstance(r, list) and r for r in x) and
        len({len(r) for r in x}) == 1 and             # rectangular
        all(all(isinstance(v, (int, np.integer)) for v in r) for r in x)
    )

def _validate_grid(g: List[List[int]], name="grid"):
    if not _is_grid(g):
        raise ValueError(f"{name} must be a non-empty rectangular 2D list of ints.")
    for y, row in enumerate(g):
        for x, v in enumerate(row):
            v_int = int(v)
            if INDEX_BASE == 0:
                # 0-based: allowed range 0..len(ARC_COLORS)-1
                if not (0 <= v_int <= len(ARC_COLORS) - 1):
                    raise ValueError(
                        f"{name}[{y}][{x}]={v} not in [0..{len(ARC_COLORS)-1}] for 0-based palette."
                    )
            else:
                # 1-based: allowed range 1..len(ARC_COLORS)
                if not (1 <= v_int <= len(ARC_COLORS)):
                    raise ValueError(
                        f"{name}[{y}][{x}]={v} not in [1..{len(ARC_COLORS)}] for 1-based palette."
                    )

def _normalize_grid(g: List[List[int]]) -> List[List[int]]:
    """Pad ragged rows (shouldn't happen if validated) and return a copy."""
    w = max(len(r) for r in g)
    return [r + [0]*(w - len(r)) for r in g]

def _pad_to(g: np.ndarray, H: int, W: int, fill: int = -1) -> np.ndarray:
    out = np.full((H, W), fill, dtype=np.int16)
    out[:g.shape[0], :g.shape[1]] = g
    return out

 # -------- loading: legacy group format (kept for backward compatibility) --------
def _load_groups(path: Path) -> Dict[str, List[List[List[int]]]]:
    """Legacy loader: group_id -> [grid, grid, ...]."""
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must be a JSON object mapping group_id -> list-of-2D-grids.")
    groups: Dict[str, List[List[List[int]]]] = {}
    for gid, val in obj.items():
        if isinstance(val, list) and val:
            # case 1: list of grids
            if all(_is_grid(g) for g in val):  # ["gid": [grid, grid, ...]]
                groups[gid] = val
                continue
            # case 2: val is [grid] (with exactly one grid) OR nested accidental wrapper
            if len(val) == 1 and _is_grid(val[0]):
                groups[gid] = [val[0]]
                continue
        # case 3: val itself is a single grid [[...], ...]
        if _is_grid(val):
            groups[gid] = [val]
            continue
        raise ValueError(f"Group '{gid}' has an invalid value; expected a list of 2D grids.")
    return groups

# -------- rasterization --------
def _grid_to_image(grid: List[List[int]], cell=CELL_SIZE, draw_grid=True) -> Image.Image:
    _validate_grid(grid)
    arr = np.array(grid, dtype=np.int16)          # HxW

    # Convert to 0-based palette indices based on INDEX_BASE.
    if INDEX_BASE == 0:
        idx = arr
    else:
        idx = arr - 1

    rgb = PALETTE_RGB[idx]                        # HxWx3
    img = np.kron(rgb, np.ones((cell,cell,1), dtype=np.uint8))
    if draw_grid and cell >= 8:
        img[::cell,:,:] = GRID_COLOR; img[-1,:,:] = GRID_COLOR
        img[:,::cell,:] = GRID_COLOR; img[:,-1,:] = GRID_COLOR
    return Image.fromarray(img, "RGB")

def _label_above(img: Image.Image, text: str) -> Image.Image:
    strip = Image.new("RGB", (img.width, LABEL_H), BG_COLOR)
    draw  = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=LABEL_H-10)
    except Exception:
        font = ImageFont.load_default()
    tx = (img.width - draw.textlength(text, font=font)) // 2
    ty = max(2, (LABEL_H - getattr(font, "size", 12)) // 2)
    draw.text((tx, ty), text, fill=(0,0,0), font=font)

    out = Image.new("RGB", (img.width, LABEL_H + img.height), BG_COLOR)
    out.paste(strip, (0,0))
    out.paste(img, (0, LABEL_H))
    return out

def _placeholder_box(base: Image.Image, text: str) -> Image.Image:
    w, h = base.size
    img = Image.new("RGB", (w, h), BG_COLOR)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=min(h - 4, LABEL_H - 4))
    except Exception:
        font = ImageFont.load_default()
    tw = draw.textlength(text, font=font)
    th = getattr(font, "size", 12)
    x = max(2, (w - tw) // 2)
    y = max(2, (h - th) // 2)
    draw.rectangle((1, 1, w - 2, h - 2), outline=GRID_COLOR, width=2)
    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    return img

def _io_pred_panel(
    input_grid: List[List[int]],
    output_grid: List[List[int]] | None,
    pred_grid: List[List[int]] | None,
    label_input: str,
    label_output: str | None,
    label_pred: str,
) -> Image.Image:
    in_img = _grid_to_image(_normalize_grid(input_grid))
    in_col = _label_above(in_img, label_input)

    if output_grid is not None:
        out_img = _grid_to_image(_normalize_grid(output_grid))
    else:
        out_img = _placeholder_box(in_img, "no ground truth")
    out_col = _label_above(out_img, label_output or "Output")

    if pred_grid is not None:
        pred_img = _grid_to_image(_normalize_grid(pred_grid))
    else:
        pred_img = _placeholder_box(out_img, "prediction not included")
    pred_col = _label_above(pred_img, label_pred)

    h = max(in_col.height, out_col.height, pred_col.height)
    w = in_col.width + COL_GAP + out_col.width + COL_GAP + pred_col.width
    canvas = Image.new("RGB", (w, h), BG_COLOR)

    y_in = (h - in_col.height) // 2
    y_out = (h - out_col.height) // 2
    y_pred = (h - pred_col.height) // 2

    x0 = 0
    canvas.paste(in_col, (x0, y_in))
    x1 = x0 + in_col.width + COL_GAP
    canvas.paste(out_col, (x1, y_out))
    x2 = x1 + out_col.width + COL_GAP
    canvas.paste(pred_col, (x2, y_pred))
    return canvas

def _two_panel(input_grid: List[List[int]], output_grid: List[List[int]] | None, left_label: str, right_label: str | None = None) -> Image.Image:
    """Build a side-by-side panel for a single example.

    If output_grid is None, only the left (input) column is shown.
    """
    in_img = _grid_to_image(_normalize_grid(input_grid))
    in_col = _label_above(in_img, left_label)

    if output_grid is None:
        return in_col

    out_img = _grid_to_image(_normalize_grid(output_grid))
    out_col = _label_above(out_img, right_label or "Output")

    h = max(in_col.height, out_col.height)
    w = in_col.width + COL_GAP + out_col.width
    canvas = Image.new("RGB", (w, h), BG_COLOR)

    y_in = (h - in_col.height) // 2
    y_out = (h - out_col.height) // 2

    canvas.paste(in_col, (0, y_in))
    canvas.paste(out_col, (in_col.width + COL_GAP, y_out))
    return canvas

def _triple_panel(guess: List[List[int]], real: List[List[int]]) -> Image.Image:
    g_img = _grid_to_image(_normalize_grid(guess))
    r_img = _grid_to_image(_normalize_grid(real))

    g_arr = np.array(guess, dtype=np.int16)
    r_arr = np.array(real,  dtype=np.int16)
    H, W = max(g_arr.shape[0], r_arr.shape[0]), max(g_arr.shape[1], r_arr.shape[1])
    diff = (_pad_to(g_arr, H, W) != _pad_to(r_arr, H, W)).astype(np.int16)

    d_vis = [[2 if v==1 else 0 for v in row] for row in diff]  # 2=red, 0=black
    d_img = _grid_to_image(d_vis)

    g_col = _label_above(g_img, "Guess")
    r_col = _label_above(r_img, "Real")
    d_col = _label_above(d_img, "Diff")

    # Compose side-by-side
    h = max(g_col.height, r_col.height, d_col.height)
    w = g_col.width + COL_GAP + r_col.width + COL_GAP + d_col.width
    canvas = Image.new("RGB", (w, h), BG_COLOR)
    def paste_center(x, col):
        y = (h - col.height)//2
        canvas.paste(col, (x, y))
        return x + col.width
    x = 0
    x = paste_center(x, g_col); x += COL_GAP
    x = paste_center(x, r_col); x += COL_GAP
    paste_center(x, d_col)
    return canvas

def _stack_rows(rows: List[Image.Image]) -> Image.Image:
    if not rows:
        return Image.new("RGB", (1,1), BG_COLOR)
    W = max(im.width for im in rows)
    H = sum(im.height for im in rows) + ROW_GAP * (len(rows)-1)
    out = Image.new("RGB", (W, H), BG_COLOR)
    y = 0
    for idx, im in enumerate(rows):
        x = (W - im.width)//2
        out.paste(im, (x, y))
        y += im.height
        if idx < len(rows)-1:
            y += ROW_GAP
    return out

# -------- main pipeline --------
def _write_png(img: Image.Image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path.as_posix())
    return path.as_posix()


def _label_with_filename(img: Image.Image, filename: str) -> Image.Image:
    """Add a filename strip at the top-left of the full image."""
    strip = Image.new("RGB", (img.width, LABEL_H), BG_COLOR)
    draw  = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=LABEL_H-10)
    except Exception:
        font = ImageFont.load_default()

    # Left-aligned filename text
    tx = 4
    ty = max(2, (LABEL_H - getattr(font, "size", 12)) // 2)
    draw.text((tx, ty), filename, fill=(0, 0, 0), font=font)

    out = Image.new("RGB", (img.width, LABEL_H + img.height), BG_COLOR)
    out.paste(strip, (0, 0))
    out.paste(img, (0, LABEL_H))
    return out


def _get_pred_grid(pred_obj: Any | None, split: str, index: int) -> List[List[int]] | None:
    if not isinstance(pred_obj, dict):
        return None
    seq = pred_obj.get(split)
    if not isinstance(seq, list) or index >= len(seq):
        return None
    item = seq[index]
    if isinstance(item, dict):
        grid = item.get("output")
    else:
        grid = item
    if _is_grid(grid):
        return grid
    return None


def _visualize_arc_task(task_path: Path, out_root: Path, pred_obj: Any | None = None) -> str:
    """Render a single ARC-style task JSON (with 'train' and 'test') to a PNG."""
    obj = json.loads(task_path.read_text())
    if not isinstance(obj, dict) or "train" not in obj or "test" not in obj:
        raise SystemExit(f"{task_path} must be an ARC-style JSON with 'train' and 'test'.")

    rows: List[Image.Image] = []

    # Visualize train pairs: input -> output (+ optional prediction)
    for i, pair in enumerate(obj.get("train", [])):
        if not isinstance(pair, dict) or "input" not in pair:
            continue
        inp = pair["input"]
        out = pair.get("output") if _is_grid(pair.get("output", [])) else None
        label_left = f"train_{i}_input"
        label_right = f"train_{i}_output" if out is not None else None
        pred = _get_pred_grid(pred_obj, "train", i)
        rows.append(
            _io_pred_panel(
                inp,
                out,
                pred,
                label_left,
                label_right,
                f"train_{i}_pred",
            )
        )

    # Visualize test pairs: input (and output if present) (+ optional prediction)
    for j, pair in enumerate(obj.get("test", [])):
        if not isinstance(pair, dict) or "input" not in pair:
            continue
        inp = pair["input"]
        out = pair.get("output") if _is_grid(pair.get("output", [])) else None
        label_left = f"test_{j}_input"
        label_right = f"test_{j}_output" if out is not None else None
        pred = _get_pred_grid(pred_obj, "test", j)
        rows.append(
            _io_pred_panel(
                inp,
                out,
                pred,
                label_left,
                label_right,
                f"test_{j}_pred",
            )
        )

    if not rows:
        raise SystemExit("No train/test pairs with valid grids found to render.")

    out_path = out_root / f"{task_path.stem}.png"
    stacked = _stack_rows(rows)
    labeled = _label_with_filename(stacked, task_path.name)
    return _write_png(labeled, out_path)

def main():
    here = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Visualize ARC-style tasks (train/test grids) as PNGs.")
    parser.add_argument(
        "--task_file",
        type=str,
        default=TASK_FILE,
        help="Path to a single ARC JSON task file (with 'train' and 'test'). Ignored if --task_dir is set.",
    )
    parser.add_argument(
        "--task_dir",
        type=str,
        default=None,
        help="Optional directory containing ARC JSON task files; visualize each *.json file.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=OUT_DIR,
        help="Output directory for generated PNGs.",
    )
    parser.add_argument(
        "--index_base",
        type=int,
        choices=[0, 1],
        default=1,
        help="Color index base: 0 for [0..9], 1 for [1..10]. Default: 1.",
    )
    parser.add_argument(
        "--pred_file",
        type=str,
        default=None,
        help="Optional JSON file with model predictions for a single task.",
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        default=None,
        help="Optional directory containing prediction JSON files matching task filenames.",
    )

    args = parser.parse_args()

    # Configure global index base for grid values.
    global INDEX_BASE
    INDEX_BASE = args.index_base

    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = here / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    pred_root = None
    if args.pred_dir is not None:
        pred_root = Path(args.pred_dir)
        if not pred_root.is_absolute():
            pred_root = here / pred_root

    if args.task_dir is not None:
        dir_path = Path(args.task_dir)
        if not dir_path.is_absolute():
            dir_path = here / dir_path

        json_files = sorted(p for p in dir_path.glob("*.json") if p.is_file())
        if not json_files:
            raise SystemExit(f"No .json files found in directory: {dir_path}")

        for task_path in json_files:
            pred_obj = None
            if pred_root is not None:
                pred_path = pred_root / task_path.name
                if pred_path.is_file():
                    try:
                        pred_obj = json.loads(pred_path.read_text())
                    except Exception:
                        pred_obj = None
            try:
                out_path = _visualize_arc_task(task_path, out_root, pred_obj)
                print("[ok]", out_path)
            except SystemExit as e:
                print(f"[skip] {task_path}: {e}")
    else:
        task_path = Path(args.task_file)
        if not task_path.is_absolute():
            task_path = here / task_path
        pred_obj = None
        if args.pred_file is not None:
            pred_path = Path(args.pred_file)
            if not pred_path.is_absolute():
                pred_path = here / pred_path
            if pred_path.is_file():
                try:
                    pred_obj = json.loads(pred_path.read_text())
                except Exception:
                    pred_obj = None
        elif pred_root is not None:
            pred_path = pred_root / task_path.name
            if pred_path.is_file():
                try:
                    pred_obj = json.loads(pred_path.read_text())
                except Exception:
                    pred_obj = None
        out_path = _visualize_arc_task(task_path, out_root, pred_obj)
        print("[ok]", out_path)

if __name__ == "__main__":
    main()


