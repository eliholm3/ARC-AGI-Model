## Use the following command to create visualization of a directory full of tasks
# python visualize_groups.py \
#  --task_dir Curve-BallDatasetTasks \
#  --out_dir Curve-BallDatasetTasks_Images

## Use the following command to create visualization of a single task 
## and update the TASK_FILE and OUT_DIR variables in the script
# python visualize_groups.py


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
            # Allow either 0-based or 1-based ARC color indices, but clamp to the
            # palette size. We detect the convention in _grid_to_image.
            if not (0 <= int(v) <= len(ARC_COLORS)):
                raise ValueError(
                    f"{name}[{y}][{x}]={v} not in [0..{len(ARC_COLORS)}]."
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
    arr = np.array(grid, dtype=np.int16)          # HxW, values in [0..len(ARC_COLORS)]

    # Auto-detect convention:
    # - If there is at least one 0, treat as 0-based (valid range 0..len-1).
    # - Otherwise, treat as 1-based (valid range 1..len), and subtract 1.
    has_zero = (arr == 0).any()
    if has_zero:
        if arr.max() > len(PALETTE_RGB) - 1:
            raise ValueError(
                f"Grid uses 0-based colors but has value > {len(PALETTE_RGB)-1}."
            )
        idx = arr
    else:
        if arr.min() < 1 or arr.max() > len(PALETTE_RGB):
            raise ValueError(
                f"Grid uses 1-based colors but has value outside [1..{len(PALETTE_RGB)}]."
            )
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


def _visualize_arc_task(task_path: Path, out_root: Path) -> str:
    """Render a single ARC-style task JSON (with 'train' and 'test') to a PNG."""
    obj = json.loads(task_path.read_text())
    if not isinstance(obj, dict) or "train" not in obj or "test" not in obj:
        raise SystemExit(f"{task_path} must be an ARC-style JSON with 'train' and 'test'.")

    rows: List[Image.Image] = []

    # Visualize train pairs: input -> output
    for i, pair in enumerate(obj.get("train", [])):
        if not isinstance(pair, dict) or "input" not in pair:
            continue
        inp = pair["input"]
        out = pair.get("output") if _is_grid(pair.get("output", [])) else None
        label_left = f"train_{i}_input"
        label_right = f"train_{i}_output" if out is not None else None
        rows.append(_two_panel(inp, out, label_left, label_right))

    # Visualize test pairs: input (and output if present)
    for j, pair in enumerate(obj.get("test", [])):
        if not isinstance(pair, dict) or "input" not in pair:
            continue
        inp = pair["input"]
        out = pair.get("output") if _is_grid(pair.get("output", [])) else None
        label_left = f"test_{j}_input"
        label_right = f"test_{j}_output" if out is not None else None
        rows.append(_two_panel(inp, out, label_left, label_right))

    if not rows:
        raise SystemExit("No train/test pairs with valid grids found to render.")

    out_path = out_root / f"{task_path.stem}.png"
    stacked = _stack_rows(rows)
    return _write_png(stacked, out_path)

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

    args = parser.parse_args()

    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = here / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    if args.task_dir is not None:
        dir_path = Path(args.task_dir)
        if not dir_path.is_absolute():
            dir_path = here / dir_path

        json_files = sorted(p for p in dir_path.glob("*.json") if p.is_file())
        if not json_files:
            raise SystemExit(f"No .json files found in directory: {dir_path}")

        for task_path in json_files:
            try:
                out_path = _visualize_arc_task(task_path, out_root)
                print("[ok]", out_path)
            except SystemExit as e:
                print(f"[skip] {task_path}: {e}")
    else:
        task_path = Path(args.task_file)
        if not task_path.is_absolute():
            task_path = here / task_path
        out_path = _visualize_arc_task(task_path, out_root)
        print("[ok]", out_path)

if __name__ == "__main__":
    main()

