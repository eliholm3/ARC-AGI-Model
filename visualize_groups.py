
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ========= EDIT THESE IF YOU WANT DIFFERENT FILENAMES =========
REAL_FILE  = "real_groups.json"
GUESS_FILE = "guess_groups.json"
OUT_DIR    = "out"
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
            if not (0 <= int(v) <= 9):
                raise ValueError(f"{name}[{y}][{x}]={v} not in [0..9].")

def _normalize_grid(g: List[List[int]]) -> List[List[int]]:
    """Pad ragged rows (shouldn't happen if validated) and return a copy."""
    w = max(len(r) for r in g)
    return [r + [0]*(w - len(r)) for r in g]

def _pad_to(g: np.ndarray, H: int, W: int, fill: int = -1) -> np.ndarray:
    out = np.full((H, W), fill, dtype=np.int16)
    out[:g.shape[0], :g.shape[1]] = g
    return out

# -------- loading: EXACT schema you posted --------
def _load_groups(path: Path) -> Dict[str, List[List[List[int]]]]:
    """
    Load your "group_id -> [grid, grid, ...]" mapping.
    Accepts also a superset: if a value is a single grid, we wrap it in a list.
    """
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
    rgb = PALETTE_RGB[arr]                        # HxWx3
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

def main():
    here = Path(__file__).resolve().parent
    real_path  = here / REAL_FILE
    guess_path = here / GUESS_FILE

    # load both as group_id -> [grid, grid, ...]
    real_groups  = _load_groups(real_path)
    guess_groups = _load_groups(guess_path)

    # union of IDs
    all_ids = sorted(set(real_groups.keys()) | set(guess_groups.keys()))
    if not all_ids:
        raise SystemExit("No groups to render.")

    for gid in all_ids:
        r_list = real_groups.get(gid, [])
        g_list = guess_groups.get(gid, [])

        n = max(len(r_list), len(g_list))
        rows = []
        for i in range(n):
            # align by index; if one missing, copy the other's shape and fill zeros
            if i < len(r_list): r = r_list[i]
            else:
                base = g_list[min(i, len(g_list)-1)]
                r = [[0]*len(base[0]) for _ in range(len(base))]

            if i < len(g_list): g = g_list[i]
            else:
                base = r_list[min(i, len(r_list)-1)]
                g = [[0]*len(base[0]) for _ in range(len(base))]

            rows.append(_triple_panel(g, r))

        stacked = _stack_rows(rows)
        out_path = here / OUT_DIR / f"group_{gid}.png"
        print("[ok]", _write_png(stacked, out_path))

if __name__ == "__main__":
    main()
