# src/data_pipeline/utils.py
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


# ----------------------------
# Load all JSONs (recursive)
# ----------------------------
def load_jsons_from_folder(dir_path):
    """
    Read every .json file under dir_path (recursively) and return a dict
    keyed by the file's relative path (without the .json extension).
    """
    root = Path(dir_path).expanduser().resolve()
    files = sorted(p for p in root.rglob("*.json") if p.is_file())

    if not files:
        raise FileNotFoundError(f"No .json files found under: {root}")

    data = {}
    for p in files:
        key = str(p.relative_to(root).with_suffix(""))  # e.g. "subdir/file"
        try:
            with p.open("r", encoding="utf-8") as fh:
                data[key] = json.load(fh)
        except Exception as e:
            print(f"Failed to read {p}: {e}")

    if not data:
        raise FileNotFoundError(f"Unable to load any .json files under: {root}")

    return data


# ----------------------------
# Preprocess helpers
# ----------------------------
def _add_one_to_all_values_in_place(data: Dict[str, Any]):
    """
    Adds +1 to every scalar value in each input/output grid across all samples.
    Done BEFORE padding so pad_value=0 remains 0.

    Robust to test examples that do not include "output".
    """
    for sample in data.values():
        for split in ["train", "test"]:
            for pairs in sample.get(split, []):
                # input grid (+1)
                if "input" in pairs and isinstance(pairs["input"], list):
                    r = 0
                    while r < len(pairs["input"]):
                        row = pairs["input"][r]
                        if isinstance(row, list):
                            c = 0
                            while c < len(row):
                                if isinstance(row[c], int):
                                    row[c] = row[c] + 1
                                c += 1
                        r += 1

                # output grid (+1) — only if present
                if "output" in pairs and isinstance(pairs["output"], list):
                    r = 0
                    while r < len(pairs["output"]):
                        row = pairs["output"][r]
                        if isinstance(row, list):
                            c = 0
                            while c < len(row):
                                if isinstance(row[c], int):
                                    row[c] = row[c] + 1
                                c += 1
                        r += 1


def get_metrics(data: Dict[str, Any]):
    metric_dict = {
        "max_train_len": 0,
        "max_test_len": 0,
        "max_train_input_height": 0,
        "max_test_input_height": 0,
        "max_train_output_height": 0,
        "max_test_output_height": 0,
        "max_train_input_width": 0,
        "max_test_input_width": 0,
        "max_train_output_width": 0,
        "max_test_output_width": 0
    }

    for sample in data.values():
        if len(sample.get('train', [])) > metric_dict['max_train_len']:
            metric_dict['max_train_len'] = len(sample['train'])
        if len(sample.get('test', [])) > metric_dict['max_test_len']:
            metric_dict['max_test_len'] = len(sample['test'])

        for pairs in sample.get('train', []):
            # input dims
            if len(pairs.get('input', [])) > metric_dict['max_train_input_height']:
                metric_dict['max_train_input_height'] = len(pairs['input'])
            for inp in pairs.get('input', []):
                if len(inp) > metric_dict['max_train_input_width']:
                    metric_dict['max_train_input_width'] = len(inp)

            # output dims (train always expected to exist, but guard anyway)
            if len(pairs.get('output', [])) > metric_dict['max_train_output_height']:
                metric_dict['max_train_output_height'] = len(pairs.get('output', []))
            for output in pairs.get('output', []):
                if len(output) > metric_dict['max_train_output_width']:
                    metric_dict['max_train_output_width'] = len(output)

        for pairs in sample.get('test', []):
            # input dims
            if len(pairs.get('input', [])) > metric_dict['max_test_input_height']:
                metric_dict['max_test_input_height'] = len(pairs['input'])
            for inp in pairs.get('input', []):
                if len(inp) > metric_dict['max_test_input_width']:
                    metric_dict['max_test_input_width'] = len(inp)

            # output dims (may be missing)
            if len(pairs.get('output', [])) > metric_dict['max_test_output_height']:
                metric_dict['max_test_output_height'] = len(pairs.get('output', []))
            for output in pairs.get('output', []):
                if len(output) > metric_dict['max_test_output_width']:
                    metric_dict['max_test_output_width'] = len(output)

    return metric_dict


def _make_zero_grid_like(grid: List[List[int]], pad_value: int = 0) -> List[List[int]]:
    """Create a new grid the same shape as `grid`, filled with pad_value."""
    out: List[List[int]] = []
    r = 0
    while r < len(grid):
        row = grid[r]
        width = len(row) if isinstance(row, list) else 0
        out.append([pad_value] * width)
        r += 1
    return out


def pad_data(data: Dict[str, Any], metric_dict=None, pad_value: int = 0):
    """
    Pads each sample so that *both* train and test grids share the same square size per sample.

    Also ensures every test pair has an "output":
    - If missing, we GUESS the output size = input size and create a placeholder
      filled with pad_value, marking pairs["_guessed_output"] = True.
    """
    for sample in data.values():
        # Ensure every test pair has "output" (create placeholder if absent)
        for pairs in sample.get('test', []):
            if "output" not in pairs or not isinstance(pairs["output"], list):
                # Guess size from input grid
                guessed = _make_zero_grid_like(pairs.get("input", []), pad_value=pad_value)
                pairs["output"] = guessed
                pairs["_guessed_output"] = True  # flag for downstream logic

        # -----------------------------------
        # 1. Compute per-sample global maxima
        #    across BOTH train and test
        # -----------------------------------
        max_input_height = 0
        max_input_width  = 0
        max_output_height = 0
        max_output_width  = 0

        # TRAIN
        for pairs in sample.get('train', []):
            # input
            if len(pairs.get('input', [])) > max_input_height:
                max_input_height = len(pairs['input'])
            for row in pairs.get('input', []):
                if len(row) > max_input_width:
                    max_input_width = len(row)

            # output
            if len(pairs.get('output', [])) > max_output_height:
                max_output_height = len(pairs['output'])
            for row in pairs.get('output', []):
                if len(row) > max_output_width:
                    max_output_width = len(row)

        # TEST
        for pairs in sample.get('test', []):
            # input
            if len(pairs.get('input', [])) > max_input_height:
                max_input_height = len(pairs['input'])
            for row in pairs.get('input', []):
                if len(row) > max_input_width:
                    max_input_width = len(row)

            # output (guessed outputs already ensured)
            if len(pairs.get('output', [])) > max_output_height:
                max_output_height = len(pairs['output'])
            for row in pairs.get('output', []):
                if len(row) > max_output_width:
                    max_output_width = len(row)

        # Single square size for this sample
        max_size = max(
            max_input_height,
            max_input_width,
            max_output_height,
            max_output_width,
        )

        # -----------------------------------
        # 2. Pad TRAIN grids to max_size
        # -----------------------------------
        for pairs in sample.get('train', []):
            # input
            while len(pairs.get('input', [])) < max_size:
                pairs['input'].append([pad_value] * max_size)
            for row in pairs.get('input', []):
                while len(row) < max_size:
                    row.append(pad_value)

            # output
            if "output" not in pairs or not isinstance(pairs["output"], list):
                pairs["output"] = _make_zero_grid_like(pairs.get("input", []), pad_value)
            while len(pairs['output']) < max_size:
                pairs['output'].append([pad_value] * max_size)
            for row in pairs['output']:
                while len(row) < max_size:
                    row.append(pad_value)

        # -----------------------------------
        # 3. Pad TEST grids to max_size
        # -----------------------------------
        for pairs in sample.get('test', []):
            # input
            while len(pairs.get('input', [])) < max_size:
                pairs['input'].append([pad_value] * max_size)
            for row in pairs.get('input', []):
                while len(row) < max_size:
                    row.append(pad_value)

            # output (present by construction above)
            while len(pairs['output']) < max_size:
                pairs['output'].append([pad_value] * max_size)
            for row in pairs['output']:
                while len(row) < max_size:
                    row.append(pad_value)

    return data


def _infer_original_size_from_padded(grid: List[List[int]], pad_value=0) -> Tuple[int, int]:
    h = 0
    w = 0
    r = 0
    while r < len(grid):
        row = grid[r]
        any_nonpad = False
        last_nonpad = -1
        c = 0
        while c < len(row):
            if row[c] != pad_value:
                any_nonpad = True
                last_nonpad = c
            c += 1
        if any_nonpad:
            if (r + 1) > h:
                h = r + 1
            if (last_nonpad + 1) > w:
                w = last_nonpad + 1
        r += 1
    return (h, w)


def build_sample_level_dataset(data: Dict[str, Any], pad_value: int = 0):
    """
    Build a list of per-sample records.
    NEW: also stores per-pair masks: 1 where value != pad_value, else 0.
    If a test pair had no output originally, we:
      - used a guessed output grid sized like the input during pad_data
      - set output_mask to ALL ZEROS here (so downstream code can ignore scoring if desired)
    """
    dataset = []
    for sample_name, sample in data.items():
        # containers
        train_pairs = []
        test_pairs = []

        # track original (unpadded) sizes per split
        train_max_h = 0
        train_max_w = 0
        test_max_h = 0
        test_max_w = 0

        # ----- TRAIN -----
        for pairs in sample.get('train', []):
            inp_grid = pairs.get('input', [])
            out_grid = pairs.get('output', [])

            # original sizes (prefer stored, else infer)
            if 'orig_input_size' in pairs:
                in_h, in_w = pairs['orig_input_size']
            else:
                in_h, in_w = _infer_original_size_from_padded(inp_grid, pad_value)
            if 'orig_output_size' in pairs:
                out_h, out_w = pairs['orig_output_size']
            else:
                out_h, out_w = _infer_original_size_from_padded(out_grid, pad_value)

            # update split-wide original size (max over inputs/outputs)
            if in_h > train_max_h: train_max_h = in_h
            if out_h > train_max_h: train_max_h = out_h
            if in_w > train_max_w: train_max_w = in_w
            if out_w > train_max_w: train_max_w = out_w

            # tensors
            inp_tensor = torch.tensor(inp_grid).long()
            out_tensor = torch.tensor(out_grid).long()

            # masks (1 for non-pad, 0 for pad)
            inp_mask = (inp_tensor != pad_value).long()
            out_mask = (out_tensor != pad_value).long()

            train_pairs.append({
                "input": inp_tensor,
                "output": out_tensor,
                "input_mask": inp_mask,
                "output_mask": out_mask
            })

        # ----- TEST -----
        for pairs in sample.get('test', []):
            inp_grid = pairs.get('input', [])
            out_grid = pairs.get('output', [])

            # original sizes
            if 'orig_input_size' in pairs:
                in_h, in_w = pairs['orig_input_size']
            else:
                in_h, in_w = _infer_original_size_from_padded(inp_grid, pad_value)

            if 'orig_output_size' in pairs:
                out_h, out_w = pairs['orig_output_size']
            else:
                out_h, out_w = _infer_original_size_from_padded(out_grid, pad_value)

            if in_h > test_max_h: test_max_h = in_h
            if out_h > test_max_h: test_max_h = out_h
            if in_w > test_max_w: test_max_w = in_w
            if out_w > test_max_w: test_max_w = out_w

            inp_tensor = torch.tensor(inp_grid).long()
            out_tensor = torch.tensor(out_grid).long()

            # masks
            inp_mask = (inp_tensor != pad_value).long()

            # If output was guessed, set output_mask to all zeros (so it won't be scored).
            if pairs.get("_guessed_output", False):
                out_mask = torch.zeros_like(out_tensor).long()
            else:
                out_mask = (out_tensor != pad_value).long()

            test_pairs.append({
                "input": inp_tensor,
                "output": out_tensor,
                "input_mask": inp_mask,
                "output_mask": out_mask,
                "guessed_output": 1 if pairs.get("_guessed_output", False) else 0,
            })

        # assemble sample-level record
        item = {
            "id": str(sample_name),
            "train_pairs": train_pairs,
            "test_pairs": test_pairs,
            "train_original_size": (train_max_h, train_max_w),
            "test_original_size": (test_max_h, test_max_w)
        }
        dataset.append(item)

    return dataset


def arc_collate_fn_bs1(batch):
    # batch size is guaranteed to be 1; return the single dict unchanged
    return batch[0]
