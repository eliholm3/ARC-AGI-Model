import json
from pathlib import Path

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
def _add_one_to_all_values_in_place(data):
    """
    Adds +1 to every scalar value in each input/output grid across all samples.
    Done BEFORE padding so pad_value=0 remains 0.
    """
    for sample in data.values():
        for split in ["train", "test"]:
            for pairs in sample.get(split, []):
                # input grid
                r = 0
                while r < len(pairs["input"]):
                    c = 0
                    row = pairs["input"][r]
                    while c < len(row):
                        row[c] = row[c] + 1
                        c += 1
                    r += 1
                # output grid
                r = 0
                while r < len(pairs["output"]):
                    c = 0
                    row = pairs["output"][r]
                    while c < len(row):
                        row[c] = row[c] + 1
                        c += 1
                    r += 1


def get_metrics(data):
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
        if (len(sample['train']) > metric_dict['max_train_len']):
            metric_dict['max_train_len'] = len(sample['train'])
        if (len(sample['test']) > metric_dict['max_test_len']):
            metric_dict['max_test_len'] = len(sample['test'])
        for pairs in sample['train']:
            if (len(pairs['input']) > metric_dict['max_train_input_height']):
                metric_dict['max_train_input_height'] = len(pairs['input'])
            if (len(pairs['output']) > metric_dict['max_train_output_height']):
                metric_dict['max_train_output_height'] = len(pairs['output'])
            for inp in pairs['input']:
                if (len(inp) > metric_dict['max_train_input_width']):
                    metric_dict['max_train_input_width'] = len(inp)
            for output in pairs['output']:
                if (len(output) > metric_dict['max_train_output_width']):
                    metric_dict['max_train_output_width'] = len(output)
        for pairs in sample['test']:
            if (len(pairs['input']) > metric_dict['max_test_input_height']):
                metric_dict['max_test_input_height'] = len(pairs['input'])
            if (len(pairs['output']) > metric_dict['max_test_output_height']):
                metric_dict['max_test_output_height'] = len(pairs['output'])
            for inp in pairs['input']:
                if (len(inp) > metric_dict['max_test_input_width']):
                    metric_dict['max_test_input_width'] = len(inp)
            for output in pairs['output']:
                if (len(output) > metric_dict['max_test_output_width']):
                    metric_dict['max_test_output_width'] = len(output)
    return metric_dict


def pad_data(data, metric_dict=None, pad_value=0):
    """
    Pads each sample so that *both* train and test grids
    share the same square size per sample.

    metric_dict is ignored (kept for backward compatibility).
    """
    for sample in data.values():
        # -----------------------------------
        # 1. Compute per-sample global maxima
        #    across BOTH train and test
        # -----------------------------------
        max_input_height = 0
        max_input_width  = 0
        max_output_height = 0
        max_output_width  = 0

        # Look at TRAIN split
        for pairs in sample.get('train', []):
            # input
            if len(pairs['input']) > max_input_height:
                max_input_height = len(pairs['input'])
            for row in pairs['input']:
                if len(row) > max_input_width:
                    max_input_width = len(row)

            # output
            if len(pairs['output']) > max_output_height:
                max_output_height = len(pairs['output'])
            for row in pairs['output']:
                if len(row) > max_output_width:
                    max_output_width = len(row)

        # Look at TEST split
        for pairs in sample.get('test', []):
            # input
            if len(pairs['input']) > max_input_height:
                max_input_height = len(pairs['input'])
            for row in pairs['input']:
                if len(row) > max_input_width:
                    max_input_width = len(row)

            # output
            if len(pairs['output']) > max_output_height:
                max_output_height = len(pairs['output'])
            for row in pairs['output']:
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
            while len(pairs['input']) < max_size:
                pairs['input'].append([pad_value] * max_size)
            for row in pairs['input']:
                while len(row) < max_size:
                    row.append(pad_value)

            # output
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
            while len(pairs['input']) < max_size:
                pairs['input'].append([pad_value] * max_size)
            for row in pairs['input']:
                while len(row) < max_size:
                    row.append(pad_value)

            # output
            while len(pairs['output']) < max_size:
                pairs['output'].append([pad_value] * max_size)
            for row in pairs['output']:
                while len(row) < max_size:
                    row.append(pad_value)

    return data



# def pad_data(data, metric_dict=None, pad_value=0):
    # """
    # Pads each sample independently to its own max square size.
    # metric_dict is ignored (kept for backward compatibility).
    # """
    # for sample in data.values():
    #     # ----- compute per-sample maxima for TRAIN -----
    #     max_train_input_height = 0
    #     max_train_input_width  = 0
    #     max_train_output_height = 0
    #     max_train_output_width  = 0

    #     for pairs in sample.get('train', []):
    #         if len(pairs['input'])  > max_train_input_height:  max_train_input_height  = len(pairs['input'])
    #         if len(pairs['output']) > max_train_output_height: max_train_output_height = len(pairs['output'])
    #         for inp in pairs['input']:
    #             if len(inp) > max_train_input_width:  max_train_input_width  = len(inp)
    #         for outp in pairs['output']:
    #             if len(outp) > max_train_output_width: max_train_output_width = len(outp)

    #     # ----- compute per-sample maxima for TEST -----
    #     max_test_input_height = 0
    #     max_test_input_width  = 0
    #     max_test_output_height = 0
    #     max_test_output_width  = 0

    #     for pairs in sample.get('test', []):
    #         if len(pairs['input'])  > max_test_input_height:  max_test_input_height  = len(pairs['input'])
    #         if len(pairs['output']) > max_test_output_height: max_test_output_height = len(pairs['output'])
    #         for inp in pairs['input']:
    #             if len(inp) > max_test_input_width:  max_test_input_width  = len(inp)
    #         for outp in pairs['output']:
    #             if len(outp) > max_test_output_width: max_test_output_width = len(outp)

    #     # ----- per-sample square sizes -----
    #     max_train_size = max(
    #         max_train_input_height,
    #         max_train_input_width,
    #         max_train_output_height,
    #         max_train_output_width
    #     )
    #     max_test_size = max(
    #         max_test_input_height,
    #         max_test_input_width,
    #         max_test_output_height,
    #         max_test_output_width
    #     )

    #     # ----- pad TRAIN for this sample -----
    #     for pairs in sample.get('train', []):
    #         # input
    #         while len(pairs['input']) < max_train_size:
    #             pairs['input'].append([pad_value] * max_train_size)
    #         for inp in pairs['input']:
    #             while len(inp) < max_train_size:
    #                 inp.append(pad_value)
    #         # output
    #         while len(pairs['output']) < max_train_size:
    #             pairs['output'].append([pad_value] * max_train_size)
    #         for outp in pairs['output']:
    #             while len(outp) < max_train_size:
    #                 outp.append(pad_value)

    #     # ----- pad TEST for this sample -----
    #     for pairs in sample.get('test', []):
    #         # input
    #         while len(pairs['input']) < max_test_size:
    #             pairs['input'].append([pad_value] * max_test_size)
    #         for inp in pairs['input']:
    #             while len(inp) < max_test_size:
    #                 inp.append(pad_value)
    #         # output
    #         while len(pairs['output']) < max_test_size:
    #             pairs['output'].append([pad_value] * max_test_size)
    #         for outp in pairs['output']:
    #             while len(outp) < max_test_size:
    #                 outp.append(pad_value)

    # return data


def _infer_original_size_from_padded(grid, pad_value=0):
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


def build_sample_level_dataset(data, pad_value=0):
    """
    Build a list of per-sample records.
    NEW: also stores per-pair masks: 1 where value != pad_value, else 0.
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
        idx = 0
        for pairs in sample['train']:
            inp_grid = pairs['input']
            out_grid = pairs['output']

            # original sizes (prefer stored, else infer)
            if ('orig_input_size' in pairs):
                in_h, in_w = pairs['orig_input_size']
            else:
                in_h, in_w = _infer_original_size_from_padded(inp_grid, pad_value)
            if ('orig_output_size' in pairs):
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

            # NEW: masks (1 for non-pad, 0 for pad)
            inp_mask = (inp_tensor != pad_value).long()
            out_mask = (out_tensor != pad_value).long()

            # store pair
            train_pairs.append({
                "input": inp_tensor,
                "output": out_tensor,
                "input_mask": inp_mask,
                "output_mask": out_mask
            })
            idx += 1

        # ----- TEST -----
        idx = 0
        for pairs in sample['test']:
            inp_grid = pairs['input']
            out_grid = pairs['output']

            if ('orig_input_size' in pairs):
                in_h, in_w = pairs['orig_input_size']
            else:
                in_h, in_w = _infer_original_size_from_padded(inp_grid, pad_value)
            if ('orig_output_size' in pairs):
                out_h, out_w = pairs['orig_output_size']
            else:
                out_h, out_w = _infer_original_size_from_padded(out_grid, pad_value)

            if in_h > test_max_h: test_max_h = in_h
            if out_h > test_max_h: test_max_h = out_h
            if in_w > test_max_w: test_max_w = in_w
            if out_w > test_max_w: test_max_w = out_w

            inp_tensor = torch.tensor(inp_grid).long()
            out_tensor = torch.tensor(out_grid).long()

            # NEW: masks (1 for non-pad, 0 for pad)
            inp_mask = (inp_tensor != pad_value).long()
            out_mask = (out_tensor != pad_value).long()

            test_pairs.append({
                "input": inp_tensor,
                "output": out_tensor,
                "input_mask": inp_mask,
                "output_mask": out_mask
            })
            idx += 1

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