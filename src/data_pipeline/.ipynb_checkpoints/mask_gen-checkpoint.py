import torch


def generate_valid_mask(
        H,  # data height
        W,  # data width
        original_size
):
    h_0, w_0 = original_size

    # Row valid mask: True for rows < h0
    row_mask = torch.arange(H).unsqueeze(1) < h_0   # (H, 1)

    # Col valid mask: True for cols < w0
    col_mask = torch.arange(W).unsqueeze(0) < w_0   # (1, W)

    # Broadcasting produces (H, W)
    mask = row_mask & col_mask

    return mask