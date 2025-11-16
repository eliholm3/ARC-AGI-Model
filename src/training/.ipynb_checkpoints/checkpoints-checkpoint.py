import os
import torch


###############################
#   Checkpoint Utilities      #
###############################

def ensure_dir(path: str):
    """
    Creates directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def save_checkpoint(
        obj,            # nn.Module or dict
        path: str
):
    """
    Saves a module or state dict to the specified path.
    """

    ensure_dir(os.path.dirname(path))
    torch.save(obj, path)
    print(f"[checkpoint] Saved: {path}")


def load_checkpoint(
        path: str,
        map_location=None,
        strict: bool = True
):
    """
    Loads a full checkpoint (.pt or .pth). If strict=False and it is a state_dict,
    caller can partially load it.
    """

    if not os.path.exists(path):
        print(f"[checkpoint] Not found: {path}")
        return None

    ckpt = torch.load(path, map_location=map_location)
    print(f"[checkpoint] Loaded: {path}")
    return ckpt
