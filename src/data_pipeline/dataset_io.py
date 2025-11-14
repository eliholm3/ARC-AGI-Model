import os
import torch

def save_sample_level(sample_list, file_path):
    file_path = os.path.expanduser(file_path)
    folder = os.path.dirname(file_path)
    if folder and (not os.path.isdir(folder)):
        os.makedirs(folder, exist_ok=True)
    torch.save(sample_list, file_path)

def load_sample_level(file_path, map_location="cpu"):
    file_path = os.path.expanduser(file_path)
    obj = torch.load(file_path, map_location=map_location)
    return obj