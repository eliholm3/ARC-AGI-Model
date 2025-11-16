import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Optional, List


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_loss_plot(train_losses: List[float], val_losses: Optional[List[float]] = None,
                   out_path: str = "checkpoints/loss.png"):
    """
    Save train (and optional validation) loss plot to `out_path` using Agg backend.
    """
    ensure_dir(os.path.dirname(out_path) or ".")
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, label="train loss", marker="o")
    if val_losses is not None and len(val_losses) > 0:
        val_epochs = list(range(1, len(val_losses) + 1))
        plt.plot(val_epochs, val_losses, label="val loss", marker="x")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def append_metrics_csv(csv_path: str, row: Dict):
    """
    Append a dictionary row to a CSV file. Writes header if file does not exist.
    """
    ensure_dir(os.path.dirname(csv_path) or ".")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
