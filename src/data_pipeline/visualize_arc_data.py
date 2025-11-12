import json
import random
import matplotlib.pyplot as plt
from pathlib import Path

def plot_grid(ax, grid, title=""):
    """Draw a colored grid for ARC input/output."""
    ax.imshow(grid, cmap="tab20", interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")

def visualize_arc_file(json_path, num_samples=3):
    """Visualize a few random tasks from one ARC JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    print(f"\nLoaded {len(data)} tasks from {json_path.name}")

    # Pick random task IDs
    task_ids = random.sample(list(data.keys()), min(num_samples, len(data)))

    for task_id in task_ids:
        task = data[task_id]
        print(f"Visualizing task: {task_id}")

        # Handle challenge-type files (train/test)
        if "train" in task:
            train_pairs = task["train"]
            test_pairs = task["test"]

            # Create a grid of subplots
            n_train = len(train_pairs)
            n_test = len(test_pairs)
            fig, axes = plt.subplots(2, max(n_train, n_test), figsize=(3 * max(n_train, n_test), 6))

            # Make sure axes is 2D array
            if n_train == 1 and n_test == 1:
                axes = axes.reshape(2, 1)
            elif n_train == 1 or n_test == 1:
                axes = axes.reshape(2, -1)

            # Plot training examples
            for i, pair in enumerate(train_pairs):
                plot_grid(axes[0, i], pair["input"], f"Train Input {i+1}")
                plot_grid(axes[1, i], pair["output"], f"Train Output {i+1}")

            # Plot test examples
            for j, pair in enumerate(test_pairs):
                if "input" in pair:
                    plot_grid(axes[0, j], pair["input"], f"Test Input {j+1}")
                if "output" in pair:
                    plot_grid(axes[1, j], pair["output"], f"Test Output {j+1}")

            fig.suptitle(f"Task ID: {task_id}", fontsize=12)
            plt.tight_layout()
            plt.show()

        # Handle solution files (lists of outputs)
        else:
            print(f"Solutions-only task: {task_id} -> contains {len(task)} entries.")
            for i, sol in enumerate(task):
                fig, ax = plt.subplots()
                plot_grid(ax, sol, f"Solution {i+1}")
                plt.show()


def main():
    data_dir = Path("arc-data-cleaned")  # or "arc-data" if you haven’t cleaned yet
    json_files = list(data_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {data_dir.resolve()}")
        return

    for json_file in json_files:
        visualize_arc_file(json_file, num_samples=2)  # show 2 random tasks per file


if __name__ == "__main__":
    main()
