import torch
import torch.nn.functional as F

# Import your generator
from arc_generator import ARCGenerator

# Probably in the same file as ARCSampleDataset
from available_functions import arc_loader  


@torch.no_grad()
def validate_phase1(generator, val_loader, device):
    generator.eval()

    total_exact = 0
    total_tests = 0

    total_pixels = 0
    total_pixels_correct = 0

    for batch in val_loader:

        # Move tensors to device
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)

        K_test = batch["test_inputs"].shape[0]

        # Forward pass
        out = generator(
            train_inputs=batch["train_inputs"],
            train_outputs=batch["train_outputs"],
            train_input_masks=batch["train_input_masks"],
            train_output_masks=batch["train_output_masks"],
            test_inputs=batch["test_inputs"],
            test_input_masks=batch["test_input_masks"],
        )

        logits = out["logits"]  # (1, K_test, C_out, H, W)
        preds = logits.argmax(dim=2)  # (1, K_test, H, W)
        preds = preds.squeeze(0)      # (K_test, H, W)

        targets = batch["test_outputs"]  # (K_test, H, W)
        H, W = targets.shape[1], targets.shape[2]

        # For pixel accuracy
        for j in range(K_test):
            pred = preds[j]
            target = targets[j]

            # -------------------------------
            # Crop prediction back to original size
            # -------------------------------
            # Provided by your dataset
            orig_h, orig_w = batch["test_original_size"].tolist()
            pred_cropped = pred[:orig_h, :orig_w]
            target_cropped = target[:orig_h, :orig_w]

            # ---- Exact Match ----
            exact = (pred_cropped == target_cropped).all().item()
            total_exact += exact
            total_tests += 1

            # ---- Pixel Accuracy ----
            total_pixels += orig_h * orig_w
            total_pixels_correct += (pred_cropped == target_cropped).sum().item()

    exact_acc = total_exact / total_tests if total_tests > 0 else 0
    pixel_acc = total_pixels_correct / total_pixels if total_pixels > 0 else 0

    print("\n=== Validation Results ===")
    print(f"Exact Match Accuracy: {exact_acc * 100:.2f}%")
    print(f"Pixel Accuracy:       {pixel_acc * 100:.2f}%\n")

    return exact_acc, pixel_acc


# -------------------------------------------------
# Example usage (in a validation script)
# -------------------------------------------------
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your trained generator
    generator = ARCGenerator(...)  # build same as train script
    generator.load_state_dict(torch.load("phase1_generator.pt", map_location=DEVICE))
    generator.to(DEVICE)

    # Use same loader or a separate validation loader
    validate_phase1(generator, arc_loader, DEVICE)
