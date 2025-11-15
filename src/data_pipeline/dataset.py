import torch
from torch.utils.data import Dataset


class ARCSampleDataset(Dataset):
    def __init__(self, sample_list):
        self.data = sample_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # stack per-sample pairs into tensors
        train_inputs = torch.stack([p["input"] for p in sample["train_pairs"]])      # [num_train, H, W]
        train_outputs = torch.stack([p["output"] for p in sample["train_pairs"]])    # [num_train, H, W]
        test_inputs = torch.stack([p["input"] for p in sample["test_pairs"]])        # [num_test, H, W]
        test_outputs = torch.stack([p["output"] for p in sample["test_pairs"]])      # [num_test, H, W]

        # masks
        train_input_masks = torch.stack([p["input_mask"] for p in sample["train_pairs"]])
        train_output_masks = torch.stack([p["output_mask"] for p in sample["train_pairs"]])
        test_input_masks  = torch.stack([p["input_mask"] for p in sample["test_pairs"]])
        test_output_masks = torch.stack([p["output_mask"] for p in sample["test_pairs"]])

        return {
            "id": sample["id"],
            "train_inputs": train_inputs,
            "train_outputs": train_outputs,
            "test_inputs": test_inputs,
            "test_outputs": test_outputs,
            "train_input_masks": train_input_masks,
            "train_output_masks": train_output_masks,
            "test_input_masks": test_input_masks,
            "test_output_masks": test_output_masks,
            "train_original_size": torch.tensor(sample["train_original_size"], dtype=torch.long),
            "test_original_size": torch.tensor(sample["test_original_size"], dtype=torch.long)
        }