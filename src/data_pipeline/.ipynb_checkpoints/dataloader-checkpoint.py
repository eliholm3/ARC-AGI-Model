from pathlib import Path
from src.data_pipeline.utils import load_jsons_from_folder, _add_one_to_all_values_in_place, pad_data, build_sample_level_dataset, arc_collate_fn_bs1
from src.data_pipeline.dataset import ARCSampleDataset
from torch.utils.data import DataLoader


class ARCDataModule:
    """
    Simple wrapper to produce a DataLoader from your folder.
    Usage:
        dm = ARCDataModule("~/path/to/training").prepare()
        loader = dm.get_loader()
        for batch in loader: ...
    """
    def __init__(
        self,
        dir_path,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        pad_value=0,
    ):
        self.dir_path = Path(dir_path).expanduser().resolve()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.pad_value = pad_value

        self.dataset = None
        self._loader = None

    def prepare(self):
        # load + preprocess
        data = load_jsons_from_folder(self.dir_path)
        _add_one_to_all_values_in_place(data)

        # pad each sample independently (metric_dict unused)
        padded = pad_data(data, metric_dict=None, pad_value=self.pad_value)
        sample_list = build_sample_level_dataset(padded, pad_value=self.pad_value)

        # build dataset + loader
        self.dataset = ARCSampleDataset(sample_list=sample_list)
        self._loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=arc_collate_fn_bs1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return self  # allow chaining

    def get_loader(self):
        if self._loader is None:
            self.prepare()
        return self._loader

    # convenience so the module itself is iterable
    def __iter__(self):
        return iter(self.get_loader())

    def __len__(self):
        return len(self.dataset) if self.dataset is not None else 0