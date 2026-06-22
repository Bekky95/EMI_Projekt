import os.path

from datasets import load_dataset, load_from_disk, DatasetDict, Dataset

from helper.directory_functions import is_dataset_dir_existing, create_dir_name, get_root


class ExtractFeatures:
    DATASET_DIR = os.path.join(get_root(), "data", "dataset")
    DATASET_NAME = ""

    def __init__(self, dataset_name="Leonardo6/memotion"):
        self.dataset_name = dataset_name
        self.dataset_dir_name = create_dir_name(self.dataset_name)
        self._load_and_save_dataset()

    def _load_and_save_dataset(self) -> None:
        if not is_dataset_dir_existing(self.dataset_dir_name):
            cur_dataset = load_dataset(self.dataset_name)
            cur_dataset.save_to_disk(os.path.join(self.DATASET_DIR, self.dataset_dir_name))

    def load_dataset_from_dir(self, dataset_name="Leonardo6/memotion") -> Dataset | DatasetDict | None:
        if is_dataset_dir_existing(create_dir_name(dataset_name)):
            return load_from_disk(os.path.join(self.DATASET_DIR, self.dataset_dir_name))
        else:
            return None

