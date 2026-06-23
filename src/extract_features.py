import os.path

from datasets import load_dataset, load_from_disk, DatasetDict, Dataset

from helper.directory_functions import is_dataset_dir_existing, create_dir_name, get_root


class ExtractFeatures:
    DATASET_DIR = os.path.join(get_root(), "data", "dataset")
    DATASET_NAME = ""

    def __init__(self, dataset_name="Leonardo6/memotion"):
        self.dataset_name = dataset_name
        self.dataset_dir_name = create_dir_name(self.dataset_name)
        self._is_loaded_locally = is_dataset_dir_existing(self.dataset_dir_name)

    def load_and_save_dataset(self, dataset_name) -> bool | None:
        if not self._is_loaded_locally:
            if dataset_name == self.dataset_name:
                cur_dataset = load_dataset(self.dataset_name)
                cur_dataset.save_to_disk(os.path.join(self.DATASET_DIR, self.dataset_dir_name))
                self._is_loaded_locally = is_dataset_dir_existing(self.dataset_dir_name)
                return True
            else:
                print(f"{dataset_name} is not this dataset")
                return None
        else:
            print("Dataset is already loaded locally")
            return None

    def load_dataset_from_dir(self, dataset_name="Leonardo6/memotion") -> Dataset | DatasetDict | None:
        if not self._is_loaded_locally:
            if dataset_name == self.dataset_name:
                return load_from_disk(os.path.join(self.DATASET_DIR, self.dataset_dir_name))
            else:
                print(f"{dataset_name} is not this dataset")
                return None
        else:
            print("Dataset is already loaded locally")
            return None

    def is_dataset_loaded_locally(self) -> bool:
        return self._is_loaded_locally
