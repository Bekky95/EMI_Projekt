from datasets import load_dataset, load_from_disk, DatasetDict, Dataset

from helper.directory_functions import is_dataset_dir_existing


class ExtractFeatures:
    DATASET_DIR = "../data/dataset/"
    DATASET_NAME = ""

    def __init__(self, dataset_name="Leonardo6/memotion"):
        self.dataset_name = dataset_name
        self.dataset_dir_name = _create_dir_name(self.dataset_name)
        self._load_and_save_dataset()

    def _load_and_save_dataset(self):
        if not is_dataset_dir_existing(self.dataset_dir_name):
            cur_dataset = load_dataset(self.dataset_name)
            cur_dataset.save_to_disk(self.DATASET_DIR + self.dataset_dir_name)

    def load_dataset_from_dir(self, dataset_name="Leonardo6/memotion") -> Dataset | DatasetDict | None:
        if is_dataset_dir_existing(_create_dir_name(dataset_name)):
            return load_from_disk(self.DATASET_DIR + self.dataset_dir_name)
        else:
            return None


def _create_dir_name(dataset_name):
    name_list = dataset_name.split("/")
    dir_name = ""
    for items in name_list:
        dir_name += (items + "_")

    return dir_name


if __name__ == "__main__":

    extractor = ExtractFeatures()
    ds = extractor.load_dataset_from_dir()
    print("meme")
