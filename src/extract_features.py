import os.path
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
import kagglehub
import pandas as pd
from PIL import Image

from helper.directory_functions import is_dataset_dir_existing, create_dir_name, get_root, \
    search_memotion_dataset_7k_dir, is_memotion_dataset_7k_existing

# TODO hier könnte man auch mit Vererbung arbeiten eine ExtractFeatures-Überklasse und Huggingface und Kaggle erben :D
#  aber das ist jetzt unnötig, will dich nur ärgern

class ExtractFeaturesHuggingface:
    DATASET_DIR = os.path.join(get_root(), "data", "dataset")
    DATASET_NAME = ""

    def __init__(self, dataset_name="Leonardo6/memotion"):
        self.dataset_name = dataset_name
        self.dataset_dir_name = create_dir_name(self.dataset_name)
        self._is_loaded_locally = is_dataset_dir_existing(self.dataset_dir_name)

    def load_and_save_dataset(self, dataset_name="Leonardo6/memotion") -> bool | None:
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
        if self._is_loaded_locally:
            if dataset_name == self.dataset_name:
                return load_from_disk(os.path.join(self.DATASET_DIR, self.dataset_dir_name))
            else:
                print(f"{dataset_name} is not this dataset")
                return None
        else:
            print("Dataset is not loaded locally")
            return None

    def is_dataset_loaded_locally(self) -> bool:
        return self._is_loaded_locally


# kagglehub.dataset_download("kerneler/starter-memotion-dataset-7k-5c8a3974-b")

class ExtractFeaturesKaggle:
    DATASET_DIR = os.path.join(get_root(), "data", "dataset")
    DATASET_NAME = ""

    def __init__(self, dataset_name="williamscott701/memotion-dataset-7k"):
        self.dataset_name = dataset_name
        self.dataset_dir_name = create_dir_name(self.dataset_name)
        self._is_full_dataset_dir_existing = is_dataset_dir_existing(self.dataset_dir_name)
        self._is_memotion_dataset_7k_dir_existing = is_memotion_dataset_7k_existing()

    # path = kagglehub.dataset_download("williamscott701/memotion-dataset-7k")
    def load_and_save_dataset(self, dataset_name="williamscott701/memotion-dataset-7k"):
        if not self._is_full_dataset_dir_existing:
            if dataset_name == self.DATASET_NAME:
                dataset_dir = os.path.join(self.DATASET_DIR, self.dataset_dir_name)
                print(f"directory to dataset: {dataset_dir}")
                kagglehub.dataset_download(dataset_name, output_dir=dataset_dir)
                self._is_full_dataset_dir_existing = is_dataset_dir_existing(self.dataset_dir_name)
                return True
            else:
                print(f"{dataset_name} is not this dataset")
                return None
        else:
            print("Dataset is already loaded locally")
            return None

    def load_dataset_from_dir(self, dataset_name="williamscott701/memotion-dataset-7k"):
        if self._is_memotion_dataset_7k_dir_existing:
            print("memotion_dataset_7k_dir_existing")

            if self._is_full_dataset_dir_existing:
                if dataset_name == self.dataset_name:
                    csv_data = pd.read_csv(self.get_labels_csv_path())
                    return csv_data
                else:
                    print(f"{dataset_name} is not this dataset")
                    return None
            else:
                csv_data = pd.read_csv(self.get_labels_csv_path())
                return csv_data
        else:
            print("Dataset is not loaded locally")
            return None

    def check_for_bad_images(self, csv_data):
        df = csv_data[["image_name", "text_corrected", "overall_sentiment"]].dropna(
            subset=["text_corrected", "overall_sentiment"])
        df["text_corrected"] = df["text_corrected"].astype(str)
        df = df[df["text_corrected"].str.strip() != ""]

        # Verify images, skip any fully corrupt
        bad_images = 0
        valid_indices = []
        for i, row in df.iterrows():
            img_path = os.path.join(self.get_images_path(), row["image_name"])
            try:
                with Image.open(img_path) as im:
                    im.verify()
                valid_indices.append(i)
            except:
                bad_images += 1
        df = df.loc[valid_indices].reset_index(drop=True)
        print(f"Skipped {bad_images} corrupt images. Valid images: {len(df)}")

    def is_dataset_loaded_locally(self) -> bool:
        return self._is_full_dataset_dir_existing

    def get_images_path(self):
        return os.path.join(search_memotion_dataset_7k_dir(), "images")

    def get_labels_csv_path(self):
        return os.path.join(search_memotion_dataset_7k_dir(), "labels.csv")

