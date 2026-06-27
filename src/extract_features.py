import os.path
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
import kagglehub
import pandas as pd
from PIL import Image

from helper.directory_functions import is_dataset_dir_existing, create_dir_name, get_root, \
    search_memotion_dataset_7k_dir, is_memotion_dataset_7k_existing


class ExtractFeaturesHuggingface:
    """
    Kapselt
    """
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

    def __init__(self, dataset_name="williamscott701/memotion-dataset-7k"):
        """
        Klasse, die das Laden und im Projektverzeichnis ablegen von
        Datensets aus Keggle kapselt

        :param dataset_name: default="williamscott701/memotion-dataset-7k"
        """
        self.dataset_name = dataset_name
        self.dataset_dir_name = create_dir_name(self.dataset_name)
        self._is_full_dataset_dir_existing = is_dataset_dir_existing(self.dataset_dir_name)
        self._is_memotion_dataset_7k_dir_existing = is_memotion_dataset_7k_existing()

    # path = kagglehub.dataset_download("williamscott701/memotion-dataset-7k")
    def load_and_save_dataset(self, dataset_name="williamscott701/memotion-dataset-7k"):
        """
        Läd das Datenset runter, legt ein Überordner an /<Username>_<Datasetname>_

        :param dataset_name: default="williamscott701/memotion-dataset-7k"
        :return: True || False je nachdem ob das Runterladen funktioniert hat
        """
        if not self._is_full_dataset_dir_existing:
            if dataset_name == self.dataset_name:
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
        """
        Läd den Datensatz aus dem Projektverzeichnis, aktuell sucht es den in load_and_save_dataset() angelegten
        Überordner oder den für das default Datenset übliche Verzeichnis memotion-dataset-7k und läd dort die
        Daten heraus

        :param dataset_name: default="williamscott701/memotion-dataset-7k"
        :return:
        """
        #TODO: der self.get_labels_csv() Aufruf ist irgendwie redundant, da
        # ja der komplette Datenpfad zu memotion_dataset_7k zurückgegeben wird und
        # da ja auch der Überordner drin ist, der in self.load_and_save_dataset
        # generiert wird
        if self._is_memotion_dataset_7k_dir_existing:

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
        """
        Code den ich aus https://www.kaggle.com/code/vishwapatel214/clip-model übernommen hab

        :param csv_data:
        :return:
        """
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
        """
        Getter für die Zustandsvariable die nachprüft, ob das in load_and_save_dataset()
        angelegte Überverzeichnis existiert

        :return:
        """
        return self._is_full_dataset_dir_existing

    def get_images_path(self) -> str:
        """
        gibt den kompletten Datenpfad zu /images zurück, beginnend bei Laufwerk C
        :return:
        """
        return os.path.join(search_memotion_dataset_7k_dir(), "images")

    def get_labels_csv_path(self) -> str:
        """
        gibt den kompletten Datenpfad zu labels.csv zurück, beginnend bei Laufwerk C
        :return:
        """
        return os.path.join(search_memotion_dataset_7k_dir(), "labels.csv")

