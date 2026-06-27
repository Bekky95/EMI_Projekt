import os.path
import re
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
import kagglehub
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

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

    def __init__(self, dataset_name="williamscott701/memotion-dataset-7k"):
        self.dataset_name = dataset_name
        self.dataset_dir_name = create_dir_name(self.dataset_name)
        self._is_full_dataset_dir_existing = is_dataset_dir_existing(self.dataset_dir_name)
        self._is_memotion_dataset_7k_dir_existing = is_memotion_dataset_7k_existing()

    # path = kagglehub.dataset_download("williamscott701/memotion-dataset-7k")
    def load_and_save_dataset(self, dataset_name="williamscott701/memotion-dataset-7k"):
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


    def label_distributions(self, df: pd.DataFrame):
        """
        plots a figure with all labels from labels.csv and their counts
        """
        tasks = ['humour', 'sarcasm', 'offensive', 'motivational', 'overall_sentiment'] # emotion-labels aus der memotion-7k- CSV-Datei
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        for ax, col in zip(axes, tasks):
            if col in df.columns:
                counts = df[col].value_counts()
                counts.plot(kind="bar", ax=ax, color="#3498db", edgecolor="white")
                ax.set_title(col, fontsize=11)
                ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=30)
                for bar, val in zip(ax.patches, counts):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                            str(val), ha="center", fontsize=8)
        plt.suptitle("Memotion 7K — Class Distributions", fontsize=14)
        plt.tight_layout()
        #plt.savefig(os.path.join(CFG["out_dir"], "eda_distributions.png"), dpi=150)
        plt.show()

    def data_cleaning_and_label_encoding(self, df: pd.DataFrame):
        # ── Fix image names ──
        df["image_name"] = df["image_name"].astype(str).str.strip()
        IMG_DIR = self.get_images_path()

        def fix_ext(name):
            if not name.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                return name + ".jpg"
            return name

        df["image_name"] = df["image_name"].apply(fix_ext)

        # ── Drop rows with missing images ──
        df["img_ok"] = df["image_name"].apply(
            lambda x: os.path.exists(os.path.join(IMG_DIR, x))
        )
        print(f"Images found: {df['img_ok'].sum()} / {len(df)}")
        df = df[df["img_ok"]].reset_index(drop=True)

        # ── Text column ──
        text_col = "text_corrected" if "text_corrected" in df.columns else "text_ocr"
        print(f"Text column: {text_col}")

        def clean_text(t):
            if pd.isna(t) or str(t).strip() in ("", "nan"):
                return "no text"
            t = str(t).lower()
            t = re.sub(r"http\S+", "", t)
            t = re.sub(r"[^a-zA-Z\s!?]", " ", t)
            t = re.sub(r"\s+", " ", t).strip()
            return t or "no text"

        df["clean_text"] = df[text_col].apply(clean_text)

        # ── Label maps ──
        SENTIMENT_MAP = {
            "very_positive": "positive", "positive": "positive",
            "neutral": "neutral",
            "negative": "negative", "very_negative": "negative"
        }
        HUMOUR_MAP = {"not_funny": 0, "funny": 1, "very_funny": 2, "hilarious": 3}
        SARCASM_MAP = {"not_sarcastic": 0, "general": 1, "twisted_meaning": 2, "very_twisted": 3}
        OFFENSIVE_MAP = {"not_offensive": 0, "slight": 1, "very_offensive": 2, "hateful_offensive": 3}
        SENTIMENT_MAP3 = {"positive": 0, "neutral": 1, "negative": 2}

        df["sentiment_3"] = df["overall_sentiment"].str.strip().str.lower().map(SENTIMENT_MAP)
        df = df.dropna(subset=["sentiment_3"]).reset_index(drop=True)
        df["label_sentiment"] = df["sentiment_3"].map(SENTIMENT_MAP3)

        def safe_map(col, mapping, default=0):
            if col not in df.columns:
                return pd.Series([default] * len(df))
            return df[col].str.strip().str.lower().map(mapping).fillna(default).astype(int)

        df["label_humour"] = safe_map("humour", HUMOUR_MAP)
        df["label_sarcasm"] = safe_map("sarcasm", SARCASM_MAP)
        df["label_offensive"] = safe_map("offensive", OFFENSIVE_MAP)

        NUM_CLASSES = {
            "sentiment": 3,
            "humour": 4,
            "sarcasm": 4,
            "offensive": 4,
        }

        print(f"Final dataset: {len(df)} rows")
        print(df["sentiment_3"].value_counts())
        return None

    def is_dataset_loaded_locally(self) -> bool:
        return self._is_full_dataset_dir_existing

    def get_images_path(self):
        return os.path.join(search_memotion_dataset_7k_dir(), "images")

    def get_labels_csv_path(self):
        return os.path.join(search_memotion_dataset_7k_dir(), "labels.csv")

