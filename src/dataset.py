from torch.utils.data import Dataset
from PIL import Image
import os
import torch


class MemotionDataset(Dataset):

    def __init__(
        self,
        dataframe,
        img_dir,
        tokenizer,
        transform,
        target="sentiment",
        max_length=64
    ):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

        self.label_cols = {
            "sentiment": "label_sentiment",
            "humour": "label_humour",
            "sarcasm": "label_sarcasm",
            "offensive": "label_offensive"
        }

        self.label_col = self.label_cols[target]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # Bild
        img_path = os.path.join(
            self.img_dir,
            row["image_name"]
        )

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Text
        text = row["clean_text"]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        label = int(row[self.label_col])

        return {
            "image": image,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }
