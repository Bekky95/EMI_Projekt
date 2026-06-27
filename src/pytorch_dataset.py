import torch
from torch.utils.data import Dataset
import os
from PIL import Image


class MemotionDataset(Dataset):
    def __init__(self, df, img_dir, tokenizer, max_len, transform):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Image ──
        img_path = os.path.join(self.img_dir, row["image_name"])
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=0)
        img_tensor = self.transform(img)

        # ── Text ──
        enc = self.tokenizer(
            row["clean_text"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # ── Labels ──
        labels = {
            "sentiment": torch.tensor(row["label_sentiment"], dtype=torch.long),
            "humour"   : torch.tensor(row["label_humour"],    dtype=torch.long),
            "sarcasm"  : torch.tensor(row["label_sarcasm"],   dtype=torch.long),
            "offensive": torch.tensor(row["label_offensive"],  dtype=torch.long),
        }

        return img_tensor, input_ids, attention_mask, labels
