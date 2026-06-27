import os
from PIL import Image, ImageFile
from PIL.JpegImagePlugin import JpegImageFile
import torch
from torch.utils.data import Dataset


class CLIPMemotionDataset(Dataset):
    """
    Each item returns:
      - pixel_values: The transformed image tensor
      - input_ids, attention_mask: tokenized text
      - label: the sentiment class
    """
    def __init__(self, dataframe, images_dir, processor, max_length=77):
        self.df = dataframe.reset_index(drop=True)
        self.images_dir = images_dir
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["image_name"])
        image = Image.open(img_path).convert("RGB")
        text = row["text_corrected"]
        label = torch.tensor(row["label"], dtype=torch.long)

        # The CLIP processor can handle both images & text in a single call,
        # but we'll call it separately for clarity. We'll do them in one go:
        encoded = self.processor(
            text=[text],
            images=[image],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # encoded is a dict with keys: pixel_values, input_ids, attention_mask
        # shape: (batch=1, channels/HW or tokens)

        # We'll squeeze out batch=1 dimension so we can return plain tensors
        pixel_values = encoded["pixel_values"].squeeze(0)       # [3, 224, 224]
        input_ids = encoded["input_ids"].squeeze(0)             # [max_length]
        attention_mask = encoded["attention_mask"].squeeze(0)   # [max_length]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label
        }


class MemotionDataset(Dataset):
    # https://arrow.apache.org/docs/python/ -> .arrow file handling
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset["train"]
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]

        # for debugging:
        # var = sample["messages"][0]
        # var2 = sample["images"]

        image: JpegImageFile = sample["images"][0]
        content: str = sample["messages"][0]["content"]
        role: str = sample["messages"][0]['role']

        if self.transform:
            image = self.transform(image)

        return image, content, role
