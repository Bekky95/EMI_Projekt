import pandas as pd
from sentence_transformers import SentenceTransformer
import ast
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from datasets import load_dataset

# Models
text_model = SentenceTransformer("all-MiniLM-L6-v2")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def image_embedding_pil(cur_image):
    inputs = clip_processor(images=cur_image, return_tensors="pt")

    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)

    if hasattr(outputs, "numpy"):
        return outputs.numpy()

    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def text_embedding(text):
    return text_model.encode(text)


def image_embedding(path):
    img = Image.open(path).convert("RGB")
    inputs = clip_processor(images=img, return_tensors="pt")

    with torch.no_grad():
        feat = clip_model.get_image_features(**inputs)

    return feat.squeeze().numpy()


if __name__ == "__main__":

    dataset = load_dataset("Leonardo6/memotion")

    records = []

    #TODO: so nicht, anders laden
    for sample in dataset["train"]:
        # TEXT
        text = sample["messages"][0]["content"]

        # IMAGE
        image = sample["images"][0]

        # LABEL (LISTE!)
        label = ast.literal_eval(sample["messages"][1]["content"])

        t = text_embedding(text)
        i = image_embedding_pil(image)

        records.append((t, i, label))

    np.savez(
        "features/memotion/memotion.npz",
        text=np.array([r[0] for r in records]),
        image=np.array([r[1] for r in records]),
        label=np.array([r[2] for r in records]),
    )
