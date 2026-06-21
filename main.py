import pandas as pd
import torch
import torchaudio
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoModelForZeroShotImageClassification, BertTokenizer

from src.dataset import MemotionDataset
from src.early_fusion import EarlyFusionModel


#TODO: pretrained Models
# Text sentence-transformers/all-MiniLM-L6-v2 384
# Bild openai/clip-vit-base-patch32 512

#TODO: Datensatz:
# Memotion (Bild + Text, Sentiment auf Memes, klein)
# https://www.kaggle.com/datasets/williamscott701/memotion-dataset-7k
# -> download als zip -> in ./data/dataset/ entpacken

#TODO: Methoden:
# Cross-Attention (uni- oder bidirektional)
# Early Fusion (Konkatenation, optional mit Projektion auf gemeinsame Dimension)

IMG_DIR = "data/dataset/memotion_dataset_7k/images"
TASK = "sentiment"
# TASK = "humour"
# TASK = "sarcasm"
# TASK = "offensive"

def main():
    # =========================
    # 1. DataFrame laden
    # =========================
    df = pd.read_csv("data/dataset/memotion_dataset_7k/labels.csv")
    # Spalten erwartet: image_path, text, label

    # =========================
    # 2. Tokenizer + Image Transform
    # =========================
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # =========================
    # 3. Dataset + DataLoader
    # =========================
    dataset = MemotionDataset(
        dataframe=df,
        img_dir=IMG_DIR,
        tokenizer=tokenizer,
        transform=image_transform,
        target=TASK,
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2
    )

    # =========================
    # 4. Modell wählen
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EarlyFusionModel(num_classes=3).to(device)
    # oder:
    # model = CrossAttentionFusion(num_classes=3).to(device)

    # =========================
    # 5. Optimizer + Loss
    # =========================
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # =========================
    # 6. Training Loop
    # =========================
    for epoch in range(5):
        print(f"Epoch {epoch + 1}")

        model.train()
        total_loss = 0

        for batch in loader:
            optimizer.zero_grad()

            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(images, input_ids, attention_mask)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("Loss:", total_loss / len(loader))


if __name__ == "__main__":
    main()
    # # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    # text_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    # # https://huggingface.co/openai/clip-vit-base-patch32?library=transformers
    # image_model_name = "openai/clip-vit-base-patch32"
    #
    # #text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    # text_model = AutoModel.from_pretrained(text_model_name)
    #
    # image_processor = AutoProcessor.from_pretrained(image_model_name)
    # image_model = AutoModelForZeroShotImageClassification.from_pretrained(image_model_name)
    #
    # # long_text = "This is a very long text. " * 100
    # # inputs = text_tokenizer(long_text, truncation=True, max_length=128)
    # # print(inputs)
