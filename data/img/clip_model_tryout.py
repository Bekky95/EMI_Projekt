import numpy as np
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import CLIPProcessor, CLIPModel

from dataset import CLIPMemotionDataset
from extract_features import ExtractFeaturesKaggle


#TODO: pretrained Models
# Text sentence-transformers/all-MiniLM-L6-v2 384
# Bild openai/clip-vit-base-patch32 512

#TODO: Datensatz:
# Memotion (Bild + Text, Sentiment auf Memes, klein)

#TODO: Methoden:
# Cross-Attention (uni- oder bidirektional)
# Early Fusion (Konkatenation, optional mit Projektion auf gemeinsame Dimension)


def merge_sentiment(label):
    if label in ["positive", "very_positive"]:
        return "positive"
    elif label in ["negative", "very_negative"]:
        return "negative"
    else:
        return "neutral"


class CLIPClassifier(nn.Module):
    """
    Wraps a CLIPModel and adds a small classifier for 3-class sentiment.
    We'll:
      - get image_embeds from model outputs
      - get text_embeds from model outputs
      - combine them, then pass through a small feedforward
    """
    def __init__(self, model_name, num_labels=3, freeze_clip=False):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.num_labels = num_labels

        # Optionally freeze entire CLIP to reduce memory usage & avoid large updates
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        embed_dim = self.clip_model.config.projection_dim * 2  # e.g., 512 + 512 = 1024
        # Add a small classifier head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        # The CLIP forward pass:
        # returns image_embeds, text_embeds, etc.
        outputs = self.clip_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        # outputs.image_embeds: [batch_size, projection_dim] default=512
        # outputs.text_embeds:  [batch_size, projection_dim]

        # By default, CLIPModel output embeddings are already normalized (unit sphere)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        # Concatenate them for classification
        fused = torch.cat([image_embeds, text_embeds], dim=1)  # shape: [B, 1024]
        logits = self.classifier(fused)
        return logits


def epoch_step(model, dataloader, is_train=False):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        if is_train:
            optimizer.zero_grad()

        logits = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)

        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.detach().cpu().numpy())

        if is_train:
            loss.backward()
            optimizer.step()

    avg_loss = total_loss / len(dataloader.dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")
    return avg_loss, acc, prec, rec, f1


if __name__ == "__main__":
    # https://www.kaggle.com/code/vishwapatel214/clip-model

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    # 1) Load & Clean Data
    extract_features = ExtractFeaturesKaggle()
    #extract_features.load_and_save_dataset()

    IMAGES_DIR = extract_features.get_images_path()

    dataset = extract_features.load_dataset_from_dir()

    # 2) Merge 5 Original Classes into 3 (positive, negative, neutral)
    dataset["merged_sentiment"] = dataset["overall_sentiment"].apply(merge_sentiment)
    label2id = {"positive": 0, "negative": 1, "neutral": 2}
    dataset["label"] = dataset["merged_sentiment"].map(label2id)
    print("Merged distribution:")
    print(dataset["merged_sentiment"].value_counts(normalize=True))

    # 3) Stratified Split
    train_df, test_df = train_test_split(
        dataset, test_size=0.2, stratify=dataset["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        test_df, test_size=0.5, stratify=test_df["label"], random_state=42
    )

    # 4) CLIP Processor
    clip_model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(clip_model_name)

    # 5) Custom Dataset
    train_dataset = CLIPMemotionDataset(train_df, IMAGES_DIR, processor)
    val_dataset = CLIPMemotionDataset(val_df, IMAGES_DIR, processor)
    test_dataset = CLIPMemotionDataset(test_df, IMAGES_DIR, processor)

    BATCH_SIZE = 8
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 6) CLIP Classification Model
    num_labels = 3
    model = CLIPClassifier(
        model_name=clip_model_name,
        num_labels=num_labels,
        freeze_clip=False
    ).to(DEVICE)

    # 7) Optimizer / Scheduler
    # Fine-tuning CLIP can be costly. We do a small LR.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    # Tried both simple scheduler and ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1, verbose=True)

    criterion = nn.CrossEntropyLoss()

    # 8) Training / Evaluation Functions
    # --> def epoch_step()

    # 9) Training Loop (with early stopping)
    EPOCHS = 5
    patience = 2
    best_val_loss = float("inf")
    no_improve = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_prec, train_rec, train_f1 = epoch_step(model, train_loader, is_train=True)
        val_loss, val_acc, val_prec, val_rec, val_f1 = epoch_step(model, val_loader, is_train=False)

        # Step scheduler on val_loss
        scheduler.step(val_loss)

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(
            f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | P: {train_prec:.4f} | R: {train_rec:.4f} | F1: {train_f1:.4f}")
        print(
            f"  Val   Loss: {val_loss:.4f}   | Acc: {val_acc:.4f}   | P: {val_prec:.4f} | R: {val_rec:.4f} | F1: {val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    if best_state:
        model.load_state_dict(best_state)

    # Final Test
    test_loss, test_acc, test_prec, test_rec, test_f1 = epoch_step(model, test_loader, is_train=False)
    print("\n--- CLIP Test Results ---")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc:  {test_acc:.4f}")
    print(f"Macro Precision: {test_prec:.4f}")
    print(f"Macro Recall:    {test_rec:.4f}")
    print(f"Macro F1:        {test_f1:.4f}")

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

