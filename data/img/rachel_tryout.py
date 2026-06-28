import os
import torchaudio
import numpy as np
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoModelForZeroShotImageClassification

from dataset import CLIPMemotionDataset
from extract_features import ExtractFeaturesHuggingface, ExtractFeaturesKaggle
#from dataset import MemotionDataset
from pytorch_dataset import MemotionDataset

from helper.directory_functions import search_memotion_dataset_7k_dir
from extract_features import ExtractFeaturesHuggingface, ExtractFeaturesKaggle
from extract_features import ExtractFeaturesKaggle
from helper.makros import DEVICE
from models import miniLM_model
from models.miniLM_model import MiniLMModel
from models.clip_model import ClipModel


#TODO: pretrained Models
# Text sentence-transformers/all-MiniLM-L6-v2 384
# Bild openai/clip-vit-base-patch32 512

#TODO: Datensatz:
# Memotion (Bild + Text, Sentiment auf Memes, klein)

#TODO: Methoden:
# Cross-Attention (uni- oder bidirektional)
# Early Fusion (Konkatenation, optional mit Projektion auf gemeinsame Dimension)

"""
CFG = {
    "data_root"      : "/kaggle/input",
    "out_dir"        : "/kaggle/working",
    "clip_model"     : "ViT-B/32",      # <- ImageModel("openai/clip-vit-base-patch32")
    "roberta_model"  : "roberta-base",  # <- TextModel("all-MiniLM-L6-v2")
    "batch_size"     : 16,
    "epochs"         : 15,
    "lr_roberta"     : 2e-5,
    "lr_head"        : 5e-4,      # ← increased
    "weight_decay"   : 1e-2,
    "max_grad_norm"  : 1.0,
    "warmup_ratio"   : 0.1,
    "patience"       : 4,
    "max_text_len"   : 128,
    "img_size"       : 224,
    "task_weights"   : {"sentiment": 1.0, "humour": 0.8, "sarcasm": 0.8, "offensive": 0.8},
    "focal_gamma"    : 2.0,
    "memory_size"    : 32,
    "fusion_dim"     : 256,
}
os.makedirs(CFG["out_dir"], exist_ok=True)
print("Config ready ✅")
"""

if __name__ == "__main__":

    # https://www.kaggle.com/code/vishwapatel214/clip-model
    extract_features = ExtractFeaturesKaggle()
    #extract_features.load_and_save_dataset()

    dataset_labels = extract_features.load_dataset_from_dir()
    #print(f"dataset output: {dataset_labels}")  # for breakpoint
    print("Shape (rows, columns): ", dataset_labels.shape)
    print("Cols:", dataset_labels.columns.tolist())
    dataset_labels.head(3)

    IMG_DIR = extract_features.get_images_path()
    ## stole the code from a kaggle user
    #extract_features.label_distributions(dataset_labels)
    encoded_labels = extract_features.data_cleaning_and_label_encoding(dataset_labels, IMG_DIR)
    #print("encoded labels: ", encoded_labels["image_name"])
    #extract_features.sort_class_weights(encoded_labels)
    d_train, d_val, d_test = extract_features.train_validate_test_split(encoded_labels)
    #extract_features.DEBUG_label_distribution_check(encoded_labels, d_train, d_val, d_test) # just for checking

    clip_model = ClipModel()
    miniLM_model = MiniLMModel()

    clip_model.set_req_grad_to_false()
    CLIP_DIM = clip_model.get_clip_dimension()
    print(f"CLIP dim        : {CLIP_DIM}")

    miniLM_model.set_req_grad_to_true()
    MINILM_DIM = miniLM_model.get_model_dimension()
    trainable = miniLM_model.get_trainable_params()
    total = miniLM_model.get_all_params()
    print(f"MiniLM dim        : {MINILM_DIM}")
    print(f"MiniLM trainable: {trainable:,} / {total:,} params")
    print("Encoders loaded ✅")

    skipped = 0
    pooled_text = []
    pooled_img = []
    image_names = encoded_labels["image_name"]
    corrected_texts = encoded_labels["text_corrected"]

    # Embedding Test
    test_img = encoded_labels["image_name"][0]
    test_text = encoded_labels["text_corrected"][0]
    image_path = os.path.join(IMG_DIR, test_img)
    text_field = test_text
    h_t = miniLM_model.text_embedding(text_field)
    #h_i = clip_model.image_embedding(image_path)
    h_i = clip_model.clip_forward()
    print("test")

    # 3) Stratified Split
    train_df, test_df = train_test_split(
        dataset, test_size=0.2, stratify=dataset["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        test_df, test_size=0.5, stratify=test_df["label"], random_state=42
    )

    # 5) Custom Dataset
    train_dataset = CLIPMemotionDataset(train_df, IMG_DIR, clip_model.get_image_preprocessor())
    val_dataset = CLIPMemotionDataset(val_df, IMG_DIR, clip_model.get_image_preprocessor())
    test_dataset = CLIPMemotionDataset(test_df, IMG_DIR, clip_model.get_image_preprocessor())

    BATCH_SIZE = 8
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    # for name, text in zip(image_names, corrected_texts):
    # #for row in encoded_labels:
    #     image_path = os.path.join(IMG_DIR, name)
    #     text_field = text
    #     if not os.path.exists(image_path):
    #         skipped += 1
    #         continue
    #     try:
    #         h_t = miniLM_model.text_embedding(text_field)
    #         h_i = clip_model.image_embedding(image_path)
    #     except Exception as e:
    #         print(f"Ueberspringe {name}: {e}")
    #         skipped += 1
    #         continue
    #
    #     pooled_text.append(h_t)
    #     pooled_img.append(h_i)
    #
    #
    # if skipped:
    #     print(f"{skipped} Eintraege uebersprungen (Bild fehlt oder Fehler).")
    #
    # save_dict = dict(
    #     h_t=np.stack(pooled_text),
    #     h_i=np.stack(pooled_img),
    # )

    # TODO Für was wird MemotionDataset jetzt benötigt??
    # img_size = 224
    # max_text_len = 128
    # # Image augmentation for training
    # TRAIN_TRANSFORMS = transforms.Compose([
    #     transforms.Resize((img_size, img_size)),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    #     transforms.RandomRotation(degrees=10),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
    #                          (0.26862954, 0.26130258, 0.27577711)),  # CLIP mean/std
    # ])
    #
    # VAL_TRANSFORMS = transforms.Compose([
    #     transforms.Resize((img_size, img_size)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
    #                          (0.26862954, 0.26130258, 0.27577711)),
    # ])
    # mini_tokenizer = miniLM_model.get_tokenizer()
    #
    # train_ds = MemotionDataset(d_train, IMG_DIR, mini_tokenizer, max_text_len, TRAIN_TRANSFORMS)
    # val_ds = MemotionDataset(d_val, IMG_DIR, mini_tokenizer, max_text_len, VAL_TRANSFORMS)
    # test_ds = MemotionDataset(d_test, IMG_DIR, mini_tokenizer, max_text_len, VAL_TRANSFORMS)
    #
    # print(f"Dataset sizes — Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")



    # train_dataset = MemotionDataset(datensatz)
    #
    # image, content, role = train_dataset[0]
    #
    # print(content)
    # print(role)
    # print(image.height, image.width)

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
