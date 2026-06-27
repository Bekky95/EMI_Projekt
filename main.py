import torch
import torchaudio
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoModelForZeroShotImageClassification

from dataset import MemotionDataset
from extract_features import ExtractFeaturesHuggingface, ExtractFeaturesKaggle
from helper.directory_functions import search_memotion_dataset_7k_dir

#TODO: pretrained Models
# Text sentence-transformers/all-MiniLM-L6-v2 384
# Bild openai/clip-vit-base-patch32 512

#TODO: Datensatz:
# Memotion (Bild + Text, Sentiment auf Memes, klein)

#TODO: Methoden:
# Cross-Attention (uni- oder bidirektional)
# Early Fusion (Konkatenation, optional mit Projektion auf gemeinsame Dimension)

if __name__ == "__main__":

    # https://www.kaggle.com/code/vishwapatel214/clip-model
    extract_features = ExtractFeaturesKaggle()
    #extract_features.load_and_save_dataset()

    dataset = extract_features.load_dataset_from_dir()
    print("dfghjk")  # for breakpoint

    #
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
