import torch
import torchaudio
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoModelForZeroShotImageClassification
import matplotlib.pyplot as plt

from extract_features import ExtractFeatures
from helper.image_functions import bytes_to_pil

#TODO: pretrained Models
# Text sentence-transformers/all-MiniLM-L6-v2 384
# Bild openai/clip-vit-base-patch32 512

#TODO: Datensatz:
# Memotion (Bild + Text, Sentiment auf Memes, klein)

#TODO: Methoden:
# Cross-Attention (uni- oder bidirektional)
# Early Fusion (Konkatenation, optional mit Projektion auf gemeinsame Dimension)

if __name__ == "__main__":

    extract_features = ExtractFeatures()
    datensatz = extract_features.load_dataset_from_dir()

    #print(datensatz['train'].features)  #{'messages': List({'role': Value('string'), 'content': Value('string')}), 'images': List(Image(mode=None, decode=True))}

    #print(datensatz['train'][0]['images'])  #[<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=735x500 at 0x1D186212310>]

    var = datensatz['train'][3]['images'][0]

    import random

    fig, axes = plt.subplots(2, 7, figsize=(14, 5))
    axes = axes.flatten()

    indices = random.sample(range(len(datensatz['train'])), 14)

    for ax, idx in zip(axes, indices):
        img = datensatz['train'][idx]['images'][0]
        label = datensatz['train'][idx]['messages'][0]['content']

        ax.imshow(img, cmap='gray')
        ax.set_title(label)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

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
