import torch, torchaudio, numpy as np
from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoModelForZeroShotImageClassification

#TODO: pretrained Models
# Text sentence-transformers/all-MiniLM-L6-v2 384
# Bild openai/clip-vit-base-patch32 512

#TODO: Datensatz:
# Memotion (Bild + Text, Sentiment auf Memes, klein)

#TODO: Methoden:
# Cross-Attention (uni- oder bidirektional)
# Early Fusion (Konkatenation, optional mit Projektion auf gemeinsame Dimension)

if __name__ == "__main__":
    text_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    image_model_name = "openai/clip-vit-base-patch32"

    #text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    text_model = AutoModel.from_pretrained(text_model_name)

    image_processor = AutoProcessor.from_pretrained(image_model_name)
    image_model = AutoModelForZeroShotImageClassification.from_pretrained(image_model_name)

    # long_text = "This is a very long text. " * 100
    # inputs = text_tokenizer(long_text, truncation=True, max_length=128)
    # print(inputs)
