import torch
import torchaudio
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoModelForZeroShotImageClassification
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
import pandas as pd

#TODO: pretrained Models
# Text sentence-transformers/all-MiniLM-L6-v2 384
# Bild openai/clip-vit-base-patch32 512

#TODO: Datensatz:
# Memotion (Bild + Text, Sentiment auf Memes, klein)

#TODO: Methoden:
# Cross-Attention (uni- oder bidirektional)
# Early Fusion (Konkatenation, optional mit Projektion auf gemeinsame Dimension)

'''
ie zentrale Strategie heisst pre-extracted features: Sie nutzen vortrainierte Modelle, um
Ihre Audio- und Video-Daten einmalig in dichte Embeddings umzuwandeln. Diese Embeddings
speichern Sie als NumPy-Arrays. Das eigentliche Fusion-Training läuft danach nur noch auf diesen
Embeddings, nicht mehr auf den rohen Daten.
'''


if __name__ == "__main__":
    # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    text_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    # https://huggingface.co/openai/clip-vit-base-patch32?library=transformers
    image_model_name = "openai/clip-vit-base-patch32"

    # Text-Embeddings
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    # Bild-Embeddings
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval() #model
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")    #extractor


    #text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    #text_model = AutoModel.from_pretrained(text_model_name)

    #image_processor = AutoProcessor.from_pretrained(image_model_name)
    #image_model = AutoModelForZeroShotImageClassification.from_pretrained(image_model_name)

    # long_text = "This is a very long text. " * 100
    # inputs = text_tokenizer(long_text, truncation=True, max_length=128)
    # print(inputs)
