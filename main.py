import torch
import torchaudio
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoModelForZeroShotImageClassification

from dataset import MemotionDataset
from extract_features import ExtractFeaturesHuggingface, ExtractFeaturesKaggle
from helper.directory_functions import search_memotion_dataset_7k_dir
from models.text_model import text_embedding
from models.image_model import image_embedding

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

    dataset_labels = extract_features.load_dataset_from_dir()
    #print(f"dataset output: {dataset}")  # for breakpoint
    print("Shape (rows, columns): ", dataset_labels.shape)
    print("Cols:", dataset_labels.columns.tolist())
    dataset_labels.head(3)

    ## stole the code from a kaggle user
    #extract_features.label_distributions(dataset_labels)
    encoded_labels = extract_features.data_cleaning_and_label_encoding(dataset_labels)
    #extract_features.sort_class_weights(encoded_labels)
    d_train, d_val, d_test = extract_features.train_validate_test_split(encoded_labels)
    extract_features.DEBUG_label_distribution_check(encoded_labels, d_train, d_val, d_test)



    # TODO embeddings testen
    # h_t = text_embedding(dataset)
    # print(h_t)
    # h_i = image_embedding(dataset)
    # print(h_i)

    ## Iteration über den ganzen Datensatz und alles in einer Datei speichern
    '''
    # Audio und Video Embedding Beispiel
    records = []
    for path in all_files:
        h_a = audio_embedding(audio_path(path))
        h_v = video_embedding(video_path(path))
        label = parse_label(path)
        speaker = parse_speaker(path)
        records.append({"h_a": h_a, "h_v": h_v,
                        "label": label, "speaker": speaker})                  
        np.savez("ravdess_features.npz",
                h_a=np.stack([r["h_a"] for r in records]),
                h_v=np.stack([r["h_v"] for r in records]),
                label=np.array([r["label"] for r in records]),
                speaker=np.array([r["speaker"] for r in records]))
    '''

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
