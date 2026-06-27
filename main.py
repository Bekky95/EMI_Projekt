import torch
import torchaudio
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoModelForZeroShotImageClassification

from dataset import MemotionDataset
from extract_features import ExtractFeaturesHuggingface, ExtractFeaturesKaggle
from helper.directory_functions import search_memotion_dataset_7k_dir
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
    #print(f"dataset output: {dataset}")  # for breakpoint
    print("Shape (rows, columns): ", dataset_labels.shape)
    print("Cols:", dataset_labels.columns.tolist())
    dataset_labels.head(3)

    ## stole the code from a kaggle user
    #extract_features.label_distributions(dataset_labels)
    encoded_labels = extract_features.data_cleaning_and_label_encoding(dataset_labels)
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

    #h_t = miniLM_model.text_embedding(encoded_labels)
    #print(h_t)
    #h_i = clip_model.image_embedding(extract_features.get_images_path())
    #print(h_i)

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
