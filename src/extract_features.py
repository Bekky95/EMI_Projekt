from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
import pandas as pd


class ExtractFeatures:
    def __init__(self):
        self.records = []
        self.text_path = ""
        self.image_path = ""

        self.text_model_name = ""
        self.image_model_name = ""
        self.text_model = None
        self.image_model = None

        self.image_extractor = None

        self.load_models()

    def load_models(self):
        # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        self.text_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        # https://huggingface.co/openai/clip-vit-base-patch32?library=transformers
        self.image_model_name = "openai/clip-vit-base-patch32"

        # Text-Embeddings
        self.text_model = SentenceTransformer("all-MiniLM-L6-v2")
        # Bild-Embeddings
        self.image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()  # model
        self.image_extractor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")  # extractor

    def extract_text_embeddings(self):
        pass

    def text_embedding(self, caption):
        return self.text_model.encode(caption)  # (384,)

    def extract_image_embeddings(self):
        pass

    def image_embedding(self, img_path):
        img = Image.open(img_path).convert("RGB")
        inputs = self.image_extractor(images=img, return_tensors="pt")
        with torch.no_grad():
            feat = self.image_model.get_image_features(**inputs)  # (1, 512)
        return feat.squeeze().numpy()

    def save_embeddings(self):

        for path in all_files:
            h_t = self.text_embedding(self.text_path(path))
            h_i = self.image_embedding(self.image_path(path))
            label = parse_label(path)
            speaker = parse_speaker(path)
            self.records.append({"h_t": h_t, "h_i": h_i,
                            "label": label, "speaker": speaker})
            np.savez("text_image_features.npz",
                     h_t=np.stack([r["h_t"] for r in self.records]),
                     h_i=np.stack([r["h_i"] for r in self.records]),
                     label=np.array([r["label"] for r in self.records]),
                     speaker=np.array([r["speaker"] for r in self.records]))

    def split_train_test(self) :
        data = np.load("text_features.npz")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(data["h_t"], data["label"],
                                         groups=data["speaker"]))


    def train_model(self, model, loader_train, loader_val, n_epochs=30, lr=1e-3):
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()
        best_f1, best_state = 0.0, None

        for epoch in range(n_epochs):
            model.train()

            for h_t, h_i, y in loader_train:
                opt.zero_grad()
                logits = model(h_t, h_i)
                loss = crit(logits, y)
                loss.backward()
                opt.step()
            f1 = self.evaluate_macro_f1(model, loader_val)
            if f1 > best_f1:
                best_f1, best_state = f1, model.state_dict()
                model.load_state_dict(best_state)
        return model, best_f1

    def evaluate_macro_f1(self, model, loader):
        model.eval()

        y_true, y_pred = [], []
        with torch.no_grad():
            for h_a, h_v, y in loader:
                logits = model(h_a, h_v)
                y_pred.extend(logits.argmax(dim=-1).tolist())
                y_true.extend(y.tolist())
            return f1_score(y_true, y_pred, average="macro")