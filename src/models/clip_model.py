# Bild openai/clip-vit-base-patch32 512

'''
CLIP ist von Haus aus auf Bild-Text-Alignment trainiert. Eine extrem starke Baseline auf Hateful
Memes ist daher: Cosinus-Ähnlichkeit zwischen CLIP-Text- und CLIP-Bild-Embedding, plus ein
lineares Modell drauf. Wenn Ihre Fusion-Methoden diese Baseline nicht schlagen, ist das interessant
und gehört in die Diskussion.
'''

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class ClipModel:
    def __init__(self):
        self.model = clip
        self.pre_processor = proc # kaggle user uses cpu

    def get_image_model(self):
        return self.model

    def get_image_preprocessor(self):
        return self.pre_processor

    def get_clip_dimension(self):
        return self.model.vision_embed_dim
        #return self.model.visual.output_dim

    def set_req_grad_to_false(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def image_embedding(self, img_path):
        img = Image.open(img_path).convert("RGB")
        inputs = proc(images=img, return_tensors="pt")
        with torch.no_grad():
            feat = clip.get_image_features(**inputs) # (1, 512)
        return feat.squeeze().numpy()