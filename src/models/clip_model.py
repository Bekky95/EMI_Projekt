# Bild openai/clip-vit-base-patch32 512

'''
CLIP ist von Haus aus auf Bild-Text-Alignment trainiert. Eine extrem starke Baseline auf Hateful
Memes ist daher: Cosinus-Ähnlichkeit zwischen CLIP-Text- und CLIP-Bild-Embedding, plus ein
lineares Modell drauf. Wenn Ihre Fusion-Methoden diese Baseline nicht schlagen, ist das interessant
und gehört in die Diskussion.
'''

from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPVisionModel, CLIPFeatureExtractor
from PIL import Image
import torch

MODEL_NAME = "openai/clip-vit-base-patch32"

clip = CLIPModel.from_pretrained(MODEL_NAME).eval()
proc = CLIPProcessor.from_pretrained(MODEL_NAME) # Constructs a CLIP processor which wraps a CLIP feature extractor and a CLIP tokenizer into a single processor.
tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)
clip_vision = CLIPVisionModel.from_pretrained(MODEL_NAME)

class ClipModel:
    def __init__(self):
        self.model = clip
        self.pre_processor = proc
        self.tokenizer = tokenizer

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

# TODO image embeddings mit ClipVisionModel testen? oder CLIPFeatureExtractor
#  https://huggingface.co/transformers/v4.6.0/model_doc/clip.html
    def image_embedding(self, img_path):
        img = Image.open(img_path).convert("RGB")
        inputs = proc(images=img, return_tensors="pt")

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            feat = model.get_image_features(**inputs) # (1, 512)

            # Normalize the embedding
            #image_features = feat / feat.norm(p=2, dim=-1, keepdim=True)

        return feat.detach().cpu().numpy()
        #return feat.squeeze().numpy()