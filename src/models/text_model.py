# Text sentence-transformers/all-MiniLM-L6-v2 384

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
TEXT_MODEL = SentenceTransformer(MODEL_NAME)

class TextModel:
    def __init__(self):
        self.model_name = MODEL_NAME
        self.text_model = TEXT_MODEL


def text_embedding(self, caption):
    return self.text_model.encode(caption) # (384,)

def train_text():
    return None