# Text sentence-transformers/all-MiniLM-L6-v2 384

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
TEXT_MODEL = SentenceTransformer(MODEL_NAME)

class MiniLMModel:
    def __init__(self):
        self.model_name = MODEL_NAME
        self.text_model = TEXT_MODEL
        self.tokenizer = TEXT_MODEL.tokenizer

    def get_model(self):
        return self.text_model

    def get_tokenizer(self):
        return self.tokenizer

    def get_model_dimension(self):
        return self.text_model.get_embedding_dimension()

    def set_req_grad_to_true(self):
        # ← Unfreeze ALL layers
        for param in self.text_model.parameters():
            param.requires_grad = True

    def get_trainable_params(self):
        return sum(p.numel() for p in self.text_model.parameters() if p.requires_grad)

    def get_all_params(self):
        return sum(p.numel() for p in self.text_model.parameters())

    def text_embedding(self, caption):
        return self.text_model.encode(caption) # (384,)

    # def train_text():
    #     return None