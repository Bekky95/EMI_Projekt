import torch.nn as nn
from torchvision import models
from transformers import BertModel


class ImageEncoder(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        base = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.proj = nn.Linear(512, dim)

    def forward(self, x):
        x = self.backbone(x).squeeze(-1).squeeze(-1)
        return self.proj(x)


class TextEncoder(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.proj = nn.Linear(768, dim)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids,
                        attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.proj(cls)
