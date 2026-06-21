from dataset import MemotionDataset
from models.cross_attention import CrossAttentionFusion
from torch.utils.data import DataLoader
import torch


def train_epoch(model, loader, opt, loss_fn):
    model.train()

    for t, i, y in loader:
        opt.zero_grad()

        out = model(t, i)
        loss = loss_fn(out, y)

        loss.backward()
        opt.step()


def main():
    dataset = MemotionDataset("features/memotion.npz")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = CrossAttentionFusion()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(10):
        train_epoch(model, loader, opt, loss_fn)
        print("epoch", epoch)


if __name__ == "__main__":
    main()