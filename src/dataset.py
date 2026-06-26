from PIL.JpegImagePlugin import JpegImageFile
from torch.utils.data import Dataset
# https://arrow.apache.org/docs/python/ -> .arrow file handling


#TODO: Anpassen oder neue für keggle dataset
class MemotionDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset["train"]
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]

        #TODO: für Datenanalyse, was ist im Datensatz so drin:
        var = sample["messages"][0]
        var2 = sample["images"]

        image: JpegImageFile = sample["images"][0]
        content: str = sample["messages"][0]["content"]
        role: str = sample["messages"][0]['role']

        if self.transform:
            image = self.transform(image)

        return image, content, role
