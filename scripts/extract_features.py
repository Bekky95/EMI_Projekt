import pandas as pd
from sentence_transformers import SentenceTransformer
import ast
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from datasets import load_dataset

# Models
text_model = SentenceTransformer("all-MiniLM-L6-v2")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def image_embedding_pil(cur_image):
    inputs = clip_processor(images=cur_image, return_tensors="pt")

    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)

    if hasattr(outputs, "numpy"):
        return outputs.numpy()

    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def text_embedding(text):
    return text_model.encode(text)


def image_embedding(path):
    img = Image.open(path).convert("RGB")
    inputs = clip_processor(images=img, return_tensors="pt")

    with torch.no_grad():
        feat = clip_model.get_image_features(**inputs)

    return feat.squeeze().numpy()


if __name__ == "__main__":
    """
    Traceback (most recent call last):
  File "C:\Users\rebek\Documents\EMI_Projekt\.venv\venv\Lib\site-packages\PIL\ImageFile.py", line 392, in load
    s = read(read_bytes)
        ^^^^^^^^^^^^^^^^
  File "C:\Users\rebek\Documents\EMI_Projekt\.venv\venv\Lib\site-packages\PIL\PngImagePlugin.py", line 999, in load_read
    cid, pos, length = self.png.read()
                       ^^^^^^^^^^^^^^^
  File "C:\Users\rebek\Documents\EMI_Projekt\.venv\venv\Lib\site-packages\PIL\PngImagePlugin.py", line 179, in read
    length = i32(s)
             ^^^^^^
  File "C:\Users\rebek\Documents\EMI_Projekt\.venv\venv\Lib\site-packages\PIL\_binary.py", line 96, in i32be
    return unpack_from(">I", c, o)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^
struct.error: unpack_from requires a buffer of at least 4 bytes for unpacking 4 bytes at offset 0 (actual buffer size is 0)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\rebek\Documents\EMI_Projekt\scripts\extract_features.py", line 80, in <module>
    for sample in dataset["train"]:
  File "C:\Users\rebek\Documents\EMI_Projekt\.venv\venv\Lib\site-packages\datasets\arrow_dataset.py", line 2765, in __iter__
    formatted_output = format_table(
                       ^^^^^^^^^^^^^
  File "C:\Users\rebek\Documents\EMI_Projekt\.venv\venv\Lib\site-packages\datasets\formatting\formatting.py", line 658, in format_table
    return formatter(pa_table, query_type=query_type)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\rebek\Documents\EMI_Projekt\.venv\venv\Lib\site-packages\datasets\formatting\formatting.py", line 411, in __call__
    return self.format_row(pa_table)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\rebek\Documents\EMI_Projekt\.venv\venv\Lib\site-packages\datasets\formatting\formatting.py", line 460, in format_row
    row = self.python_features_decoder.decode_row(row)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\rebek\Documents\EMI_Projekt\.venv\venv\Lib\site-packages\datasets\formatting\formatting.py", line 224, in decode_row
    return self.features.decode_example(row, token_per_repo_id=self.token_per_repo_id) if self.features else row
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\rebek\Documents\EMI_Projekt\.venv\venv\Lib\site-packages\datasets\features\features.py", line 2213, in decode_example
    return {
           ^
  File "C:\Users\rebek\Documents\EMI_Projekt\.venv\venv\Lib\site-packages\datasets\features\features.py", line 2214, in <dictcomp>
    column_name: decode_nested_example(feature, value, token_per_repo_id=token_per_repo_id)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\rebek\Documents\EMI_Projekt\.venv\venv\Lib\site-packages\datasets\features\features.py", line 1511, in decode_nested_example
    if decode_nested_example(sub_schema, first_elmt) != first_elmt:
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\rebek\Documents\EMI_Projekt\.venv\venv\Lib\site-packages\datasets\features\features.py", line 1517, in decode_nested_example
    return schema.decode_example(obj, token_per_repo_id=token_per_repo_id) if obj is not None else None
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\rebek\Documents\EMI_Projekt\.venv\venv\Lib\site-packages\datasets\features\image.py", line 193, in decode_example
    image.load()  # to avoid "Too many open files" errors
    ^^^^^^^^^^^^
  File "C:\Users\rebek\Documents\EMI_Projekt\.venv\venv\Lib\site-packages\PIL\ImageFile.py", line 399, in load
    raise OSError(msg) from e
OSError: image file is truncated
    """
    dataset = load_dataset("Leonardo6/memotion")

    records = []

    for sample in dataset["train"]:
        # TEXT
        text = sample["messages"][0]["content"]

        # IMAGE
        image = sample["images"][0]

        # LABEL (LISTE!)
        label = ast.literal_eval(sample["messages"][1]["content"])

        t = text_embedding(text)
        i = image_embedding_pil(image)

        records.append((t, i, label))

    np.savez(
        "features/memotion/memotion.npz",
        text=np.array([r[0] for r in records]),
        image=np.array([r[1] for r in records]),
        label=np.array([r[2] for r in records]),
    )
