from PIL import Image


def load_image(path):
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        print(f"Fehler beim Laden von {path}: {e}")
        return Image.new("RGB", (224, 224))  # Fallback