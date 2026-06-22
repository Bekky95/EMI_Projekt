from PIL import Image
import io


def bytes_to_pil(b):
    """JPEG-Bytes -> PIL-Image"""
    return Image.open(io.BytesIO(b))
