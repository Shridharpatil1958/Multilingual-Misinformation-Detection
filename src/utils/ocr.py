# src/utils/ocr.py

"""
OCR pipeline for extracting text from WhatsApp forward screenshots.
Supports both Tesseract (fast) and EasyOCR (more accurate for Indic scripts).
"""

import base64
import io
from typing import Optional
from PIL import Image
from src.utils.logger import get_logger

logger = get_logger(__name__)


def decode_image_b64(image_b64: str) -> Image.Image:
    """Decode base64 image string to PIL Image."""
    if "," in image_b64:
        image_b64 = image_b64.split(",")[1]
    img_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def preprocess_image(img: Image.Image) -> Image.Image:
    """
    Enhance image for better OCR:
    - Convert to grayscale
    - Resize if too small
    - Basic contrast enhancement
    """
    from PIL import ImageEnhance, ImageFilter

    # Convert to grayscale
    img = img.convert("L")

    # Resize if too small (Tesseract works better at 300+ DPI equivalent)
    min_dim = 1000
    if min(img.size) < min_dim:
        scale = min_dim / min(img.size)
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)

    # Enhance contrast
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = img.filter(ImageFilter.SHARPEN)

    return img


def extract_text_tesseract(img: Image.Image, lang: str = "eng+hin+tam") -> str:
    """
    Extract text using Tesseract OCR.
    Supports: eng (English), hin (Hindi/Devanagari), tam (Tamil).
    Install: sudo apt-get install tesseract-ocr tesseract-ocr-hin tesseract-ocr-tam
    """
    try:
        import pytesseract
        img = preprocess_image(img)
        text = pytesseract.image_to_string(img, lang=lang)
        return text.strip()
    except ImportError:
        logger.warning("pytesseract not installed. Falling back to EasyOCR.")
        return extract_text_easyocr(img)
    except Exception as e:
        logger.error(f"Tesseract OCR failed: {e}")
        return ""


def extract_text_easyocr(img: Image.Image, languages: list = None) -> str:
    """
    Extract text using EasyOCR (better for mixed-script and low-quality images).
    Languages: ['en', 'hi', 'ta']
    """
    try:
        import easyocr
        if languages is None:
            languages = ["en", "hi", "ta"]

        reader = easyocr.Reader(languages, gpu=False, verbose=False)
        img_array = __import__("numpy").array(img)
        results = reader.readtext(img_array, detail=0)
        return " ".join(results).strip()
    except ImportError:
        raise RuntimeError("Neither pytesseract nor easyocr is installed.")
    except Exception as e:
        logger.error(f"EasyOCR failed: {e}")
        return ""


def extract_text_from_image_b64(image_b64: str, prefer: str = "tesseract") -> str:
    """
    Main OCR entry point.
    Args:
        image_b64: Base64-encoded image (with or without data URI prefix)
        prefer: "tesseract" or "easyocr"
    Returns:
        Extracted text string
    """
    img = decode_image_b64(image_b64)

    if prefer == "easyocr":
        text = extract_text_easyocr(img)
    else:
        text = extract_text_tesseract(img)

    if not text:
        logger.info("Tesseract returned empty, falling back to EasyOCR")
        text = extract_text_easyocr(img)

    logger.info(f"OCR extracted {len(text)} characters")
    return text
