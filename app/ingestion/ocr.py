"""OCR fallback for image-heavy or scanned documents.

Uses pytesseract (Tesseract OCR) with pdf2image for PDF processing.
Falls back gracefully if Tesseract is not installed.
"""
import logging
import tempfile
import os
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Check availability at import time
PYTESSERACT_AVAILABLE = False
PDF2IMAGE_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    logger.warning("pytesseract not available — OCR features disabled")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    logger.warning("pdf2image not available — PDF OCR disabled")


def ocr_image(image_path: str) -> str:
    """
    Run OCR on a single image file.

    Args:
        image_path: Path to an image file (PNG, JPG, etc.)

    Returns:
        Extracted text, or empty string if OCR is unavailable.
    """
    if not PYTESSERACT_AVAILABLE:
        logger.warning(f"pytesseract unavailable, skipping OCR on {image_path}")
        return ""

    try:
        from PIL import Image

        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        logger.exception(f"OCR failed for image {image_path}: {e}")
        return ""


def ocr_pdf(pdf_path: str, dpi: int = 200) -> List[Dict[str, Any]]:
    """
    Run OCR on each page of a PDF.

    Args:
        pdf_path: Path to the PDF file.
        dpi:      Resolution for rendering PDF pages to images.

    Returns:
        List of dicts:
        [
            {
                "page_num": int,       # 1-indexed
                "text": str,           # extracted text
                "image_paths": list,  # temp image files created (to clean up)
            },
            ...
        ]
    """
    if not PYTESSERACT_AVAILABLE or not PDF2IMAGE_AVAILABLE:
        logger.warning(
            f"OCR dependencies missing — cannot OCR PDF {pdf_path}. "
            f"pytesseract={PYTESSERACT_AVAILABLE}, pdf2image={PDF2IMAGE_AVAILABLE}"
        )
        return []

    try:
        from pdf2image import convert_from_path

        # Convert PDF pages to PIL Images
        pages = convert_from_path(pdf_path, dpi=dpi)

        results = []
        for i, page_image in enumerate(pages, start=1):
            image_paths = []

            # Save page as temp image for OCR
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".png", mode="wb"
            ) as tmp_img:
                page_image.save(tmp_img, "PNG")
                img_path = tmp_img.name
                image_paths.append(img_path)

            text = pytesseract.image_to_string(page_image).strip()

            # Clean up temp image immediately after OCR
            for p in image_paths:
                try:
                    os.unlink(p)
                except Exception:
                    pass

            results.append({
                "page_num": i,
                "text": text,
                "image_paths": image_paths,
            })

        return results

    except Exception as e:
        logger.exception(f"PDF OCR failed for {pdf_path}: {e}")
        return []


def ocr_pages_from_images(image_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Run OCR on a list of image files.

    Args:
        image_paths: List of image file paths.

    Returns:
        List of dicts with page_num (1-indexed), text, image_paths.
    """
    if not PYTESSERACT_AVAILABLE:
        logger.warning("pytesseract unavailable, skipping batch OCR")
        return []

    results = []
    for i, img_path in enumerate(image_paths, start=1):
        text = ocr_image(img_path)
        results.append({
            "page_num": i,
            "text": text,
            "image_paths": [img_path],
        })

    return results
