

"""seed_cli.image

Parse a directory structure from an image (PNG/JPG).

Design:
- This module is intentionally *best-effort*
- It extracts text via OCR
- Then feeds the extracted text into the normal parser pipeline

Optional dependency:
- pytesseract
- pillow

If OCR is unavailable, a clear error is raised.
"""

from pathlib import Path
from typing import Tuple, Optional, List

from .parsers import parse_any, Node


def _require_ocr():
    try:
        import pytesseract  # noqa
        from PIL import Image  # noqa
    except Exception as e:
        raise RuntimeError(
            "OCR support requires optional dependencies: "
            "pip install seed-cli[image]"
        ) from e


def extract_text_from_image(path: Path) -> str:
    """Extract raw text from an image using OCR."""
    _require_ocr()
    from PIL import Image
    import pytesseract

    img = Image.open(path)
    return pytesseract.image_to_string(img)


def parse_image(
    image_path: Path,
    *,
    vars: Optional[dict] = None,
    mode: str = "loose",
) -> Tuple[Optional[Path], List[Node]]:
    """Parse directory structure from an image.

    Steps:
    1. OCR image -> text
    2. Delegate to parse_any()
    """
    text = extract_text_from_image(image_path)
    return parse_any(str(image_path), text, vars=vars, mode=mode)
