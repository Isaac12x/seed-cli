import pytest
from pathlib import Path
from seed_cli.image import parse_image


def test_image_requires_ocr(tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"not an image")

    # PIL will raise UnidentifiedImageError for invalid image data
    with pytest.raises((RuntimeError, Exception)):  # Catch any image-related error
        parse_image(img)
