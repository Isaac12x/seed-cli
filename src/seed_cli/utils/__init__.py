
"""seed_cli.utils

Utility functions for common operations.

These utilities provide convenient ways to perform common tasks
that might be useful for users.
"""

import re
from pathlib import Path
from typing import Optional, List, Tuple


def extract_tree_from_image(
    image_path: Path,
    output_path: Optional[Path] = None,
    *,
    vars: Optional[dict] = None,
    raw: bool = False,
) -> Path:
    """Extract tree structure from an image and save to a .tree file.
    
    Uses OCR to extract text from an image, then cleans OCR artifacts
    to produce a usable tree structure.
    
    Args:
        image_path: Path to the image file (.png, .jpg, .jpeg)
        output_path: Optional output path. If not provided, uses image_path with .tree extension
        vars: Optional template variables (not currently used, preserved for API compatibility)
        raw: If True, output raw OCR text without cleaning (for debugging)
    
    Returns:
        Path to the created .tree file
    
    Raises:
        RuntimeError: If image dependencies are not installed
        FileNotFoundError: If image_path doesn't exist
    
    Limitations:
        OCR of tree structures has inherent limitations:
        
        1. Tree characters (│├└─) are often misread by OCR as L, |, _, etc.
        2. The spatial indentation that shows hierarchy may be lost
        3. File/directory names may have spacing artifacts
        
        The output cleans filenames and tree characters but may lose
        some hierarchy information. Manual adjustment may be needed.
        
        For best results:
        - Use high-contrast images with clear tree structure
        - Consider using --raw to see OCR output and manually fix
        - The output extracts all items but hierarchy may be flattened
    """
    from seed_cli.image import read_tree, tree_lines_to_text
    
    # Check if image exists
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Determine output path
    if output_path is None:
        output_path = image_path.with_suffix(".tree")
    
    # Extract text from image using OCR with layout preservation
    raw_text = read_tree(image_path, mode="ocr")
    
    if raw:
        # Output raw OCR text for debugging
        output_path.write_text(tree_lines_to_text(raw_text), encoding="utf-8")
    else:
        # Clean and reconstruct tree structure
        cleaned_text = tree_lines_to_text(raw_text)
        output_path.write_text(cleaned_text, encoding="utf-8")
    
    return output_path


def has_image_support() -> bool:
    """Check if image/OCR dependencies are installed.
    
    Returns:
        True if pytesseract and PIL are available, False otherwise
    """
    try:
        import pytesseract  # noqa
        from PIL import Image  # noqa
        import cv2 # noqa
        import numpy as np # noqa

        return True
    except ImportError:
        return False
