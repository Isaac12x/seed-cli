

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
from __future__ import annotations

from pathlib import Path

from dataclasses import dataclass
from typing import Any, BinaryIO, Dict, Iterable, List, Optional, Tuple, Union, Optional
import io
import os
import re
from .parsers import parse_any, Node


TREE_REPLACEMENTS = {
    "|": "│",
    "+": "├──",
    "\\": "└──",
    "-": "──",
}

TREE_PREFIX_RE = re.compile(r"^(│\s+|├──|└──|\s+)+")

def _require_ocr():
    try:
        import cv2
        import numpy as np
        from PIL import Image # noqa
        import pytesseract # noqa
    except Exception as e:
        raise RuntimeError(
            "OCR support requires optional dependencies: "
            "pip install seed-cli[image]"
        ) from e


def _require_pytesseract():
    try:
        import pytesseract
        from pytesseract import Output
        return pytesseract, Output
    except ImportError as e:
        raise RuntimeError(
            "OCR support requires optional dependencies: "
            "pip install seed-cli[image]"
        ) from e


def _require_cv2():
    try:
        import cv2
        return cv2
    except ImportError as e:
        raise RuntimeError(
            "OCR support requires optional dependencies: "
            "pip install seed-cli[image]"
        ) from e


def _require_numpy():
    try:
        import numpy as np
        return np
    except ImportError as e:
        raise RuntimeError(
            "OCR support requires optional dependencies: "
            "pip install seed-cli[image]"
        ) from e

def _require_pymupdf():
    try:
        import fitz
        return fitz
    except ImportError as e:
        raise RuntimeError(
            "OCR support requires optional dependencies: "
            "pip install seed-cli[image]"
        ) from e

# old implementation
def extract_text_from_image(path: Path, preserve_layout: bool = True) -> str:
    """Extract raw text from an image using OCR.
    
    Args:
        path: Path to the image file
        preserve_layout: If True, use OCR settings that preserve spatial layout
                        (important for tree structures)

    
    """
    _require_ocr()
    from PIL import Image, ImageEnhance, ImageFilter
    pytesseract, Output = _require_pytesseract()

    img = Image.open(path)
    
    # Convert to grayscale for better OCR
    if img.mode != 'L':
        img = img.convert('L')
    
    # Enhance contrast to make tree characters more readable
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    
    # Sharpen to improve character recognition
    img = img.filter(ImageFilter.SHARPEN)
    
    if preserve_layout:
        # PSM 6: Assume a single uniform block of text - preserves layout
        # OEM 3: Default OCR Engine Mode
        config = '--psm 6 --oem 3 -c preserve_interword_spaces=1'
        return pytesseract.image_to_string(img, config=config)
    else:
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
    # text = extract_text_from_image(image_path)
    text = read_tree(image_path, mode="ocr")
    normalize_tree(text)
    return parse_any(str(image_path), text, vars=vars, mode=mode)


# -----------------------------
# Public types
# -----------------------------

@dataclass(frozen=True)
class TreeLine:
    text: str
    depth: int
    confidence: float  # 0..1 (best-effort for non-OCR fallback)


Source = Union[str, bytes, bytearray, BinaryIO]


# -----------------------------
# Streaming loaders (image/PDF)
# -----------------------------

def _read_all_bytes(src: Source) -> bytes:
    if isinstance(src, (bytes, bytearray)):
        return bytes(src)
    if hasattr(src, "read"):
        return src.read()
    # else assume path
    with open(str(src), "rb") as f:
        return f.read()


def load_pages_as_images(
    src: Source,
    *,
    page: Optional[int] = None,
    dpi: int = 200,
) -> List[np.ndarray]:
    """
    Returns list of BGR images (OpenCV).
    - If src is an image -> [image]
    - If src is a PDF -> list of rendered pages (or a single page if page is set)
    """
    # Path-based sniff
    cv2 = _require_cv2()
    if isinstance(src, str):
        ext = os.path.splitext(src.lower())[1]
        if ext == ".pdf":
            return _load_pdf_pages_from_path(src, page=page, dpi=dpi)
        img = cv2.imread(src)
        if img is None:
            raise FileNotFoundError(src)
        return [img]

    # Bytes/file-like sniff
    b = _read_all_bytes(src)

    if b[:4] == b"%PDF":
        return _load_pdf_pages_from_bytes(b, page=page, dpi=dpi)

    img = _decode_image_bytes(b)
    return [img]


def _decode_image_bytes(b: bytes) -> np.ndarray:
    cv2 = _require_cv2()
    np = _require_numpy()

    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image bytes.")
    return img


def _load_pdf_pages_from_path(path: str, *, page: Optional[int], dpi: int) -> List[np.ndarray]:
    fitz = _require_pymupdf()
    np = _require_numpy()
    doc = fitz.open(path)
    return _render_pdf_pages(doc, page=page, dpi=dpi)


def _load_pdf_pages_from_bytes(b: bytes, *, page: Optional[int], dpi: int) -> List[np.ndarray]:
    fitz = _require_pymupdf()
    np = _require_numpy()
    doc = fitz.open(stream=b, filetype="pdf")
    return _render_pdf_pages(doc, page=page, dpi=dpi)


def _render_pdf_pages(doc: Any, *, page: Optional[int], dpi: int) -> List[np.ndarray]:
    # PyMuPDF scaling: 72 dpi base
    fitz = _require_pymupdf()
    np = _require_numpy()
    cv2 = _require_cv2()

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    pages = [page] if page is not None else list(range(doc.page_count))
    out: List[np.ndarray] = []

    for pno in pages:
        p = doc.load_page(pno)
        pix = p.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        # PyMuPDF yields RGB; OpenCV uses BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.append(img)

    return out


# -----------------------------
# Preprocess
# -----------------------------

def preprocess_for_tree(img_bgr: np.ndarray) -> np.ndarray:
    """
    Returns binarized image (uint8 0/255) tuned for ASCII tree screenshots.
    """
    cv2 = _require_cv2()
    np = _require_numpy()

    if not isinstance(img_bgr, np.ndarray):
        raise TypeError("img_bgr must be a NumPy ndarray")

    img_bgr = cv2.resize(
        img_bgr,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC,
    )

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # slight denoise without killing thin glyphs
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, bw = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return bw


# -----------------------------
# OCR path (Tesseract)
# -----------------------------

def ocr_words_with_boxes(bw: np.ndarray) -> Dict[str, List]:
    pytesseract, TessOutput = _require_pytesseract()
    config = (
        "--psm 11 "
        "-c preserve_interword_spaces=1 "
        "-c tessedit_char_whitelist=│├└─|+\\-_./[]()"
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )
    return pytesseract.image_to_data(
        bw,
        output_type=TessOutput.DICT,
        config=config
    )


def group_words_into_lines(
    ocr: Dict[str, List],
    *,
    y_bucket: int = 10,
) -> Dict[int, List[Tuple[int, int, int, str, float]]]:
    """
    Returns {line_key: [(x, y, w, text, conf), ...]}.
    """
    lines: Dict[int, List[Tuple[int, int, int, str, float]]] = {}
    n = len(ocr["text"])

    for i in range(n):
        txt = (ocr["text"][i] or "").strip("\n")
        if not txt.strip():
            continue

        x = int(ocr["left"][i])
        y = int(ocr["top"][i])
        w = int(ocr["width"][i])
        conf = float(ocr["conf"][i])  # -1 may appear

        key = round(y / y_bucket) * y_bucket
        lines.setdefault(key, []).append((x, y, w, txt, conf))

    return lines


def _estimate_char_width(tokens: List[Tuple[int, int, int, str, float]]) -> float:
    """
    Estimate monospace character width from OCR tokens: median(width / len(text)).
    """
    np = _require_numpy()
    samples = []
    for (_x, _y, w, txt, conf) in tokens:
        t = txt.strip()
        if not t or len(t) == 0:
            continue
        # skip ultra-low confidence junk
        if conf >= 0 and conf < 20:
            continue
        samples.append(w / max(1, len(t)))

    if not samples:
        return 8.0
    return float(np.median(samples))


def reconstruct_lines_with_drift_repair(
    lines: Dict[int, List[Tuple[int, int, int, str, float]]],
    *,
    min_gap_cols: int = 1,
) -> List[Tuple[str, float]]:
    """
    Auto-repair indentation drift:
    - Snap OCR tokens to a fixed-width column grid derived from estimated char_width.
    - Insert spaces based on column gaps.
    Returns list of (line_text, line_confidence_0_1).
    """
    # Flatten all tokens to estimate global char width + min x
    all_tokens = [t for y in lines for t in lines[y]]
    char_w = _estimate_char_width(all_tokens)
    min_x = min((t[0] for t in all_tokens), default=0)

    out: List[Tuple[str, float]] = []

    for y in sorted(lines):
        toks = sorted(lines[y], key=lambda t: t[0])

        # Build (col, text)
        placed: List[Tuple[int, str]] = []
        confs: List[float] = []

        for (x, _yy, w, txt, conf) in toks:
            col = int(round((x - min_x) / max(char_w, 1e-6)))
            placed.append((max(0, col), txt))
            if conf >= 0:
                confs.append(conf)

        placed.sort(key=lambda t: t[0])

        # Stitch with spaces based on column positions
        cursor = 0
        parts: List[str] = []
        for col, txt in placed:
            if col > cursor:
                gap = col - cursor
                # ensure at least one space between tokens when OCR collapses them
                if gap < min_gap_cols:
                    gap = min_gap_cols
                parts.append(" " * gap)
                cursor = col

            parts.append(txt)
            cursor += len(txt)

        line_text = "".join(parts)
        line_conf = (sum(confs) / len(confs)) / 100.0 if confs else 0.0
        out.append((line_text, float(round(line_conf, 3))))

    return out


# -----------------------------
# Tree glyph normalization + depth
# -----------------------------

def normalize_tree_glyphs(s: str) -> str:
    for k, v in TREE_REPLACEMENTS.items():
        s = s.replace(k, v)
    return s


def compute_tree_depth(line: str) -> int:
    """
    Depth derived from leading tree structure.
    Best-effort for common tree output patterns.
    """
    depth = 0
    for ch in line:
        if ch == "│":
            depth += 1
        elif ch in ("├──", "└──"):
            depth += 1
            break
        elif ch == " ":
            continue
        else:
            break
    return depth

def normalize_branch_width(line: str) -> str:
    # Normalize all variants to ├── / └──
    line = line.replace("├─", "├──")
    line = line.replace("└─", "└──")
    line = line.replace("├───", "├──")
    line = line.replace("└───", "└──")
    return line

def render_indent(depth: int, has_sibling_at_levels: list[bool]) -> str:
    parts = []
    for level in range(depth):
        if has_sibling_at_levels[level]:
            parts.append("│   ")
        else:
            parts.append("    ")
    return "".join(parts)

def normalize_name(name: str) -> str:
    # only remove trailing OCR garbage, never internal chars
    return name.rstrip(")]}›››")

def diff_tree(actual: str, expected: str):
    import difflib
    for line in difflib.unified_diff(
        expected.splitlines(),
        actual.splitlines(),
        lineterm=""
    ):
        print(line)

def render_canonical_tree(lines: List[TreeLine]) -> str:
    """
    Render a deterministic ASCII tree from structured TreeLine entries.

    This function:
    - ignores original spacing completely
    - re-renders indentation from depth
    - guarantees consistent ├── / └── / │ alignment
    """

    if not lines:
        return ""

    # Precompute: for each line, does another node exist later at same depth?
    has_next_at_depth = []
    n = len(lines)

    for i, line in enumerate(lines):
        depth = line.depth
        found = False
        for j in range(i + 1, n):
            if lines[j].depth == depth:
                found = True
                break
            if lines[j].depth < depth:
                break
        has_next_at_depth.append(found)

    # Also track continuation bars for parent levels
    active_columns = []

    output = []

    for i, line in enumerate(lines):
        depth = line.depth
        name = line.text.rstrip()

        # Root node (depth 0, first line)
        if depth == 0 and i == 0:
            output.append(name)
            active_columns = []
            continue

        # Trim active_columns to current depth
        active_columns = active_columns[:depth]

        # Determine if this node has a next sibling
        has_next = has_next_at_depth[i]

        # Build indent
        indent_parts = []
        for has_bar in active_columns:
            indent_parts.append("│   " if has_bar else "    ")

        indent = "".join(indent_parts)

        # Branch symbol
        branch = "├── " if has_next else "└── "

        output.append(indent + branch + name)

        # Update active_columns for children
        if len(active_columns) < depth:
            active_columns.append(has_next)
        else:
            if depth > 0:
                active_columns[-1] = has_next

    return "\n".join(output)

def normalize_tree(tree: str) -> str:
    for i, line in enumerate(lines):
        lines[i] = TreeLine(
            text=normalize_name(line.text),
            depth=line.depth,
            confidence=line.confidence,
        )

    final_tree = render_canonical_tree(lines)
    print(final_tree)

# -----------------------------
# CV fallback (no OCR engine)
# -----------------------------

def _crop_to_content(bw: np.ndarray) -> np.ndarray:
    np = _require_numpy()
    inv = 255 - bw
    ys, xs = np.where(inv > 0)
    if len(xs) == 0 or len(ys) == 0:
        return bw
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    pad = 5
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(bw.shape[1] - 1, x1 + pad)
    y1 = min(bw.shape[0] - 1, y1 + pad)
    return bw[y0:y1+1, x0:x1+1]


def segment_lines_projection(bw: np.ndarray, *, min_line_height: int = 6) -> List[Tuple[int, int]]:
    """
    Segment into line y-ranges using horizontal projection (no OCR).
    Returns list of (y_start, y_end) inclusive.
    """
    np = _require_numpy()
    inv = 255 - bw
    proj = (inv > 0).sum(axis=1)

    # threshold: line has ink
    ink = proj > max(5, int(0.002 * bw.shape[1]))

    ranges: List[Tuple[int, int]] = []
    in_run = False
    y0 = 0
    for y, v in enumerate(ink):
        if v and not in_run:
            in_run = True
            y0 = y
        elif not v and in_run:
            in_run = False
            y1 = y - 1
            if (y1 - y0 + 1) >= min_line_height:
                ranges.append((y0, y1))
    if in_run:
        y1 = len(ink) - 1
        if (y1 - y0 + 1) >= min_line_height:
            ranges.append((y0, y1))

    return ranges


def build_hershey_templates(
    charset: str,
    *,
    font_scale: float = 0.45,
    thickness: int = 1,
    pad: int = 6,
    box: int = 24,
) -> Dict[str, np.ndarray]:
    """
    Render templates for simple template matching classifier.
    """
    cv2 = _require_cv2()
    np = _require_numpy()
    font=cv2.FONT_HERSHEY_SIMPLEX,
    templates: Dict[str, np.ndarray] = {}
    for ch in charset:
        img = np.full((box, box), 255, dtype=np.uint8)
        ((w, h), _baseline) = cv2.getTextSize(ch, font, font_scale, thickness)
        x = (box - w) // 2
        y = (box + h) // 2
        cv2.putText(img, ch, (x, y), font, font_scale, 0, thickness, cv2.LINE_AA)
        templates[ch] = img
    return templates


def classify_char_template(
    glyph_bw: np.ndarray,
    templates: Dict[str, np.ndarray],
) -> Tuple[str, float]:
    """
    Classify a single character using normalized cross-correlation.
    Returns (best_char, score_0_1).
    """
    cv2 = _require_cv2()
    np = _require_numpy()

    g = cv2.resize(glyph_bw, (24, 24), interpolation=cv2.INTER_AREA)
    # ensure white background
    # glyph_bw is expected 0/255, we want same as templates (white bg with black ink)
    best_ch = "?"
    best = -1.0

    for ch, t in templates.items():
        # matchTemplate expects both uint8 or float
        res = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)
        score = float(res[0, 0])
        if score > best:
            best = score
            best_ch = ch

    # map [-1..1] => [0..1]
    return best_ch, float((best + 1.0) / 2.0)


def extract_chars_connected_components(line_bw: np.ndarray) -> List[np.ndarray]:
    """
    Extract character glyph images from a line using connected components.
    """
    np = _require_numpy()
    cv2 = _require_cv2()
    inv = 255 - line_bw
    num, labels, stats, _centroids = cv2.connectedComponentsWithStats(inv, connectivity=8)

    glyphs: List[Tuple[int, np.ndarray]] = []

    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < 10:
            continue
        # filter out huge blobs (rare; adjust if needed)
        if h > line_bw.shape[0] * 0.95 and w > line_bw.shape[1] * 0.5:
            continue

        roi = line_bw[y:y+h, x:x+w]
        glyphs.append((x, roi))

    glyphs.sort(key=lambda t: t[0])
    return [g for _, g in glyphs]


def read_tree_image_no_ocr(
    img_bgr: np.ndarray,
    *,
    charset: Optional[str] = None,
) -> List[TreeLine]:
    """
    Pure CV fallback (no OCR engine).
    Produces best-effort text by segmenting lines and classifying glyphs via templates.
    """
    cv2 = _require_cv2()
    np = _require_numpy()

    if charset is None:
        # include common tree glyph substitutions + typical filenames
        charset = (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
            " ._-/[](){}"
            "|+-\\"
        )

    bw = preprocess_for_tree(img_bgr)
    bw = _crop_to_content(bw)

    line_ranges = segment_lines_projection(bw)

    templates = build_hershey_templates(charset)

    out: List[TreeLine] = []

    for (y0, y1) in line_ranges:
        line_img = bw[y0:y1+1, :]
        # a little padding can help connected components
        line_img = cv2.copyMakeBorder(line_img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255)

        glyphs = extract_chars_connected_components(line_img)
        chars: List[str] = []
        confs: List[float] = []

        for g in glyphs:
            ch, score = classify_char_template(g, templates)
            chars.append(ch)
            confs.append(score)

        raw = "".join(chars)
        norm = normalize_tree_glyphs(raw)
        depth = compute_tree_depth(norm)
        conf = float(round(sum(confs) / len(confs), 3)) if confs else 0.0

        out.append(TreeLine(text=norm, depth=depth, confidence=conf))

    return out


# -----------------------------
# Main public API
# -----------------------------

def read_tree(
    src: Source,
    *,
    page: Optional[int] = None,
    dpi: int = 200,
    mode: str = "ocr",
) -> List[TreeLine]:
    """
    Read an ASCII tree from:
      - image path
      - image bytes / screenshot bytes
      - PDF path/bytes (render pages)

    mode:
      - "ocr": use Tesseract OCR path (best accuracy)
      - "auto": try OCR, fallback to no-OCR CV
      - "no_ocr": always use CV template-matching fallback
    """
    _require_ocr()

    images = load_pages_as_images(src, page=page, dpi=dpi)
    if not images:
        return []

    # For now: if multiple pages, concatenate with blank line separators
    all_lines: List[TreeLine] = []

    for idx, img in enumerate(images):
        # if mode in ("no_ocr",):
        #     lines = read_tree_image_no_ocr(img)
        # else:
        try:
            lines = read_tree_image_ocr(img)
        except Exception:
            if mode == "auto":
                lines = read_tree_image_no_ocr(img)
            else:
                raise

        if idx > 0:
            all_lines.append(TreeLine(text="", depth=0, confidence=1.0))
        all_lines.extend(lines)

    return all_lines


def read_tree_image_ocr(img_bgr: np.ndarray) -> List[TreeLine]:
    bw = preprocess_for_tree(img_bgr)
    ocr = ocr_words_with_boxes(bw)
    grouped = group_words_into_lines(ocr)
    repaired = reconstruct_lines_with_drift_repair(grouped)

    out: List[TreeLine] = []
    for raw, conf in repaired:
        norm = normalize_tree_glyphs(raw)
        depth = compute_tree_depth(norm)
        name = line.lstrip(" │├└─").strip()
        out.append(TreeLine(text=norm, depth=depth, confidence=conf))
    return out


def tree_lines_to_text(lines: List[TreeLine]) -> str:
    return "\n".join(l.text for l in lines)
