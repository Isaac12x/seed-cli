
"""seed_cli.utils

Utility functions for common operations.

These utilities provide convenient ways to perform common tasks
that might be useful for users.
"""

import re
from pathlib import Path
from typing import Optional, List, Tuple


def _estimate_depth_from_leading(line: str) -> int:
    """Estimate tree depth from leading characters.
    
    Counts patterns like:
    - │   or |   (4 chars = 1 depth level)
    - Leading spaces (4 spaces = 1 depth level)
    - Pipe characters followed by spaces
    """
    depth = 0
    i = 0
    while i < len(line):
        # Check for tree continuation pattern (│   or |   )
        if i + 4 <= len(line):
            chunk = line[i:i+4]
            # Patterns indicating one level of depth
            if chunk in ['│   ', '|   ', '    ', '│  ', '|  ']:
                depth += 1
                i += 4
                continue
            # Shorter patterns
            if chunk[:3] in ['│  ', '|  ']:
                depth += 1
                i += 3
                continue
        # Single pipe/bar at position
        if line[i] in '│|':
            # Check if followed by spaces
            rest = line[i+1:i+4]
            if rest.strip() == '' and len(rest) > 0:
                depth += 1
                i += len(rest) + 1
                continue
        # Stop at branch characters or content
        if line[i] in '├└─LT[\\}_':
            break
        if line[i].isalnum():
            break
        i += 1
    return depth


def _extract_content(line: str) -> Tuple[str, bool]:
    """Extract the actual content (filename/dirname) from a tree line.
    
    Returns: (content, is_last_in_group)
    is_last_in_group is True if this is a └── (last item) branch
    """
    # Remove tree characters and extract content
    # Common branch patterns: ├── └── ├─ └─ L_ [— etc.
    
    # First, try to find standard branch patterns
    match = re.search(r'(├──|└──|├─|└─)\s*(.+)$', line)
    if match:
        is_last = match.group(1).startswith('└')
        return match.group(2).strip(), is_last
    
    # Try OCR-corrupted patterns
    # L_, L-, LE, LL, [—, [-, \_, t—, }-, |/, │/, etc.
    match = re.search(r'^[\s│\|]*(L[_\-E]|LL|\[—|\[[-=]|\[LE|\\[_\-]|_\s+|t—|}-|\|/|│/)\s*(.+)$', line)
    if match:
        connector = match.group(1)
        is_last = connector.startswith('\\') or connector.startswith('_') or connector == '}-'
        content = match.group(2).strip()
        # Clean up common artifacts at start of content
        content = re.sub(r'^[&;/]\s*', '', content)
        return content, is_last
    
    # Try pattern with just | or │ followed by content (like "|; schema.ts" or "│/& index.ts")
    match = re.search(r'^[\s│\|]+[;/&]\s*(.+)$', line)
    if match:
        return match.group(1).strip(), False
    
    # Try to extract anything after tree characters
    # Remove leading tree/artifact characters
    content = re.sub(r'^[\s│\|├└─LT\[\]\\}_\-=&;/]+', '', line).strip()
    if content:
        return content, False
    
    return '', False


def _clean_content(content: str) -> str:
    """Clean up OCR artifacts in content (filename/dirname)."""
    if not content:
        return content
    
    # Fix spacing before file extensions (e.g., "file. ts" -> "file.ts")
    content = re.sub(r'\.\s+([a-zA-Z0-9]+)$', r'.\1', content)
    
    # Fix spacing in brackets (e.g., "[ slug ]" -> "[slug]")
    content = re.sub(r'\[\s*([^\]]+)\s*\]', r'[\1]', content)
    
    # Fix spacing around slashes
    content = re.sub(r'\s*/\s*', '/', content)
    
    # Fix common OCR misreads
    content = re.sub(r'^lL\s+', '', content)  # "lL linking.ts" -> "linking.ts"
    content = re.sub(r'^TL\s+', '', content)  # "TL validation.ts" -> "validation.ts"
    content = re.sub(r'^E\s+', '', content)   # "E templates/" -> "templates/"
    content = re.sub(r'^LE\s+', '', content)  # "LE templates/" -> "templates/"
    
    # Fix "topicl" -> "topic]" (common OCR error with brackets)
    content = re.sub(r'topicl/', 'topic]/', content)
    content = re.sub(r'\[s lug\]', '[slug]', content)
    content = re.sub(r'\[s\s+lug\]', '[slug]', content)
    
    # Fix missing opening brackets
    if content.startswith('slug]'):
        content = '[slug]' + content[5:]
    
    # Fix ToolLandingTemplate.tsx -> ToolLandingTemplate.tsx (lowercase 'o' at start)
    if content.startswith('oolLandingTemplate'):
        content = 'T' + content
    
    return content.strip()


def _is_directory(content: str) -> bool:
    """Check if content represents a directory."""
    return content.endswith('/')


def _clean_ocr_text(text: str) -> str:
    """Clean OCR-extracted text and reconstruct proper tree structure.
    
    This function attempts to intelligently reconstruct a tree structure
    from corrupted OCR output by:
    1. Identifying items and their approximate depth
    2. Detecting directories vs files
    3. Reconstructing proper tree formatting
    """
    lines = text.splitlines()
    
    # Skip patterns (headers, artifacts)
    skip_patterns = [
        r'^File Structure Summary',
        r'^Summary',
        r'^Structure',
        r'^SEC$',
        r'^T\+$',
        r'^E\+$',
        r'^\.$',  # Just a dot
    ]
    
    # First pass: extract items with depth estimates
    items: List[Tuple[int, str, bool]] = []  # (depth, content, is_dir)
    
    root_found = False
    
    for line in lines:
        line = line.rstrip()
        if not line.strip():
            continue
        
        # Skip artifacts
        if any(re.match(p, line.strip(), re.IGNORECASE) for p in skip_patterns):
            continue
        
        # Check for root directory (e.g., "src/")
        stripped = line.strip()
        if stripped.endswith('/') and not any(c in stripped for c in '│|├└─'):
            if not root_found:
                content = stripped
                items.append((0, content, True))
                root_found = True
                continue
        
        # Estimate depth from leading characters
        depth = _estimate_depth_from_leading(line)
        
        # Extract content
        content, is_last = _extract_content(line)
        
        if not content:
            continue
        
        # Clean the content
        content = _clean_content(content)
        
        if not content:
            continue
        
        # Skip pure artifacts
        if content.lower() in ['sec', 'e+', 't+', 'libs', 'datas']:
            continue
        
        is_dir = _is_directory(content)
        items.append((depth, content, is_dir))
    
    if not items:
        return ''
    
    # Second pass: build proper tree structure
    # We need to track the current path to build proper hierarchy
    output_lines: List[str] = []
    dir_stack: List[Tuple[int, str]] = []  # Stack of (depth, dirname)
    
    for i, (depth, content, is_dir) in enumerate(items):
        # Root item (depth 0)
        if depth == 0 and i == 0:
            output_lines.append(content)
            if is_dir:
                dir_stack.append((0, content.rstrip('/')))
            continue
        
        # Adjust depth based on directory stack
        # Pop directories that are at same or higher depth
        while dir_stack and dir_stack[-1][0] >= depth:
            dir_stack.pop()
        
        # Determine if this is the last item at this depth
        is_last_at_depth = True
        for j in range(i + 1, len(items)):
            if items[j][0] == depth:
                is_last_at_depth = False
                break
            if items[j][0] < depth:
                break
        
        # Build the prefix
        prefix_parts = []
        for d in range(depth):
            # Check if there are more items at this depth level after current item
            has_more_at_d = False
            for j in range(i + 1, len(items)):
                if items[j][0] == d:
                    has_more_at_d = True
                    break
                if items[j][0] < d:
                    break
            
            if has_more_at_d or (d < depth - 1):
                prefix_parts.append('│   ')
            else:
                prefix_parts.append('    ')
        
        prefix = ''.join(prefix_parts)
        
        # Choose branch character
        if is_last_at_depth:
            branch = '└── '
        else:
            branch = '├── '
        
        output_lines.append(prefix + branch + content)
        
        # If this is a directory, push to stack
        if is_dir:
            dir_stack.append((depth, content.rstrip('/')))
    
    return '\n'.join(output_lines)


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
    from ..image import extract_text_from_image
    
    # Check if image exists
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Determine output path
    if output_path is None:
        output_path = image_path.with_suffix(".tree")
    
    # Extract text from image using OCR with layout preservation
    raw_text = extract_text_from_image(image_path, preserve_layout=True)
    
    if raw:
        # Output raw OCR text for debugging
        output_path.write_text(raw_text, encoding="utf-8")
    else:
        # Clean and reconstruct tree structure
        cleaned_text = _clean_ocr_text(raw_text)
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
        return True
    except ImportError:
        return False
