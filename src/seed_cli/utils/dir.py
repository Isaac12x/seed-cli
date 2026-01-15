from typing import Dict, Tuple, Optional, List, Set
from pathlib import Path

def _detect_single_root_dir(
    sidx: Dict[str, "Node"]
) -> Optional[str]:
    """
    If all spec paths are under a single first-segment directory
    and that directory itself is a dir node, return that root segment.
    Otherwise None.
    """
    keys = [k for k in sidx.keys() if k and k != "."]
    if not keys:
        return None

    first_parts = []
    for k in keys:
        parts = Path(k).parts
        if not parts:
            return None
        first_parts.append(parts[0])

    root = first_parts[0]
    if any(p != root for p in first_parts):
        return None

    # Ensure root itself exists as a dir node in spec
    root_node = sidx.get(root)
    if not root_node or not getattr(root_node, "is_dir", False):
        return None

    # Ensure nothing exists outside root (by first-part check) already true
    return root

def _normalize_spec_index(
    sidx: Dict[str, "Node"],
    base: Path,
) -> Dict[str, "Node"]:
    """
    Normalize spec paths:
    - expand ~
    - resolve absolute paths
    - make them relative to base if possible
    """
    out = {}
    for rel, node in sidx.items():
        p = Path(rel).expanduser()

        if p.is_absolute():
            try:
                p = p.resolve().relative_to(base)
            except ValueError:
                # Spec path is absolute but outside base → leave as-is
                p = p.resolve()
        out[str(p)] = node
    return out



def _strip_prefix_index(idx: dict, prefix: str) -> dict:
    out = {}
    pre = prefix.rstrip("/") + "/"

    for k, v in idx.items():
        if k == prefix:
            # drop the root dir itself
            continue
        if k.startswith(pre):
            out[k[len(pre):]] = v
        else:
            out[k] = v  # defensive: leave untouched
    return out


def _strip_prefix_checks(
    checks: Dict[str, dict],
    prefix: str
) -> Dict[str, dict]:
    out = {}
    pre = prefix + "/"
    for k, v in checks.items():
        if k == prefix:
            continue
        if k.startswith(pre):
            out[k[len(pre):]] = v
        else:
            out[k] = v
    return out
