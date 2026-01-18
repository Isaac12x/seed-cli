

"""seed_cli.graphviz

Graphviz (.dot) export for execution plans and folder structures.

This produces a directed acyclic graph:
- Nodes are paths (directories/files)
- Edges represent dependencies (depends_on) or parent-child relationships
- Node labels include operation and annotations

Intended for:
  seed plan --dot > plan.dot
  seed capture --dot > structure.dot
  dot -Tpng plan.dot -o plan.png
"""

from typing import List, Dict, Set, Optional
from pathlib import Path
import re

from .planning import PlanResult, PlanStep
from .parsers import Node


def _node_id(path: str) -> str:
    # Graphviz-safe identifier
    return path.replace("/", "_").replace(".", "_") or "root"


def _path_from_node_id(node_id: str) -> str:
    # Reverse the node_id transformation
    return node_id.replace("_", "/").replace("//", "/")


def plan_to_dot(plan: PlanResult) -> str:
    """Convert a PlanResult to Graphviz DOT format."""
    lines: List[str] = []
    lines.append("digraph seed_plan {")
    lines.append("  rankdir=LR;")
    lines.append("  node [shape=box, fontname=Helvetica];")

    # Emit nodes
    for step in plan.steps:
        nid = _node_id(step.path)
        label = f"{step.op}\n{step.path}"
        if step.annotation:
            label += f"\n@{step.annotation}"
        if step.reason:
            label += f"\n({step.reason})"
        lines.append(f'  "{nid}" [label="{label}"];')

    # Emit edges
    for step in plan.steps:
        if not step.depends_on:
            continue
        nid = _node_id(step.path)
        for dep in step.depends_on:
            did = _node_id(dep)
            lines.append(f'  "{did}" -> "{nid}";')

    lines.append("}")
    return "\n".join(lines)


def nodes_to_dot(nodes: List[Node]) -> str:
    """Convert a list of Nodes (folder structure) to Graphviz DOT format.
    
    Creates a hierarchical graph where:
    - Nodes represent files and directories
    - Edges represent parent-child relationships
    - Directory nodes are distinguished from file nodes
    """
    lines: List[str] = []
    lines.append("digraph folder_structure {")
    lines.append("  rankdir=TB;")
    lines.append("  node [fontname=Helvetica];")
    
    # Build a set of all paths and parent-child relationships
    path_to_node: Dict[str, Node] = {}
    edges: List[tuple[str, str]] = []
    
    for node in nodes:
        path_str = node.relpath.as_posix()
        path_to_node[path_str] = node
        
        # Create parent-child edge
        if path_str != ".":
            parent = str(node.relpath.parent)
            if parent == ".":
                parent = "root"
            edges.append((parent, path_str))
    
    # Emit nodes with different shapes for dirs vs files
    for path_str, node in sorted(path_to_node.items()):
        nid = _node_id(path_str) if path_str != "." else "root"
        label = path_str if path_str != "." else "."
        
        # Add annotation if present
        if node.annotation:
            label += f"\n@{node.annotation}"
        if node.comment:
            label += f"\n({node.comment})"
        
        # Different shapes for directories vs files
        shape = "box" if node.is_dir else "ellipse"
        style = "filled" if node.is_dir else ""
        color = "lightblue" if node.is_dir else "lightgray"
        
        attrs = [f'label="{label}"', f'shape={shape}']
        if style:
            attrs.append(f'style={style}')
        if color:
            attrs.append(f'fillcolor={color}')
        
        lines.append(f'  "{nid}" [{", ".join(attrs)}];')
    
    # Emit edges
    for parent, child in edges:
        parent_id = _node_id(parent) if parent != "root" else "root"
        child_id = _node_id(child)
        lines.append(f'  "{parent_id}" -> "{child_id}";')
    
    lines.append("}")
    return "\n".join(lines)


def dot_to_nodes(dot_content: str) -> List[Node]:
    """Parse a Graphviz DOT file into a list of Node objects.
    
    Extracts nodes from DOT format and reconstructs the folder structure.
    Handles both plan-style DOT (with operations) and structure-style DOT.
    """
    nodes: List[Node] = []
    node_labels: Dict[str, str] = {}
    edges: List[tuple[str, str]] = []
    
    # Parse DOT file
    # Match node definitions: "node_id" [label="...", ...];
    node_pattern = re.compile(r'"([^"]+)"\s*\[([^\]]+)\];')
    # Match edge definitions: "from" -> "to";
    edge_pattern = re.compile(r'"([^"]+)"\s*->\s*"([^"]+)";')
    
    for line in dot_content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("digraph") or line.startswith("}"):
            continue
        
        # Skip graph attributes
        if "rankdir=" in line or "node [" in line:
            continue
        
        # Parse node
        node_match = node_pattern.search(line)
        if node_match:
            node_id = node_match.group(1)
            attrs = node_match.group(2)
            
            # Extract label from attributes
            label_match = re.search(r'label="([^"]+)"', attrs)
            if label_match:
                label = label_match.group(1)
                node_labels[node_id] = label
            else:
                node_labels[node_id] = node_id
        
        # Parse edge
        edge_match = edge_pattern.search(line)
        if edge_match:
            from_id = edge_match.group(1)
            to_id = edge_match.group(2)
            edges.append((from_id, to_id))
    
    # Build path structure from edges
    # Find root nodes (nodes with no incoming edges)
    all_targets = {to_id for _, to_id in edges}
    root_ids = set(node_labels.keys()) - all_targets
    
    # If no root found, use "root" or first node
    if not root_ids:
        if "root" in node_labels:
            root_ids = {"root"}
        elif node_labels:
            root_ids = {list(node_labels.keys())[0]}
    
    # Build path mapping: node_id -> path
    id_to_path: Dict[str, str] = {}
    path_to_id: Dict[str, str] = {}
    
    # Extract paths from labels
    for node_id, label in node_labels.items():
        # Try to extract path from label (could be multi-line)
        lines = label.split("\n")
        path_str = lines[0].strip()  # First line is usually the path
        
        # Clean up path (remove operation prefixes if present)
        path_str = re.sub(r'^(create|mkdir|update|delete|skip)\s+', '', path_str)
        path_str = path_str.strip()
        
        # If path looks like a path, use it; otherwise reconstruct from node_id
        if "/" in path_str or path_str == ".":
            id_to_path[node_id] = path_str
            path_to_id[path_str] = node_id
        else:
            # Reconstruct from node_id
            path_str = _path_from_node_id(node_id)
            id_to_path[node_id] = path_str
            path_to_id[path_str] = node_id
    
    # Build parent-child relationships from edges
    children: Dict[str, List[str]] = {}
    for from_id, to_id in edges:
        if from_id not in children:
            children[from_id] = []
        children[from_id].append(to_id)
    
    # Reconstruct paths using parent-child relationships
    def get_path(node_id: str, visited: Set[str] = None) -> str:
        if visited is None:
            visited = set()
        if node_id in visited:
            return id_to_path.get(node_id, _path_from_node_id(node_id))
        
        visited.add(node_id)
        
        if node_id in id_to_path:
            return id_to_path[node_id]
        
        # Find parent
        parent_id = None
        for p_id, kids in children.items():
            if node_id in kids:
                parent_id = p_id
                break
        
        if parent_id:
            parent_path = get_path(parent_id, visited)
            # Extract name from label or node_id
            label = node_labels.get(node_id, node_id)
            name = label.split("\n")[0].strip()
            # Remove operation prefix if present
            name = re.sub(r'^(create|mkdir|update|delete|skip)\s+', '', name)
            name = name.strip()
            
            if name and name != node_id:
                path = f"{parent_path}/{name}" if parent_path != "." else name
            else:
                # Fallback: use node_id
                name_part = _path_from_node_id(node_id).split("/")[-1]
                path = f"{parent_path}/{name_part}" if parent_path != "." else name_part
            
            id_to_path[node_id] = path
            return path
        else:
            # Root node
            path = id_to_path.get(node_id, ".")
            id_to_path[node_id] = path
            return path
    
    # Build all paths
    for node_id in node_labels.keys():
        get_path(node_id)
    
    # Create Node objects
    for node_id, path_str in id_to_path.items():
        label = node_labels.get(node_id, "")
        
        # Extract annotation and comment from label
        annotation = None
        comment = None
        
        if "@" in label:
            ann_match = re.search(r'@(\w+)', label)
            if ann_match:
                annotation = ann_match.group(1)
        
        if "(" in label:
            com_match = re.search(r'\(([^)]+)\)', label)
            if com_match:
                comment = com_match.group(1)
        
        # Determine if directory (check if it has children or ends with /)
        is_dir = False
        if node_id in children:
            is_dir = True
        elif path_str.endswith("/"):
            is_dir = True
            path_str = path_str.rstrip("/")
        elif any(e[0] == node_id for e in edges):
            # Has outgoing edges, likely a directory
            is_dir = True
        
        # Normalize path
        if path_str == ".":
            path = Path(".")
        else:
            path = Path(path_str)
        
        nodes.append(Node(relpath=path, is_dir=is_dir, comment=comment, annotation=annotation))
    
    return nodes
