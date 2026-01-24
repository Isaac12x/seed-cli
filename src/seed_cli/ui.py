

"""seed_cli.ui

Console UI helpers for pretty, consistent output.
No side effects beyond formatting strings.
"""

from typing import Iterable, Dict, List
from dataclasses import dataclass

try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    _RICH = True
except Exception:
    _RICH = False


def format_step(step) -> str:
    """
    Format a single plan step for human output.
    """
    path = step.path
    op = step.op.upper()
    reason = f" ({step.reason})" if step.reason else ""
    optional = " ?" if getattr(step, "optional", False) else ""
    return f"{op:8} {path}{optional}{reason}"


def render_list(title: str, items: List[str]) -> str:
    if not items:
        return f"{title}: none"
    body = "\n".join(f"  - {i}" for i in items)
    return f"{title}:\n{body}"



@dataclass
class Summary:
    created: int = 0
    updated: int = 0
    deleted: int = 0
    skipped: int = 0
    backed_up: int = 0


def render_summary(summary: Summary) -> str:
    from io import StringIO
    if _RICH:
        table = Table(title="Seed Summary")
        table.add_column("Action")
        table.add_column("Count", justify="right")
        table.add_row("Created", str(summary.created))
        table.add_row("Updated", str(summary.updated))
        table.add_row("Deleted", str(summary.deleted))
        table.add_row("Skipped", str(summary.skipped))
        if summary.backed_up > 0:
            table.add_row("Backed Up", str(summary.backed_up))
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        console.print(table)
        return buf.getvalue()
    else:
        base = (
            f"Created: {summary.created}\n"
            f"Updated: {summary.updated}\n"
            f"Deleted: {summary.deleted}\n"
            f"Skipped: {summary.skipped}"
        )
        if summary.backed_up > 0:
            base += f"\nBacked Up: {summary.backed_up}"
        return base


def render_list(title: str, items: Iterable[str]) -> str:
    from io import StringIO
    items = list(items)
    if not items:
        return f"{title}: none"

    if _RICH:
        table = Table(title=title)
        table.add_column("Item")
        for i in items:
            table.add_row(i)
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        console.print(table)
        return buf.getvalue()
    else:
        body = "\n".join(f"- {i}" for i in items)
        return f"{title}:\n{body}"


def render_kv(title: str, kv: Dict[str, str]) -> str:
    from io import StringIO
    if not kv:
        return f"{title}: none"

    if _RICH:
        table = Table(title=title)
        table.add_column("Key")
        table.add_column("Value")
        for k, v in kv.items():
            table.add_row(str(k), str(v))
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        console.print(table)
        return buf.getvalue()
    else:
        body = "\n".join(f"{k}: {v}" for k, v in kv.items())
        return f"{title}:\n{body}"
