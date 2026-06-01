"""Console UI helpers for pretty, consistent output."""

import os
import sys
from typing import Iterable, Dict
from dataclasses import dataclass

try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    _RICH = True
except Exception:
    _RICH = False

_CLICK_COLORS = {
    "black",
    "red",
    "green",
    "yellow",
    "blue",
    "magenta",
    "cyan",
    "white",
    "bright_black",
    "bright_red",
    "bright_green",
    "bright_yellow",
    "bright_blue",
    "bright_magenta",
    "bright_cyan",
    "bright_white",
}

_STEP_STYLES = {
    "mkdir": "bold blue",
    "create": "bold green",
    "update": "bold yellow",
    "delete": "bold red",
    "skip": "dim",
}

_SUMMARY_STYLES = {
    "Created": "green",
    "Updated": "yellow",
    "Deleted": "red",
    "Skipped": "dim",
    "Backed Up": "cyan",
}


def should_color(stream=None) -> bool:
    """Return whether terminal-facing output should include ANSI color."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") or os.environ.get("CLICOLOR_FORCE"):
        return True
    if os.environ.get("CLICOLOR") == "0":
        return False
    if os.environ.get("TERM") == "dumb":
        return False

    stream = sys.stdout if stream is None else stream
    return bool(getattr(stream, "isatty", lambda: False)())


def style_text(text: str, style: str, *, color: bool = False) -> str:
    """Apply a small Rich-compatible style string as ANSI text."""
    if not color:
        return text

    tokens = style.split()
    fg = next((token for token in tokens if token in _CLICK_COLORS), None)
    try:
        import click

        return click.style(
            text,
            fg=fg,
            bold="bold" in tokens,
            dim="dim" in tokens,
        )
    except Exception:
        return text


def _rich_text(text: str, style: str, *, color: bool):
    if color and _RICH:
        return Text(text, style=style)
    return text


def _rich_console(buf, *, color: bool):
    return Console(
        file=buf,
        force_terminal=color,
        color_system="auto" if color else None,
        no_color=not color,
    )


def format_step(step, *, color: bool = False) -> str:
    """
    Format a single plan step for human output.
    """
    path = step.path
    op = f"{step.op.upper():8}"
    op = style_text(op, _STEP_STYLES.get(step.op, "bold"), color=color)
    reason = (
        style_text(f" ({step.reason})", "dim", color=color) if step.reason else ""
    )
    optional = (
        style_text(" ?", "bold magenta", color=color)
        if getattr(step, "optional", False)
        else ""
    )
    return f"{op:8} {path}{optional}{reason}"

@dataclass
class Summary:
    created: int = 0
    updated: int = 0
    deleted: int = 0
    skipped: int = 0
    backed_up: int = 0


def render_summary(summary: Summary, *, color: bool = False) -> str:
    from io import StringIO

    rows = [
        ("Created", summary.created),
        ("Updated", summary.updated),
        ("Deleted", summary.deleted),
        ("Skipped", summary.skipped),
    ]
    if summary.backed_up > 0:
        rows.append(("Backed Up", summary.backed_up))

    if _RICH:
        table = Table(
            title=_rich_text("Seed Summary", "bold cyan", color=color),
            border_style="cyan" if color else None,
        )
        table.add_column("Action")
        table.add_column("Count", justify="right")
        for action, count in rows:
            style = _SUMMARY_STYLES[action]
            table.add_row(
                _rich_text(action, style, color=color),
                _rich_text(str(count), f"bold {style}", color=color),
            )
        buf = StringIO()
        console = _rich_console(buf, color=color)
        console.print(table)
        return buf.getvalue()
    else:
        return "\n".join(
            f"{style_text(action, _SUMMARY_STYLES[action], color=color)}: "
            f"{style_text(str(count), 'bold', color=color)}"
            for action, count in rows
        )


def render_list(title: str, items: Iterable[str], *, color: bool = False) -> str:
    from io import StringIO

    items = list(items)
    if not items:
        return (
            f"{style_text(title, 'bold cyan', color=color)}: "
            f"{style_text('none', 'dim', color=color)}"
        )

    if _RICH:
        table = Table(
            title=_rich_text(title, "bold cyan", color=color),
            border_style="cyan" if color else None,
        )
        table.add_column("Item")
        for i in items:
            table.add_row(_rich_text(i, "cyan", color=color))
        buf = StringIO()
        console = _rich_console(buf, color=color)
        console.print(table)
        return buf.getvalue()
    else:
        body = "\n".join(
            f"{style_text('-', 'cyan', color=color)} "
            f"{style_text(i, 'cyan', color=color)}"
            for i in items
        )
        return f"{style_text(title, 'bold cyan', color=color)}:\n{body}"


def render_kv(title: str, kv: Dict[str, str], *, color: bool = False) -> str:
    from io import StringIO

    if not kv:
        return (
            f"{style_text(title, 'bold cyan', color=color)}: "
            f"{style_text('none', 'dim', color=color)}"
        )

    if _RICH:
        table = Table(
            title=_rich_text(title, "bold cyan", color=color),
            border_style="cyan" if color else None,
        )
        table.add_column("Key")
        table.add_column("Value")
        for k, v in kv.items():
            table.add_row(
                _rich_text(str(k), "bold", color=color),
                _rich_text(str(v), "cyan", color=color),
            )
        buf = StringIO()
        console = _rich_console(buf, color=color)
        console.print(table)
        return buf.getvalue()
    else:
        body = "\n".join(
            f"{style_text(str(k), 'bold', color=color)}: "
            f"{style_text(str(v), 'cyan', color=color)}"
            for k, v in kv.items()
        )
        return f"{style_text(title, 'bold cyan', color=color)}:\n{body}"
