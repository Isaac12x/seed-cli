

"""seed_cli.templating

Lightweight templating for specs.

Supports:
- {{var}} substitution
- strict vs loose mode
- default values via {{var|default}}

This intentionally avoids full Jinja for safety and predictability.
"""

import re
from typing import Dict

_VAR_RE = re.compile(r"\{\{\s*([a-zA-Z_][\w-]*)(?:\|([^}]+))?\s*\}\}")


class TemplateError(ValueError):
    pass


def apply_vars(text: str, vars: Dict[str, str], *, mode: str = "strict") -> str:
    """Apply {{vars}} substitution to text.

    Modes:
    - strict: missing vars raise TemplateError
    - loose: missing vars left unchanged
    """

    def repl(match: re.Match) -> str:
        name = match.group(1)
        default = match.group(2)
        if name in vars:
            return str(vars[name])
        if default is not None:
            return default
        if mode == "loose":
            return match.group(0)
        raise TemplateError(f"Missing template variable: {name}")

    return _VAR_RE.sub(repl, text)
