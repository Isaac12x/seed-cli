
from typing import Any, Iterable


class SeedPlugin:
    """
    Base class for Seed plugins.

    Plugins may modify the pipeline at well-defined phases.
    All methods are optional.
    """

    name = "unnamed"

    # ---------- Parsing ----------

    def before_parse(self, text: str, context: dict) -> str:
        return text

    def after_parse(self, nodes: Iterable[Any], context: dict) -> None:
        return None

    # ---------- Planning ----------

    def before_plan(self, nodes: Iterable[Any], context: dict) -> None:
        return None

    def after_plan(self, plan, context: dict) -> None:
        return None

    # ---------- Execution ----------

    def before_build(self, plan, context: dict) -> None:
        return None

    def after_build(self, context: dict) -> None:
        return None

    # ---------- Sync ----------

    def before_sync_delete(self, relpath: str, context: dict) -> bool:
        """
        Return False to veto deletion.
        """
        return True
