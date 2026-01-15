

import importlib
import logging
from typing import Iterable, List, Type

from seed_cli.plugins.base import SeedPlugin

log = logging.getLogger(__name__)

ENTRYPOINT_GROUP = "seed.plugins"


def load_plugins(
    *,
    modules: Iterable[str] | None = None,
    strict: bool = False,
) -> List[SeedPlugin]:
    """
    Load Seed plugins.

    Parameters:
    - modules: iterable of module paths to load plugins from
    - strict: if True, raise on plugin load failure

    Returns:
    - list of plugin instances
    """
    plugins: List[SeedPlugin] = []

    # Explicit module loading (dev / tests)
    for mod in modules or []:
        try:
            plugins.extend(_load_from_module(mod))
        except Exception:
            log.exception("Failed to load plugin module: %s", mod)
            if strict:
                raise

    # Entry-point loading (future-proof)
    try:
        plugins.extend(_load_from_entrypoints())
    except Exception:
        log.exception("Failed to load entrypoint plugins")
        if strict:
            raise

    return plugins


def _load_from_module(module_path: str) -> List[SeedPlugin]:
    module = importlib.import_module(module_path)
    return _collect_plugins(module)


def _load_from_entrypoints() -> List[SeedPlugin]:
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover (older Python)
        return []

    eps = entry_points()
    group = eps.select(group=ENTRYPOINT_GROUP)
    plugins: List[SeedPlugin] = []

    for ep in group:
        try:
            cls = ep.load()
            if issubclass(cls, SeedPlugin):
                plugins.append(cls())
        except Exception:
            log.exception("Failed to load plugin entrypoint: %s", ep.name)

    return plugins


def _collect_plugins(module) -> List[SeedPlugin]:
    plugins: List[SeedPlugin] = []
    for obj in vars(module).values():
        if isinstance(obj, type) and issubclass(obj, SeedPlugin) and obj is not SeedPlugin:
            plugins.append(obj())
    return plugins
