"""seed_cli.gbrain

Compile .seed/.tree specs into gbrain-schema-pack-v1 manifests, install them,
activate them, and reconcile drift in both directions. See `docs/gbrain` for
the PRD this module implements.
"""

from .amend import amend, AmendResult, AmendChange
from .compiler import compile_pack, CompiledPack
from .kindmap import load_kindmap, DEFAULT_KINDMAP
from .manifest import dump_manifest, lint_manifest, manifest_hash
from .exporter import export_gbrain, GbrainExportResult
from . import gbrain_cli

__all__ = [
    "amend",
    "AmendResult",
    "AmendChange",
    "compile_pack",
    "CompiledPack",
    "load_kindmap",
    "DEFAULT_KINDMAP",
    "dump_manifest",
    "lint_manifest",
    "manifest_hash",
    "export_gbrain",
    "GbrainExportResult",
    "gbrain_cli",
]
