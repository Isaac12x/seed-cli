# gbrain PR draft — seed-spec provider for `schema detect`

**Status:** draft (target: garrytan/gbrain, ≥ v0.43)
**Audience:** brains created or maintained by seed-cli where the user runs
`gbrain schema detect` instead of `seed export gbrain`.
**Acceptance criterion:** PRD AC11.

## Why

`gbrain schema detect` discovers page types heuristically from `source_path`
clustering — it has to *guess* the brain's shape. When the brain was created
by seed-cli, the ground truth is sitting at `<source>/brain.seed` (or
`<source>/.seed/specs/v*.tree`). This PR teaches `detect` to read that file
first, so users who never ran `seed export gbrain` still get spec-accurate
candidates with high confidence.

The change is **purely additive**: when no `.seed` / `.tree` / `.seed/specs/`
artefact exists, the existing heuristic path is unchanged.

## Surface

```
gbrain schema detect [--source <id>]
```

No new flags. Detection follows the existing 3-stage pipeline:

1. **detect** — emits candidate page_types
2. **suggest** — heuristic refinement
3. **review-candidates** — human gate; `--apply` promotes survivors

When the seed provider fires, it injects high-confidence candidates ahead of
the heuristic ones. Everything else (`suggest`, `review-candidates --apply`)
is unchanged; spec-derived candidates simply sail through.

## Provider contract

```ts
interface DetectProvider {
  name: string;
  detect(opts: { sourcePath: string; sourceId?: string }): Promise<DetectCandidate[] | null>;
}

interface DetectCandidate {
  type: string;
  primitive: 'entity' | 'concept' | 'media' | 'temporal' | 'annotation';
  pathPrefixes: string[];
  aliases: string[];
  extractable: boolean;
  expertRouting: boolean;
  confidence: number;       // 0..1; 1.0 from seed-spec, ~0.4..0.7 from heuristics
  provenance: {             // shown by review-candidates
    provider: string;       // 'seed-spec' here
    sourceFile: string;     // e.g. '<source>/brain.seed'
    kind?: string;          // original `!kind` token from the spec
  };
}
```

`detect()` returning `null` means "I have no opinion on this source"; the
pipeline falls through to the next provider.

## Provider: `seed-spec`

Looks at the source-root in order:

1. `brain.seed` (or any `*.seed` at root) — preferred (spec is lossless).
2. `*.tree` at root — supported, lossy (`!kind` may not be present).
3. `.seed/specs/current.tree` or the highest-numbered `v*.tree` in
   `.seed/specs/` — used as a fallback when only history exists.

Parses with the existing seed-spec grammar (see seed-cli `parsers.py`):

| seed construct                | candidate field           |
|-------------------------------|---------------------------|
| typed dir `foo/<x>/ !kind`    | `type` from kindmap; `pathPrefixes = ['foo/']` |
| typed dir `notes/ !note`      | `type = 'note'`; `pathPrefixes = ['notes/']`   |
| untyped placeholder `foo/<x>/`| derived from singularised `foo` |
| `... ` (extras)               | no candidate; falls to catch-all |
| `?` (optional)                | ignored for typing        |

Confidence is `1.0` for any candidate that originated in a `!kind`-marked
node; `0.8` for the derived-from-placeholder case. The kindmap used by
seed-cli (bundled under `seed_cli/resources/gbrain/kindmap.yml`) is mirrored
here as a small JSON file under `src/core/schema-pack/providers/seed-kindmap.json`
so gbrain doesn't take a Python runtime dep on seed-cli.

## Implementation outline

```
src/core/schema-pack/providers/
  seed-spec.ts        # the new provider
  seed-spec.test.ts
  seed-kindmap.json   # mirrors seed-cli's default kindmap
```

```ts
// src/core/schema-pack/providers/seed-spec.ts
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { parseSeedSpec } from './seed-parser';     // ~150 LOC, line-by-line port of seed-cli/parsers.py
import { resolveKind } from './seed-kindmap';      // tiny lookup

export const seedSpecProvider: DetectProvider = {
  name: 'seed-spec',
  async detect({ sourcePath }) {
    const specPath = await findSpec(sourcePath);
    if (!specPath) return null;
    const nodes = parseSeedSpec(await fs.readFile(specPath, 'utf8'));
    const candidates = compileCandidates(nodes, { specPath });
    return candidates;
  },
};

async function findSpec(root: string): Promise<string | null> {
  for (const name of ['brain.seed']) {
    const full = path.join(root, name);
    if (await exists(full)) return full;
  }
  // fall back to any *.seed at root
  for (const entry of await fs.readdir(root)) {
    if (entry.endsWith('.seed')) return path.join(root, entry);
  }
  // fall back to .seed/specs/current.tree
  const current = path.join(root, '.seed/specs/current.tree');
  if (await exists(current)) return current;
  // fall back to highest-numbered v*.tree under .seed/specs/
  return await findLatestVersionedSpec(root);
}
```

`compileCandidates` mirrors `seed_cli/gbrain/compiler.py` so seed-cli and
gbrain agree byte-for-byte on what a spec compiles to.

## Registration

Insert before the heuristic clustering pass:

```ts
// src/core/schema-pack/detect.ts
const providers: DetectProvider[] = [
  seedSpecProvider,                // NEW
  heuristicClusterProvider,        // existing
];
```

The pipeline already iterates providers in order; first non-null wins (with
heuristic provider always emitting at least `[]`, which is treated as "no
candidates" rather than "no opinion").

## Tests

- `tests/regressions/seed-spec-provider.test.ts`
  - sample brain with `brain.seed`: `detect` returns spec-derived candidates
    at confidence 1.0
  - sample brain without any seed artefacts: provider returns `null`,
    heuristic candidates appear unchanged
  - `.tree`-only brain: provider returns lossy candidates (no `!kind`) at
    confidence 0.8
  - `.seed/specs/v3.tree` only: provider falls back to versioned history

- Integration: `gbrain schema detect` against a seed-cli fixture brain
  yields candidates whose JSON matches the manifest seed-cli would produce
  via `seed export gbrain` (same fixture).

## Out of scope (intentional)

- Writing `gbrain.yml` (tier 5 activation): seed-cli owns this via
  `seed export gbrain --activate repo`.
- Running migrations (`unify-types`): seed-cli drives this from the spec
  history.
- `seed amend` reverse-drift: stays in seed-cli.

## Compatibility / risk

- **Risk:** seed-spec parser drift between seed-cli (Python) and gbrain (TS).
  *Mitigation:* land a shared kindmap JSON + a fixture-pinned conformance
  test that runs the same fixture brain through both implementations.
- **Risk:** spec artefact present but stale (user never ran apply).
  *Mitigation:* `review-candidates` is still the human gate; spec-derived
  candidates are *suggested*, not forced.
- **Risk:** confidence inflation makes `--apply` reckless.
  *Mitigation:* keep `review-candidates` interactive by default; only allow
  `--apply --yes` to auto-promote confidence ≥ 0.95 (already in v0.41).

## Open questions

1. Should the provider read `.seed/gbrain/kindmap.yml` overrides too?
   (Currently: yes — the spec's intent includes its kindmap.)
2. Should successor packs detected in `.seed/specs/` trigger
   `pack_upgrade_available`? (Currently: no — that's seed-cli's job via
   `seed export gbrain --migrate auto`.)
