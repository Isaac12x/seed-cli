# Tree-to-Seed Conversion Design

## Goal

Add `seed utils convert INPUT.tree [OUTPUT.seed]` to convert an existing tree
spec into a deterministic, compact, declarative `.seed` spec. When `OUTPUT` is
omitted, the output is written beside the input using the same stem and a
`.seed` suffix.

Collapsed brace expressions are first-class `.seed` syntax. Every expression
emitted by the converter must be accepted by the normal Seed parser, so the
result works with `plan`, `apply`, `diff`, templates, and other spec consumers.

## Command Behavior

- `INPUT` is a filesystem path to an existing `.tree` file.
- `OUTPUT` is optional and must use the `.seed` suffix.
- The default output for `path/to/project.tree` is
  `path/to/project.seed`.
- The command creates a missing output parent directory.
- Existing output is replaced deterministically.
- Missing inputs, directories, or unsupported suffixes produce a concise
  command error and a non-zero exit status.
- `seed utils convert --help` documents both positional arguments and the
  default output behavior.

## Compact Seed Syntax

Brace groups expand comma-separated alternatives:

```text
memories/{global,facts,episodes}.jsonl
services/<service-id>/{service.json,knowledge/,prompts/,tools.json}
```

The parser expands these examples to ordinary node paths before applying the
existing tree hierarchy rules. Whitespace around alternatives is ignored.
Multiple brace groups in one path are expanded as a Cartesian product.
Unmatched braces and brace pairs without a comma remain literal for backward
compatibility with valid filenames.

The converter may:

- join a directory chain that has only one child;
- group two or more compatible leaf siblings in braces;
- factor a shared filename extension outside a group;
- factor a shared trailing slash outside a group for directory-only siblings;
- mix files and directories by retaining `/` inside directory alternatives.

The converter must not combine nodes whose comments, annotations, optional
markers, or metadata differ. Compatible markers are rendered once after the
collapsed expression and are inherited by every expanded node.

The `|` character has no new special meaning. It remains part of a path name;
the requested `vector.sqlite|chroma/` example can therefore be represented
without inventing conditional filesystem semantics.

## Architecture

`seed_cli.conversion` will contain:

- brace-expression expansion used by the parser;
- a small path trie built from parsed `Node` objects;
- deterministic compact rendering;
- the file conversion entry point and path validation.

The existing Click command group will expose `utils convert` and dispatch it
through the current command runner. The legacy argparse builder will receive
the same subcommand so internal parser coverage and compatibility remain
intact.

## Data Flow

1. Resolve and validate the input and output paths.
2. Parse the `.tree` input with `parse_spec`, including current include,
   annotation, template, and metadata behavior.
3. Deduplicate nodes by their complete semantic identity.
4. Build a trie from node paths.
5. Render stable compact path lines, sorting alternatives by path name.
6. Write UTF-8 output with a final newline.
7. Parse the generated `.seed` in tests and compare its semantic nodes with
   the source nodes.

## Error Handling

Library functions raise `ValueError` for invalid extensions and
`FileNotFoundError` for missing paths. The CLI converts these to Click-style
errors without tracebacks. Parse failures leave an existing output untouched
because rendering completes before the file is written.

## Testing

Tests will first fail for:

- default and explicit output paths, including nested source paths;
- command help;
- missing and invalid inputs;
- brace expansion with shared suffixes, directory groups, mixed entries, and
  multiple groups;
- compact rendering of the requested hierarchy patterns;
- round-trip semantic equality, including comments and inline metadata.

The focused tests and full repository suite must pass before release notes,
commit, and push.
