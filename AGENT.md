# Agent Guide

This repository contains the `mcce-features` Python package for extracting electrostatic features from MCCE output folders. Keep changes conservative, scientific-output aware, and compatible with the existing CLI.

## Project Layout

- `src/mcce_features/cli.py`: Typer CLI entry point. Keep command names stable.
- `src/mcce_features/core.py`: command orchestration, folder iteration, TSV output,
  and collation.
- `src/mcce_features/features.py`: MCCE parsing and feature extraction. Keep feature
  extraction in `MCCEFeatureExtractor`.
- `scripts/merge_features.py`: standalone CSV merge utility.
- `4lzt/`: sample MCCE working directory used for manual checks.
- Root `*.csv` and `*.tsv` files are data/output artifacts. Do not rewrite them
  unless a task explicitly asks for regenerated output.
- `dist/`, `src/*.egg-info`, `.venv-release/`, and cache directories are generated
  artifacts and should not be edited by hand.

## Environment

The package requires Python `>=3.12`. Install locally with:

```bash
python -m pip install -e ".[test]"
```

If optional test dependencies are not needed:

```bash
python -m pip install -e .
```

## Commands

The installed console script is:

```bash
mcce-features
```

Keep the existing command structure unless the user explicitly asks for a CLI
breaking change:

```bash
mcce-features version
mcce-features extract <mcce_folder>
mcce-features extract-folders <folder_file> --output-file <output.tsv>
mcce-features extract-subfolders-with-book <simulations_folder> --collated-features-tsv <output.tsv> --subfolders-startwith <"">
```

Useful local checks:

```bash
mcce-features --log-level DEBUG extract 4lzt
mcce-features extract-folders folder.lst --output-file mcce_elefeatures.tsv
python scripts/merge_features.py 99pdbs.csv mcce_elefeatures.tsv -o merged_features.csv
python -m pytest
```

There is not currently a formatter, linter, or test command configured in
`pyproject.toml`. Prefer adding standard tooling deliberately instead of assuming
one is already present.

## Logging

The CLI configures the root logger in `cli.setup_logging()` with this format:

```text
[LEVEL]: message
```

Preserve that log style for now. Use `logging.getLogger(__name__)` inside modules
and avoid configuring logging outside the CLI boundary. Follow the current level
semantics:

- `debug`: parsing details, fallback decisions, skipped malformed records.
- `info`: command progress, counts, output file locations.
- `warning`: recoverable missing or incomplete scientific data.
- `critical`: required MCCE source files are missing and extraction cannot proceed.
- `exception`: per-folder failures during batch extraction when processing should
  continue.

Avoid replacing logs with `print()` except for intentional command output such as
`version` and single-folder feature display.

## Feature Extraction Boundary

Keep feature extraction class-based. `MCCEFeatureExtractor` owns:

- required source file validation through `SOURCE_FILES`;
- parsing `step1_out.pdb`, `acc.res`, `sum_crg.out`, and `pK.out`;
- residue and atom state;
- registered feature names and output ordering;
- feature group methods such as composition, pKa perturbation, surface charge,
  patch localization, spatial moment, and charge separation.

When adding or changing features:

1. Add the feature name to `MCCEFeatureExtractor.feature_names` in the exact TSV output order.
2. Return the value from one feature group method as a `dict[str, float]`.
3. Let `extract_all_features()` enforce missing or extra registered features.
4. Update `README.md` feature documentation if the public feature set changes.
5. Add or update focused tests when test coverage is introduced.

Keep scientific constants near the extractor logic for now:

- `ACIDS`
- `BASES`
- `RESIDUE_ALIASES`
- `PK0_VALUES`
- `ANCHOR_ATOM_NAMES`
- `SOURCE_FILES`

Future refactors may move these into typed config objects, but do not scatter
them across command or script modules.

## Coding Style

Match the current code while moving toward standard Python:

- Use `pathlib.Path` for filesystem paths in new code.
- Prefer explicit return values over implicit mutation when practical, but do not
  break the extractor's current stateful workflow without tests.
- Keep parsing helpers small and close to the parser that uses them until a shared
  abstraction is clearly needed.
- Preserve existing residue IDs, feature names, TSV column names, and command
  names.
- Avoid broad rewrites of numerical logic without sample-data verification.
- Keep public behavior stable before optimizing internals.

The current code uses simple classes for `Atom` and `Residue`. A future technical
standardization pass may convert these to dataclasses, but only after verifying
that parsing, default values, and output are unchanged.

## Refactor Priorities

For the next cleanup phases, prefer this order:

1. Add a small test suite around `4lzt/` or synthetic fixtures for parser and
   feature output behavior.
2. Normalize path handling in `core.py` and `features.py`.
3. Split large extractor methods only where the new helpers have clear ownership.
4. Introduce dataclasses or typed records for `Atom` and `Residue`.
5. Add configured formatter/linter/test commands to `pyproject.toml`.
6. Review generated data artifacts and decide what belongs in version control.

Do not optimize performance before there are tests for representative MCCE output.

## Data And Output Safety

Feature output is scientific data. Treat changes to numeric rounding, missing-data
fallbacks, pH selection, residue aliasing, patch thresholds, and surface thresholds
as behavior changes. Verify them with sample input and call them out in summaries.

Current extraction rounds values in `extract_all_features()` before returning:

- values with absolute value below `1.0` use three significant figures;
- other values use three decimal places.

Do not change this output formatting casually because downstream CSV/TSV files may
depend on it.

## Git Hygiene

The worktree may contain user-generated data or local artifacts. Before editing,
check status with:

```bash
git status --short
```

Do not edit any files when the current branch is `main`. If work is needed while
on `main`, stop and ask the user to switch to or create a working branch first.

Never revert unrelated user changes. Keep edits scoped to source, docs, tests, or
configuration files needed for the requested task.
