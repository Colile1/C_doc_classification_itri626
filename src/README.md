# src/

This folder holds the installable package `docclf`, which contains every piece
of logic the project uses. Nothing here reads command-line arguments or prints
results: the root-level `run_*.py` orchestrators own all input and output, so
each module stays importable from a notebook, a script or a test without side
effects.

| Path | Purpose |
|------|---------|
| `docclf/` | The project package. Its own README maps the modules |

The separation matters for two reasons. A notebook can import a function and
show its result without triggering a training run, and a test can exercise the
same code path the orchestrators use rather than a parallel copy of it.

`python -c "import docclf"` succeeds from a clean install, and no module inside
the package calls `print()` or `argparse`.
