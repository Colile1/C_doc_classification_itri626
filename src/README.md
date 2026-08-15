# src/

Holds the installable package `docclf`. Nothing in this folder reads
command-line arguments or prints results: the root-level `run_*.py`
orchestrators own all I/O, and everything here stays importable from a
notebook, a script or a test without side effects.

| Path | Purpose |
|------|---------|
| `docclf/` | The project package (see its README for the module map) |

Done when: `python -c "import docclf"` succeeds from a clean install and
no module inside `docclf` calls `print()` or `argparse`.
