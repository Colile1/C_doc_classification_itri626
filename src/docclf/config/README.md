# src/docclf/config/

Turns a YAML file in `configs/` into a validated, typed configuration
object, resolves the file paths for whichever machine the code is running
on, and seeds every source of randomness. This module is the reason the
identical repository runs unchanged on the laptop, on Colab and on a
school GPU.

## Planned files

| File | Responsibility |
|------|----------------|
| `schema.py` | `RunConfig` dataclass — the closed set of settings a run may specify. Unknown keys are an error, not a warning |
| `loader.py` | `load_config(path: str, env: str \| None = None) -> RunConfig` — read YAML, follow `extends:`, apply the `env_<env>.yaml` overlay, validate against the schema, raise a message naming the offending field |
| `paths.py` | `resolve_paths(cfg: RunConfig, env: str) -> ProjectPaths` — map logical names (`raw`, `processed`, `splits`, `checkpoints`) onto real directories for `local`, `colab` or `hpc` |
| `seed.py` | `set_global_seed(n: int) -> None` — seed `random`, `numpy`, `torch`, CUDA, and set the deterministic cuDNN flags |

## Integration

Called first by every `run_*.py` orchestrator, before any data is
touched. Depends only on `utils`. Must NOT import `data`, `models` or
`training` — a config load has to work in an environment where torch
weights are unavailable.

Done when: a config with an unknown key is rejected by name, and two runs
with the same seed produce byte-identical split files.

Built at step 2. `load_config` gained the `env` argument so that one call
loads the run config and the machine overlay together; `resolve_paths`
then only has to turn the merged names into absolute paths. `seed.py`
imports torch lazily, so a config load still works where torch is absent.
Evidence: `tests/test_config.py`, 14 tests.
