# src/docclf/utils/

Small shared helpers with no project-specific knowledge. Anything that
would otherwise be copied into two modules belongs here.

## The modules

| File | Responsibility |
|------|----------------|
| `env.py` | `detect_environment() -> str` returning `local`, `colab` or `hpc`; `select_device(cfg) -> torch.device` preferring CUDA and falling back to CPU with a logged note |
| `logging.py` | `get_logger(name)` — one structured logger, `Description: value` formatting, no bare prints anywhere in the package |
| `timing.py` | `timed(label)` — a context manager recording wall-clock duration into the run log |
| `hashing.py` | `perceptual_hash(path) -> str` — used by both `data.clean` and `splits.leakage`, which is exactly why it lives here |

## Integration

Imported by every other module; imports none of them. If a helper here
ever needs to import from `config` or above, it is not a utility and
belongs in the calling module.

The tests hold this module to one guarantee: no module in `docclf` outside `utils` calls `print()`, and `hashing` has a test proving two visually identical images collide and two different ones do not.