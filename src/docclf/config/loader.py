# loader.py — read a YAML run configuration, validate it, and build a RunConfig.
# Author: Colile Sibanda

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from docclf.config.schema import (
    CLASS_NAMES,
    SECTION_TYPES,
    RunConfig,
    dataclass_field_names,
)

_TUPLE_FIELDS = {"ratios", "mean", "std", "crop_scale", "classes"}
_VALID_ENVIRONMENTS = ("local", "colab", "hpc")


def load_config(path: str | Path, env: str | None = None) -> RunConfig:
    """Purpose: turn a configuration file into a validated RunConfig.

    Inputs: `path` to a YAML file in `configs/`; `env` naming an optional
    `env_<env>.yaml` overlay in the same directory.
    Output: a frozen RunConfig. Raises ValueError naming the offending field
    when a key is unknown or a value is out of range.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            f"Expected a YAML file such as configs/smoke.yaml."
        )
    merged = _read_with_inheritance(config_path, seen=[])
    if env is not None:
        merged = _apply_environment_overlay(merged, config_path.parent, env)
    merged.setdefault("name", config_path.stem)
    return _build_run_config(merged, config_path)


def _read_with_inheritance(path: Path, seen: list[Path]) -> dict[str, Any]:
    """Purpose: read one YAML file and merge it over the files it extends.

    Inputs: the file path and the chain already visited (cycle detection).
    Output: a plain nested dictionary. Raises ValueError on an inheritance cycle.
    """
    resolved = path.resolve()
    if resolved in seen:
        chain = " -> ".join(item.name for item in [*seen, resolved])
        raise ValueError(f"Configuration inheritance cycle: {chain}.")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"Configuration file {path} must contain a YAML mapping.")
    document = dict(raw)
    parents = document.pop("extends", None)
    if parents is None:
        return document
    if isinstance(parents, str):
        parents = [parents]
    base: dict[str, Any] = {}
    for parent in parents:
        parent_path = (resolved.parent / parent).resolve()
        if not parent_path.is_file():
            raise FileNotFoundError(
                f"Configuration {path.name} extends {parent}, which does not exist "
                f"at {parent_path}."
            )
        base = _deep_merge(base, _read_with_inheritance(parent_path, [*seen, resolved]))
    return _deep_merge(base, document)


def _apply_environment_overlay(
    document: dict[str, Any], directory: Path, env: str
) -> dict[str, Any]:
    """Purpose: merge the machine-specific overlay over the run configuration.

    Inputs: the merged document, the configs directory, the environment name.
    Output: the document with `env_<env>.yaml` applied. Raises on an unknown name.
    """
    if env not in _VALID_ENVIRONMENTS:
        raise ValueError(
            f"Unknown environment: {env!r}. Valid values are "
            f"{', '.join(_VALID_ENVIRONMENTS)}."
        )
    overlay_path = directory / f"env_{env}.yaml"
    if not overlay_path.is_file():
        raise FileNotFoundError(
            f"Environment overlay not found: {overlay_path}. "
            f"Create it or omit the --env argument."
        )
    overlay = _read_with_inheritance(overlay_path, seen=[])
    return _deep_merge(document, overlay)


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Purpose: merge two configuration mappings, the overlay winning per key.

    Inputs: two mappings. Output: a new dictionary; nested mappings merge by key.
    """
    result = dict(base)
    for key, value in overlay.items():
        current = result.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(current, value)
        else:
            result[key] = value
    return result


def _build_run_config(document: Mapping[str, Any], source: Path) -> RunConfig:
    """Purpose: validate the merged document and instantiate the config objects.

    Inputs: the merged mapping and the file it came from (used in messages).
    Output: a RunConfig. Raises ValueError naming any unknown or invalid field.
    """
    allowed = set(dataclass_field_names(RunConfig))
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown configuration key(s) in {source.name}: {', '.join(unknown)}. "
            f"Valid top-level keys are: {', '.join(sorted(allowed))}."
        )
    kwargs: dict[str, Any] = {}
    for key, value in document.items():
        if key in SECTION_TYPES:
            kwargs[key] = _build_section(SECTION_TYPES[key], key, value, source)
        else:
            kwargs[key] = _coerce(key, value)
    config = RunConfig(**kwargs)
    _validate(config, source)
    return config


def _build_section(section_type: type, name: str, value: Any, source: Path) -> Any:
    """Purpose: build one nested configuration section from its mapping.

    Inputs: the section dataclass, its key, the raw value, the source file.
    Output: an instance of the section type. Raises ValueError on unknown keys.
    """
    if not isinstance(value, Mapping):
        raise ValueError(
            f"Configuration section {name!r} in {source.name} must be a mapping, "
            f"got {type(value).__name__}."
        )
    allowed = set(dataclass_field_names(section_type))
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown configuration key(s) in {source.name} under {name!r}: "
            f"{', '.join(unknown)}. Valid keys are: {', '.join(sorted(allowed))}."
        )
    return section_type(**{key: _coerce(key, item) for key, item in value.items()})


def _coerce(key: str, value: Any) -> Any:
    """Purpose: convert YAML lists into the tuples the frozen dataclasses expect.

    Inputs: the field name and its raw value. Output: the value, tupled if needed.
    """
    if key in _TUPLE_FIELDS and isinstance(value, list):
        return tuple(value)
    return value


def _validate(config: RunConfig, source: Path) -> None:
    """Purpose: enforce the invariants the fixed design depends on.

    Inputs: the assembled RunConfig and its source file.
    Output: None. Raises ValueError naming the field that is wrong.
    """
    if set(config.classes) != set(CLASS_NAMES):
        raise ValueError(
            f"Field 'classes' in {source.name} must be exactly the eight fixed "
            f"classes: {', '.join(CLASS_NAMES)}. The class set is fixed by the brief."
        )
    if len(config.split.ratios) != 3:
        raise ValueError(
            f"Field 'split.ratios' in {source.name} must hold three values "
            f"(train, validation, test), got {len(config.split.ratios)}."
        )
    total = sum(config.split.ratios)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Field 'split.ratios' in {source.name} must sum to 1.0, got {total}."
        )
    if any(ratio <= 0 for ratio in config.split.ratios):
        raise ValueError(
            f"Field 'split.ratios' in {source.name} must be strictly positive, "
            f"got {config.split.ratios}."
        )
    if config.image.size < 32:
        raise ValueError(
            f"Field 'image.size' in {source.name} must be at least 32, "
            f"got {config.image.size}."
        )
    if config.train.epochs < 1:
        raise ValueError(
            f"Field 'train.epochs' in {source.name} must be at least 1, "
            f"got {config.train.epochs}."
        )
    if config.train.batch_size < 1:
        raise ValueError(
            f"Field 'train.batch_size' in {source.name} must be at least 1, "
            f"got {config.train.batch_size}."
        )
    if config.data.n_per_class < 1:
        raise ValueError(
            f"Field 'data.n_per_class' in {source.name} must be at least 1, "
            f"got {config.data.n_per_class}."
        )
