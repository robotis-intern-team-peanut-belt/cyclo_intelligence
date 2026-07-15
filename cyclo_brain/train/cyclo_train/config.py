# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experiment configuration: load, merge, resolve, validate, hash."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

WORKSPACE = Path(os.environ.get("CYCLO_WORKSPACE", "/workspace"))

TRAIN_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = Path(os.environ.get("CYCLO_EXPERIMENTS_DIR", TRAIN_ROOT / "experiments"))

# Must start alphanumeric, so "." and ".." can never be names: they are used
# unescaped as directory names under /workspace/runs and /workspace/model.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_VAR_RE = re.compile(r"\$\{([a-z_]+)\}")

TRACKING_BACKENDS = ("wandb", "none")
TRACKING_MODES = ("online", "offline", "disabled")
DEVICES = ("cuda", "cpu", "mps")


class ConfigError(Exception):
    """A malformed or unusable experiment config."""


def validate_name(name: str, *, what: str = "name") -> str:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ConfigError(
            f"invalid {what}: {name!r}\n"
            f"It becomes a directory name, so it must match {_NAME_RE.pattern}"
        )
    return name


@dataclass
class DatasetConfig:
    repo_id: str
    root: str


@dataclass
class TrackingConfig:
    backend: str = "none"
    project: str = "cyclo-lerobot"
    entity: str | None = None
    mode: str = "online"
    notes: str | None = None


@dataclass
class ExperimentConfig:
    name: str
    dataset: DatasetConfig
    # policy/train stay open dicts: they are forwarded verbatim as
    # --policy.<k>=<v> / --<k>=<v>, so any upstream lerobot flag is reachable
    # from YAML without a code change.
    policy: dict[str, Any]
    train: dict[str, Any] = field(default_factory=dict)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    backend: str = "lerobot"

    @property
    def run_dir(self) -> Path:
        return WORKSPACE / "runs" / self.name

    @property
    def output_dir(self) -> Path:
        # lerobot refuses to start when --output_dir already exists, and creates
        # it itself at the first checkpoint save. So it must own a subdirectory
        # of the run dir rather than the run dir itself.
        return self.run_dir / "output"

    @property
    def dropbox_dir(self) -> Path:
        return WORKSPACE / "model" / self.backend / self.name

    @property
    def log_file(self) -> Path:
        return self.run_dir / "train.log"

    @property
    def manifest_file(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def resolved_config_file(self) -> Path:
        return self.run_dir / "resolved_config.yaml"

    @property
    def last_checkpoint_config(self) -> Path:
        # lerobot requires this path via --config_path to resume.
        return self.output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def resolve_experiment_path(ref: str) -> Path:
    candidates: list[Path] = []
    as_path = Path(ref)
    if as_path.suffix in (".yaml", ".yml"):
        candidates += [as_path, EXPERIMENTS_DIR / ref]
    else:
        candidates += [EXPERIMENTS_DIR / f"{ref}.yaml", EXPERIMENTS_DIR / f"{ref}.yml"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    known = ", ".join(sorted(p.stem for p in EXPERIMENTS_DIR.glob("*.yaml"))) or "(none)"
    raise ConfigError(f"Experiment not found: {ref!r}\nAvailable: {known}")


def _load_layered(path: Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    _seen = set(_seen or set())
    if path in _seen:
        raise ConfigError(f"Circular _base chain at {path}")
    _seen.add(path)

    raw = _read_yaml(path)
    base_ref = raw.pop("_base", None)
    if not base_ref:
        return raw
    if not isinstance(base_ref, str):
        raise ConfigError(f"{path}: _base must be a path string, got {type(base_ref).__name__}")

    base_path = Path(base_ref)
    base_path = base_path if base_path.is_absolute() else path.parent / base_ref
    base_path = base_path.resolve()
    if not base_path.is_file():
        raise ConfigError(f"{path}: _base not found: {base_ref}")
    return _deep_merge(_load_layered(base_path, _seen), raw)


def _substitute(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match) -> str:
            key = match.group(1)
            if key not in variables:
                raise ConfigError(
                    f"Unknown variable ${{{key}}} (known: {', '.join(sorted(variables))})"
                )
            return variables[key]
        return _VAR_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _substitute(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, variables) for v in value]
    return value


def _apply_overrides(data: dict, overrides: list[str]) -> dict:
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"--set expects key=value, got: {item!r}")
        dotted, raw = item.split("=", 1)
        node: Any = data
        parts = dotted.split(".")
        for part in parts[:-1]:
            nxt = node.setdefault(part, {})
            if not isinstance(nxt, dict):
                raise ConfigError(f"--set {dotted}: '{part}' is not a mapping")
            node = nxt
        node[parts[-1]] = yaml.safe_load(raw)
    return data


def load(ref: str, overrides: list[str] | None = None) -> ExperimentConfig:
    """Resolve an experiment reference into a validated ExperimentConfig."""
    path = resolve_experiment_path(ref)
    data = _load_layered(path)
    data = _apply_overrides(data, overrides or [])

    name = data.get("name")
    if not name:
        raise ConfigError(f"{path}: 'name' is required")
    data = _substitute(data, {"workspace": str(WORKSPACE), "name": str(name)})

    unknown = set(data) - {"name", "dataset", "policy", "train", "tracking", "backend"}
    if unknown:
        raise ConfigError(f"{path}: unknown top-level key(s): {', '.join(sorted(unknown))}")
    for required in ("dataset", "policy"):
        if required not in data:
            raise ConfigError(f"{path}: '{required}' is required")

    try:
        cfg = ExperimentConfig(
            name=str(data["name"]),
            dataset=DatasetConfig(**data["dataset"]),
            policy=dict(data["policy"]),
            train=dict(data.get("train") or {}),
            tracking=TrackingConfig(**(data.get("tracking") or {})),
            backend=str(data.get("backend", "lerobot")),
        )
    except TypeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc

    _validate(cfg, path)
    return cfg


def _validate(cfg: ExperimentConfig, path: Path) -> None:
    try:
        validate_name(cfg.name)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from exc

    if cfg.backend != "lerobot":
        raise ConfigError(f"{path}: unsupported backend {cfg.backend!r} (only 'lerobot' today)")
    if "type" not in cfg.policy:
        raise ConfigError(f"{path}: policy.type is required (e.g. 'act')")

    device = cfg.policy.get("device", "cuda")
    # A device index is accepted by lerobot but silently ignored: accelerate picks
    # the device itself. CUDA_VISIBLE_DEVICES is the only real GPU selector.
    if device not in DEVICES:
        raise ConfigError(
            f"{path}: policy.device must be one of {DEVICES}, got {device!r}\n"
            "To pin a GPU use CUDA_VISIBLE_DEVICES=<n>, not a device index."
        )

    if cfg.tracking.backend not in TRACKING_BACKENDS:
        raise ConfigError(
            f"{path}: tracking.backend must be one of {TRACKING_BACKENDS}, "
            f"got {cfg.tracking.backend!r}"
        )
    if cfg.tracking.backend == "wandb" and not cfg.tracking.project:
        # lerobot starts wandb only when enable AND project are truthy.
        raise ConfigError(f"{path}: tracking.project is required when tracking.backend is 'wandb'")
    if cfg.tracking.mode not in TRACKING_MODES:
        raise ConfigError(f"{path}: tracking.mode must be one of {TRACKING_MODES}")

    root = Path(cfg.dataset.root)
    if not (root / "meta").is_dir():
        raise ConfigError(
            f"{path}: dataset.root has no meta/ directory:\n  {root}\n"
            "Point dataset.root at a prepared LeRobot dataset."
        )
