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

"""Publish a finished checkpoint into the inference dropbox.

`promote` is the only writer to /workspace/model/<backend>/<name>/, so a
half-finished or failed run can never be selected for inference.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import config as config_mod
from .config import ConfigError, validate_name

PRETRAINED_DIR = "pretrained_model"
CHECKPOINTS_DIR = "checkpoints"
LAST_LINK = "last"


class PromoteError(Exception):
    """Promotion could not be completed."""


def _workspace() -> Path:
    # Read through the module so tests (and CYCLO_WORKSPACE) can repoint it.
    return config_mod.WORKSPACE


def _contained(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def resolve_step(output_dir: Path, step: str) -> Path:
    """Return the checkpoint directory for `step` ('last' or e.g. '010000')."""
    ckpt_root = output_dir / CHECKPOINTS_DIR
    if not ckpt_root.is_dir():
        raise PromoteError(
            f"No checkpoints yet at {ckpt_root}\n"
            "The run has not reached its first train.save_freq step."
        )
    target = ckpt_root / step
    if step == LAST_LINK and target.is_dir() and not target.is_symlink():
        raise PromoteError(
            f"{target} is a real directory, not the symlink lerobot maintains.\n"
            "Pass an explicit --step instead."
        )
    if not target.exists():
        available = sorted(
            p.name for p in ckpt_root.iterdir() if p.is_dir() and p.name != LAST_LINK
        )
        raise PromoteError(
            f"No checkpoint {step!r}. Available: {', '.join(available) or '(none)'}"
        )
    return target.resolve()


def promote(
    name: str,
    *,
    step: str = LAST_LINK,
    dest_name: str | None = None,
    backend: str = "lerobot",
    force: bool = False,
) -> Path:
    """Copy a run's checkpoint into the inference dropbox. Returns the dest dir."""
    try:
        validate_name(name, what="run name")
        validate_name(backend, what="backend")
        if dest_name is not None:
            validate_name(dest_name, what="--dest-name")
    except ConfigError as exc:
        raise PromoteError(str(exc)) from exc

    ws = _workspace()
    run_dir = ws / "runs" / name
    if not run_dir.is_dir():
        raise PromoteError(f"No such run: {run_dir}")

    output_dir = run_dir / "output"
    ckpt_dir = resolve_step(output_dir, step)
    src = ckpt_dir / PRETRAINED_DIR
    if not src.is_dir():
        raise PromoteError(f"Checkpoint has no {PRETRAINED_DIR}/: {ckpt_dir}")

    step_name = ckpt_dir.name
    dropbox_root = ws / "model" / backend
    dest_root = dropbox_root / (dest_name or name)
    if not _contained(dest_root, dropbox_root):
        raise PromoteError(f"Refusing to write outside {dropbox_root}: {dest_root}")

    dest = dest_root / CHECKPOINTS_DIR / step_name / PRETRAINED_DIR
    if dest.exists():
        if not force:
            raise PromoteError(f"Already promoted: {dest}\nUse --force to overwrite.")
        shutil.rmtree(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    # pretrained_model/ only: training_state/ is hundreds of MB of optimizer
    # state needed to resume, not to serve, and would be uploaded to the Hub.
    shutil.copytree(src, dest)

    # Mirror lerobot's layout so the inference engine's auto-descent works.
    last_link = dest_root / CHECKPOINTS_DIR / LAST_LINK
    if last_link.is_symlink() or last_link.exists():
        last_link.unlink()
    last_link.symlink_to(step_name)

    manifest_src = run_dir / "manifest.json"
    if manifest_src.is_file():
        try:
            data = json.loads(manifest_src.read_text())
        except json.JSONDecodeError:
            data = {}
        data["promoted"] = {"from_run": name, "step": step_name, "dest": str(dest_root)}
        (dest_root / "manifest.json").write_text(json.dumps(data, indent=2) + "\n")

    return dest_root


def _promoted_from() -> dict[str, str]:
    """Map run name -> published name, read from the dropbox manifests."""
    out: dict[str, str] = {}
    model_root = _workspace() / "model"
    if not model_root.is_dir():
        return out
    for manifest_file in model_root.glob("*/*/manifest.json"):
        try:
            data = json.loads(manifest_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        origin = (data.get("promoted") or {}).get("from_run")
        if origin:
            out[origin] = manifest_file.parent.name
    return out


def list_runs() -> list[dict]:
    """Summarise every run in the workspace."""
    runs_root = _workspace() / "runs"
    if not runs_root.is_dir():
        return []

    published = _promoted_from()
    rows = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        ckpt_root = run_dir / "output" / CHECKPOINTS_DIR
        steps = (
            sorted(p.name for p in ckpt_root.iterdir() if p.is_dir() and p.name != LAST_LINK)
            if ckpt_root.is_dir()
            else []
        )
        meta = {}
        manifest_file = run_dir / "manifest.json"
        if manifest_file.is_file():
            try:
                meta = json.loads(manifest_file.read_text())
            except json.JSONDecodeError:
                meta = {}
        rows.append(
            {
                "name": run_dir.name,
                "steps": steps,
                "last": steps[-1] if steps else None,
                "created_at": meta.get("created_at", "?"),
                "config_hash": meta.get("config_hash", "?"),
                "resumes": len(meta.get("resumes") or []),
                "promoted": published.get(run_dir.name),
            }
        )
    return rows
