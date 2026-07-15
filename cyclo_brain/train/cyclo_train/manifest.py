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

"""Run manifest: what code, what data, what hardware produced a checkpoint."""

from __future__ import annotations

import datetime as dt
import json
import platform
import socket
import os
from pathlib import Path
from typing import Any

from .config import ExperimentConfig

SCHEMA_VERSION = 1
_UNKNOWN = "unknown"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _env(name: str) -> str:
    return os.environ.get(name) or _UNKNOWN


def _code_provenance() -> dict[str, str]:
    # The repo's .git is not mounted into the container, so container.sh resolves
    # these on the host and passes them in.
    return {
        "repo_git_sha": _env("CYCLO_GIT_SHA"),
        "repo_dirty": _env("CYCLO_GIT_DIRTY"),
        "lerobot_submodule_sha": _env("CYCLO_LEROBOT_SHA"),
        "image": _env("CYCLO_TRAIN_IMAGE"),
    }


def _dataset_provenance(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return {"error": f"no meta/info.json at {root}"}
    try:
        info = json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"error": f"unreadable meta/info.json: {exc}"}

    features = info.get("features") or {}

    def shape(key: str) -> Any:
        return (features.get(key) or {}).get("shape")

    return {
        "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "fps": info.get("fps"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        # Recorded because the state/action width must match what the inference
        # engine assembles at serving time.
        "action_shape": shape("action"),
        "observation_state_shape": shape("observation.state"),
        "camera_keys": sorted(
            k for k, v in features.items() if (v or {}).get("dtype") in {"image", "video"}
        ),
    }


def _runtime_provenance() -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "in_container": Path("/.dockerenv").exists(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES") or "(unset: all visible)",
    }
    # Best effort: the CLI must stay importable where torch is absent.
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as exc:  # noqa: BLE001
        info["torch"] = f"unavailable: {exc}"
    return info


def build(cfg: ExperimentConfig, argv: list[str]) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "name": cfg.name,
        "backend": cfg.backend,
        "created_at": _now(),
        "config_hash": cfg.hash(),
        "code": _code_provenance(),
        "runtime": _runtime_provenance(),
        "dataset": {
            "repo_id": cfg.dataset.repo_id,
            "root": str(cfg.dataset.root),
            **_dataset_provenance(Path(cfg.dataset.root)),
        },
        "paths": {
            "run_dir": str(cfg.run_dir),
            "output_dir": str(cfg.output_dir),
            "dropbox_dir": str(cfg.dropbox_dir),
        },
        "argv": argv,
        "config": cfg.to_dict(),
        "resumes": [],
    }


def write(cfg: ExperimentConfig, argv: list[str], *, resume: bool) -> dict[str, Any]:
    """Write the manifest for a fresh run, or append a resume record to it.

    A resume must never replace the original: doing so would re-stamp the
    checkpoint with today's git SHA and timestamp, and that record is copied to
    the dropbox (and on to HuggingFace) by `promote`.
    """
    if resume and cfg.manifest_file.is_file():
        try:
            data = json.loads(cfg.manifest_file.read_text())
        except (OSError, json.JSONDecodeError):
            data = build(cfg, argv)
        data.setdefault("resumes", []).append(
            {
                "at": _now(),
                "config_hash": cfg.hash(),
                "code": _code_provenance(),
                "argv": argv,
            }
        )
    else:
        data = build(cfg, argv)
        if resume:
            data["resumes"].append({"at": _now(), "note": "resumed with no prior manifest"})

    cfg.manifest_file.write_text(json.dumps(data, indent=2) + "\n")
    return data
