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

"""Dataset paths, identity, and metadata. Pure: json + pathlib only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import DataPrepError

# Trailing mobile-base DOFs. The tail must match exactly and in this order.
MOBILE_NAMES: tuple[str, ...] = ("linear_x", "linear_y", "angular_z")
VECTOR_KEYS: tuple[str, ...] = ("observation.state", "action")
# 'count' is a length-1 list, so it is never index-trimmed.
TRIMMED_STATS: tuple[str, ...] = ("min", "max", "mean", "std")

IMAGENET_MEAN = [[[0.485]], [[0.456]], [[0.406]]]
IMAGENET_STD = [[[0.229]], [[0.224]], [[0.225]]]


def local_name(repo_id: str) -> str:
    """Directory name for a hub repo id.

    The org prefix is stripped: every consumer (experiment configs, the trainer)
    expects dataset/<name>, so keeping it would download to a path nothing reads.
    """
    return repo_id.split("/")[-1]


def dataset_root(workspace: Path, repo_id_or_name: str) -> Path:
    return workspace / "dataset" / local_name(repo_id_or_name)


def read_info(root: Path) -> dict[str, Any]:
    """Load a dataset's meta/info.json, falling back to a bare info.json."""
    for rel in ("meta/info.json", "info.json"):
        path = root / rel
        if path.is_file():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                raise DataPrepError(f"{path}: invalid JSON: {exc}") from exc
    raise DataPrepError(f"No info.json under {root}")


def codebase_version(root: Path) -> str:
    try:
        return str(read_info(root).get("codebase_version", "unknown"))
    except DataPrepError:
        return "unknown"


def write_json(path: Path, data: Any) -> None:
    """Match the on-disk convention of the datasets already shipped."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4, ensure_ascii=False)
        fh.write("\n")


def vector_names(info: dict[str, Any], key: str) -> list[str]:
    feature = (info.get("features") or {}).get(key)
    if not feature:
        raise DataPrepError(f"info.json has no feature {key!r}")
    return list(feature["names"])


def kept_indices(names: list[str]) -> list[int]:
    """Indices surviving a mobile trim, selected by NAME not position."""
    return [i for i, n in enumerate(names) if n not in MOBILE_NAMES]


def has_mobile_tail(names: list[str]) -> bool:
    return tuple(names[-3:]) == MOBILE_NAMES


class TrimPlan:
    """Which columns survive per vector key: the single source of truth for 22 -> 19.

    Pure index selection, so it is dimension-independent and testable with lists.
    """

    def __init__(self, indices_by_key: dict[str, list[int]]) -> None:
        self.indices_by_key = indices_by_key

    @classmethod
    def for_info(cls, info: dict[str, Any], *, require_mobile: bool) -> TrimPlan:
        indices: dict[str, list[int]] = {}
        for key in VECTOR_KEYS:
            names = vector_names(info, key)
            if not has_mobile_tail(names):
                if require_mobile:
                    raise DataPrepError(
                        f"{key} does not end with {list(MOBILE_NAMES)}: {names[-3:]}\n"
                        "Refusing to trim: the dataset is already trimmed, or the "
                        "mobile DOFs are not where they are expected."
                    )
                indices[key] = list(range(len(names)))
            else:
                indices[key] = kept_indices(names)
        return cls(indices)

    def apply_to_info(self, info: dict[str, Any]) -> None:
        for key, keep in self.indices_by_key.items():
            feature = info["features"][key]
            names = list(feature["names"])
            feature["names"] = [names[i] for i in keep]
            feature["shape"] = [len(keep)]

    def trim_stats_block(self, stats: dict[str, Any], key: str) -> None:
        if key not in stats:
            raise DataPrepError(f"stats has no entry for {key!r}")
        keep = self.indices_by_key[key]
        for stat in TRIMMED_STATS:
            if stat in stats[key]:
                stats[key][stat] = [stats[key][stat][i] for i in keep]

    def trim_sequence(self, values: list[Any], key: str) -> list[Any]:
        keep = self.indices_by_key[key]
        return [values[i] for i in keep]


def ensure_visual_stats(info: dict[str, Any], stats: dict[str, Any]) -> None:
    """Fill in ImageNet defaults for image/video features that lack stats."""
    for key, feature in (info.get("features") or {}).items():
        if (feature or {}).get("dtype") not in {"image", "video"}:
            continue
        block = stats.setdefault(key, {})
        block.setdefault("mean", IMAGENET_MEAN)
        block.setdefault("std", IMAGENET_STD)
