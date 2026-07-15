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

"""Derive a mobile-base-free copy of a v3.0 dataset (22 dims -> 19).

numpy/pandas are imported inside functions so that importing cyclo_train stays
pyyaml-only.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .dataset import (
    VECTOR_KEYS,
    TrimPlan,
    ensure_visual_stats,
    read_info,
    write_json,
)
from .errors import DataPrepError


@dataclass(frozen=True)
class TrimResult:
    dst: Path
    kept_dims: int
    original_dims: int
    episodes_meta_trimmed: bool


def _copytree_hardlink(src: Path, dst: Path) -> None:
    """Clone a tree with hardlinks, so the GB of video cost nothing."""

    def copy_or_link(source: str, dest: str) -> str:
        try:
            os.link(source, dest)
        except OSError:
            shutil.copy2(source, dest)
        return dest

    shutil.copytree(src, dst, copy_function=copy_or_link)


def _detach(path: Path) -> None:
    """Give a hardlinked file its own inode before rewriting it.

    Without this, writing through a hardlink would silently mutate the source
    dataset that the copy shares inodes with.
    """
    if not path.exists() or path.stat().st_nlink <= 1:
        return
    tmp = path.with_name(f".{path.name}.trim_tmp")
    try:
        shutil.copy2(path, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _trim_parquet_vectors(path: Path, plan: TrimPlan) -> None:
    import numpy as np
    import pandas as pd

    df = pd.read_parquet(path)
    for key in VECTOR_KEYS:
        if key not in df.columns:
            continue
        keep = plan.indices_by_key[key]
        # float32 is what the source parquet carries; preserve it.
        df[key] = df[key].map(lambda v, k=keep: np.asarray(v, dtype=np.float32)[k])
    _detach(path)
    df.to_parquet(path, index=False)


def _trim_episode_stats(meta_dir: Path, plan: TrimPlan) -> bool:
    """Trim the flattened per-episode stats columns in meta/episodes/*.parquet.

    These are stats/<vector_key>/{min,max,mean,std}. They are dropped by the
    older tool, which leaves a 19-dim dataset carrying 22-dim per-episode stats
    and breaks downstream merge/split/delete_episodes.
    """
    import numpy as np
    import pandas as pd

    episode_files = sorted((meta_dir / "episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        return False

    stat_names = ("min", "max", "mean", "std")
    touched = False
    for path in episode_files:
        df = pd.read_parquet(path)
        changed = False
        for key in VECTOR_KEYS:
            keep = plan.indices_by_key[key]
            for stat in stat_names:
                column = f"stats/{key}/{stat}"
                if column not in df.columns:
                    continue
                df[column] = df[column].map(
                    lambda v, k=keep: np.asarray(v, dtype=np.float32)[k]
                )
                changed = True
        if changed:
            _detach(path)
            df.to_parquet(path, index=False)
            touched = True
    return touched


def trim_mobile(src: Path, dst: Path, *, force: bool = False) -> TrimResult:
    """Write a 19-dim copy of the 22-dim v3.0 dataset at `src`."""
    if not (src / "meta" / "info.json").is_file():
        raise DataPrepError(f"Not a dataset (no meta/info.json): {src}")
    if dst.exists():
        if not force:
            raise DataPrepError(f"Destination already exists: {dst}\nUse --force to replace it.")
        shutil.rmtree(dst)

    info = read_info(src)
    if info.get("codebase_version") != "v3.0":
        raise DataPrepError(
            f"{src}: expected a v3.0 dataset, found {info.get('codebase_version')!r}.\n"
            "Convert it to v3.0 first."
        )

    plan = TrimPlan.for_info(info, require_mobile=True)
    original_dims = len(info["features"]["action"]["names"])

    _copytree_hardlink(src, dst)

    info_path = dst / "meta" / "info.json"
    stats_path = dst / "meta" / "stats.json"
    info = read_info(dst)
    try:
        stats = json.loads(stats_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DataPrepError(f"{stats_path}: unreadable: {exc}") from exc

    plan.apply_to_info(info)
    for key in VECTOR_KEYS:
        plan.trim_stats_block(stats, key)
    ensure_visual_stats(info, stats)

    _detach(info_path)
    _detach(stats_path)
    write_json(info_path, info)
    write_json(stats_path, stats)

    parquets = sorted((dst / "data").glob("chunk-*/*.parquet"))
    if not parquets:
        raise DataPrepError(f"No parquet files under {dst / 'data'}")
    for path in parquets:
        _trim_parquet_vectors(path, plan)

    episodes_trimmed = _trim_episode_stats(dst / "meta", plan)

    kept = len(plan.indices_by_key["action"])
    return TrimResult(
        dst=dst,
        kept_dims=kept,
        original_dims=original_dims,
        episodes_meta_trimmed=episodes_trimmed,
    )
