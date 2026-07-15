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

"""Local LeRobot v2.1 -> v3.0 converter, with optional mobile-dim trimming.

Distinct from the upstream converter on purpose:
  - it can trim the mobile DOFs in the same pass, rebuilding per-episode stats
    consistently;
  - it writes a `<name>_v30` sibling and leaves the v2.1 source untouched,
    whereas upstream converts in place;
  - it floors global std at 1e-3, which upstream does not.

The arithmetic here reproduces the datasets already in the workspace. Do not
"improve" the std floor, the pooled-variance formula, or the float32 casts
without re-deriving every dataset that depends on them.

numpy/pandas are imported inside functions to keep `import cyclo_train` light.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import VECTOR_KEYS, TrimPlan, read_info, write_json
from .errors import DataPrepError

STD_FLOOR = 1e-3


@dataclass(frozen=True)
class ConversionResult:
    dst: Path
    episodes: int
    frames: int
    kept_dims: int


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise DataPrepError(f"missing {path}")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _weighted_global_stats(episode_stats: list[dict], plan: TrimPlan) -> dict[str, Any]:
    """Count-weighted pooled mean/std across episodes, with std floored."""
    import numpy as np

    buckets: dict[str, dict[str, list]] = {}
    for row in episode_stats:
        for key, stats in row["stats"].items():
            count_raw = stats.get("count", [0])
            count = int(count_raw[0] if isinstance(count_raw, list) else count_raw)
            if count <= 0:
                continue
            keep = plan.indices_by_key.get(key)

            def sel(name: str):
                values = stats[name]
                return np.asarray(
                    [values[i] for i in keep] if keep is not None else values, dtype=np.float64
                )

            slot = buckets.setdefault(key, {"mean": [], "std": [], "min": [], "max": [], "count": []})
            slot["mean"].append(sel("mean"))
            slot["std"].append(sel("std"))
            slot["min"].append(sel("min"))
            slot["max"].append(sel("max"))
            slot["count"].append(count)

    result: dict[str, Any] = {}
    for key, slot in buckets.items():
        means = np.stack(slot["mean"])
        stds = np.stack(slot["std"])
        counts = np.asarray(slot["count"], dtype=np.float64)
        total = counts.sum()
        weights = counts.reshape((-1,) + (1,) * (means.ndim - 1)) / total
        global_mean = (weights * means).sum(axis=0)
        pooled_var = (weights * (stds**2 + (means - global_mean) ** 2)).sum(axis=0)
        result[key] = {
            "mean": global_mean.tolist(),
            "std": np.maximum(np.sqrt(pooled_var), STD_FLOOR).tolist(),
            "min": np.stack(slot["min"]).min(axis=0).tolist(),
            "max": np.stack(slot["max"]).max(axis=0).tolist(),
        }
    return result


def _flatten_stats(stats: dict) -> dict:
    return {
        f"stats/{key}/{stat}": value
        for key, block in stats.items()
        for stat, value in block.items()
    }


def _trim_episode_stats(stats: dict, plan: TrimPlan) -> dict:
    out = json.loads(json.dumps(stats))
    for key, keep in plan.indices_by_key.items():
        if key not in out:
            continue
        for stat in ("min", "max", "mean", "std"):
            if stat in out[key]:
                out[key][stat] = [out[key][stat][i] for i in keep]
    return out


def convert(
    src: Path,
    dst: Path,
    *,
    no_mobile: bool = False,
    force: bool = False,
) -> ConversionResult:
    """Convert the v2.1 dataset at `src` into a v3.0 dataset at `dst`."""
    import numpy as np
    import pandas as pd

    src, dst = src.resolve(), dst.resolve()
    if dst.exists():
        if not force:
            raise DataPrepError(f"Destination already exists: {dst}\nUse --force to replace it.")
        shutil.rmtree(dst)

    info = read_info(src)
    if info.get("codebase_version") != "v2.1":
        raise DataPrepError(
            f"{src}: expected a v2.1 dataset, found {info.get('codebase_version')!r}"
        )

    source_episodes = int(info["total_episodes"])
    source_frames = int(info["total_frames"])
    fps = info["fps"]
    chunks_size = int(info.get("chunks_size", 1000))
    data_template = info["data_path"]
    video_template = info["video_path"]

    dst.mkdir(parents=True)

    episodes = _load_jsonl(src / "meta" / "episodes.jsonl")
    episode_stats_rows = _load_jsonl(src / "meta" / "episodes_stats.jsonl")
    tasks_rows = _load_jsonl(src / "meta" / "tasks.jsonl")
    stats_by_episode = {row["episode_index"]: row["stats"] for row in episode_stats_rows}

    plan = TrimPlan.for_info(info, require_mobile=no_mobile)
    if no_mobile:
        plan.apply_to_info(info)

    frames = []
    for episode in episodes:
        index = episode["episode_index"]
        # Read via the source's own template rather than a hardcoded chunk-000,
        # so datasets with more than chunks_size episodes convert correctly.
        parquet = src / data_template.format(
            episode_chunk=index // chunks_size, episode_index=index
        )
        if not parquet.is_file():
            raise DataPrepError(f"missing episode data: {parquet}")
        frame = pd.read_parquet(parquet)
        for key in VECTOR_KEYS:
            keep = plan.indices_by_key[key]
            frame[key] = frame[key].map(lambda v, k=keep: np.asarray(v, dtype=np.float32)[k])
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data_path = dst / "data" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(data_path, index=False)

    video_keys = [k for k, v in info["features"].items() if (v or {}).get("dtype") == "video"]
    episode_rows = []
    for episode in episodes:
        index = episode["episode_index"]
        row = {
            "episode_index": index,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": int(data.index[data["episode_index"] == index][0]),
            "dataset_to_index": int(data.index[data["episode_index"] == index][-1]) + 1,
            "tasks": episode["tasks"],
            "length": episode["length"],
            "distance_traveled": episode.get("distance_traveled", 0.0),
            "left_eef_displacement": episode.get("left_eef_displacement", 0.0),
            "right_eef_displacement": episode.get("right_eef_displacement", 0.0),
        }
        for video_key in video_keys:
            src_video = src / video_template.format(
                episode_chunk=index // chunks_size, video_key=video_key, episode_index=index
            )
            _copy_or_link(src_video, dst / "videos" / video_key / "chunk-000" / f"file-{index:03d}.mp4")
            row[f"videos/{video_key}/chunk_index"] = 0
            row[f"videos/{video_key}/file_index"] = index
            row[f"videos/{video_key}/from_timestamp"] = 0.0
            row[f"videos/{video_key}/to_timestamp"] = episode["length"] / fps
        row.update(_flatten_stats(_trim_episode_stats(stats_by_episode[index], plan)))
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0
        episode_rows.append(row)

    episodes_path = dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(episode_rows).to_parquet(episodes_path, index=False)

    task_map = {r["task_index"]: r.get("task") or r["task_name"] for r in tasks_rows}
    pd.DataFrame(
        {"task_index": list(task_map.keys())},
        index=pd.Index(list(task_map.values()), name="task"),
    ).to_parquet(dst / "meta" / "tasks.parquet")

    trimmed = [
        {"episode_index": r["episode_index"], "stats": _trim_episode_stats(r["stats"], plan)}
        for r in episode_stats_rows
    ]
    write_json(dst / "meta" / "stats.json", _weighted_global_stats(trimmed, plan))

    for key, feature in info["features"].items():
        if (feature or {}).get("dtype") != "video":
            feature["fps"] = fps
    info.update(
        {
            "codebase_version": "v3.0",
            "total_episodes": len(episodes),
            "total_frames": int(len(data)),
            "total_tasks": len(task_map),
            "chunks_size": info.get("chunks_size", 10000),
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 1,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "splits": {"train": f"0:{len(episodes)}"},
        }
    )
    for dead in ("total_videos", "total_chunks"):
        info.pop(dead, None)
    write_json(dst / "meta" / "info.json", info)

    for optional in ("README.md", "meta/frame_reuse.parquet", "meta/manual_review_summary.json",
                     "meta/subtasks.parquet"):
        source = src / optional
        if source.exists():
            (dst / optional).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dst / optional)
    if (src / "annotations").is_dir():
        shutil.copytree(src / "annotations", dst / "annotations")

    # Compare against the values captured BEFORE mutating info, so these can
    # actually fail.
    if len(episode_rows) != source_episodes:
        raise DataPrepError(
            f"episode count changed: source {source_episodes}, converted {len(episode_rows)}"
        )
    if len(data) != source_frames:
        raise DataPrepError(f"frame count changed: source {source_frames}, converted {len(data)}")

    return ConversionResult(
        dst=dst,
        episodes=len(episodes),
        frames=int(len(data)),
        kept_dims=len(plan.indices_by_key["action"]),
    )
