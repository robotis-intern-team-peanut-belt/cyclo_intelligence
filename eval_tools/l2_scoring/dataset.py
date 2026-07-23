"""Read-only inspection of a recorded eval dataset folder.

The eval folder is treated as READ-ONLY. Two dataset layouts are supported and
auto-detected:

A) **Simple / per-episode-file** (the task spec's example) — one video file per
   (episode, camera):

    <root>/
    ├── meta/episodes.json         # list of dicts OR jsonl
    └── videos/episode_0_camera_top.mp4 ...

B) **LeRobot v3.0** (what the real RobotisSW datasets use) — episodes are stored
   as time-slices inside concatenated per-camera mp4s, with metadata in parquet:

    <root>/
    ├── meta/info.json                     # video_path template, fps, robot_type
    ├── meta/episodes/chunk-*/file-*.parquet
    └── videos/<video_key>/chunk-000/file-000.mp4   # many episodes per file

For both layouts each Episode exposes a list of `clips`, one per camera:
    {"camera": str, "path": abs mp4, "start": float|None, "end": float|None,
     "rotate": int}
`start/end` are None for layout A (whole file) and the [from,to] timestamps for
layout B. `rotate` (deg) comes from the dataset's camera rotation config.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# folder name -> policy metadata. Tolerant of extra text around the tokens.
_STEP_RE = re.compile(r"step[_-]?(\d+)", re.IGNORECASE)
_CHUNK_RE = re.compile(r"chunk[_-]?(\d+)", re.IGNORECASE)


def parse_folder_name(folder_name: str) -> dict:
    """Extract policy_id / checkpoint_step / chunk_size from a folder name.

    ``policy_A_step10000_chunk50`` -> policy_id=policy_A, step=10000, chunk=50.
    Any token that isn't present comes back as None (human fills it in the UI).
    """
    step_m = _STEP_RE.search(folder_name)
    chunk_m = _CHUNK_RE.search(folder_name)
    cut = len(folder_name)
    for m in (step_m, chunk_m):
        if m is not None:
            cut = min(cut, m.start())
    policy_id = folder_name[:cut].rstrip("_-") or folder_name
    return {
        "policy_id": policy_id,
        "checkpoint_step": int(step_m.group(1)) if step_m else None,
        "chunk_size": int(chunk_m.group(1)) if chunk_m else None,
    }


@dataclass
class Episode:
    index: int
    length: int | None
    task: str
    clips: list = field(default_factory=list)  # [{camera,path,start,end,rotate}]

    @property
    def camera_names(self) -> list:
        return [c["camera"] for c in self.clips]


@dataclass
class DatasetInfo:
    root: str
    folder_name: str
    policy_id: str | None
    checkpoint_step: int | None
    chunk_size: int | None
    episodes: list
    layout: str = "simple"           # "simple" | "lerobot_v30"
    robot_type: str | None = None
    fps: int | None = None
    warnings: list = field(default_factory=list)

    def to_public(self) -> dict:
        return {
            "root": self.root,
            "folder_name": self.folder_name,
            "policy_id": self.policy_id,
            "checkpoint_step": self.checkpoint_step,
            "chunk_size": self.chunk_size,
            "layout": self.layout,
            "robot_type": self.robot_type,
            "fps": self.fps,
            "num_episodes": len(self.episodes),
            "warnings": self.warnings,
            "episodes": [
                {"index": e.index, "length": e.length, "task": e.task,
                 "cameras": e.camera_names}
                for e in self.episodes
            ],
        }

    def episode(self, index: int) -> Episode | None:
        for e in self.episodes:
            if e.index == index:
                return e
        return None


# ---------------------------------------------------------------------------
# layout detection
# ---------------------------------------------------------------------------
def _is_lerobot_v30(root: Path) -> bool:
    return ((root / "meta" / "info.json").exists()
            and any((root / "meta" / "episodes").glob("**/*.parquet"))
            and (root / "videos").is_dir())


def load_dataset(root: str | Path) -> DatasetInfo:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"eval folder not found: {root}")
    if _is_lerobot_v30(root):
        return _load_lerobot_v30(root)
    return _load_simple(root)


# ---------------------------------------------------------------------------
# layout B: LeRobot v3.0
# ---------------------------------------------------------------------------
def _short_cam(video_key: str) -> str:
    """observation.images.rgb.cam_left_head -> cam_left_head"""
    return video_key.split(".")[-1]


def _camera_rotations(root: Path) -> dict:
    """Read per-camera rotation (deg) from the conversion config, if present.

    RobotisSW datasets keep it in the top-level info.json under
    conversion_config.camera_rotations, e.g. {"cam_left_wrist": 270, ...}.
    """
    top = root / "info.json"
    if top.exists():
        try:
            cfg = json.load(open(top)).get("conversion_config", {})
            return {k: int(v) for k, v in cfg.get("camera_rotations", {}).items()}
        except Exception:
            pass
    return {}


def _load_lerobot_v30(root: Path) -> DatasetInfo:
    import pandas as pd

    warnings: list = []
    info = json.load(open(root / "meta" / "info.json"))
    fps = info.get("fps")
    robot_type = info.get("robot_type")
    video_path_tmpl = info.get(
        "video_path",
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    )
    video_keys = [k for k, v in info.get("features", {}).items()
                  if isinstance(v, dict) and v.get("dtype") == "video"]
    if not video_keys:
        video_keys = [k for k in info.get("features", {}) if "images" in k]
    rotations = _camera_rotations(root)

    parquets = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    df = df.sort_values("episode_index")

    episodes: list = []
    for _, row in df.iterrows():
        idx = int(row["episode_index"])
        length = int(row["length"]) if "length" in row and row["length"] is not None else None
        task = row.get("tasks")
        if isinstance(task, (list, tuple)) or hasattr(task, "tolist"):
            try:
                task = list(task)
            except TypeError:
                task = [task]
            task = task[0] if task else ""
        task = str(task) if task is not None else ""

        clips = []
        for vk in video_keys:
            ci = row.get(f"videos/{vk}/chunk_index")
            fi = row.get(f"videos/{vk}/file_index")
            frm = row.get(f"videos/{vk}/from_timestamp")
            to = row.get(f"videos/{vk}/to_timestamp")
            if ci is None or fi is None:
                continue
            rel = video_path_tmpl.format(video_key=vk, chunk_index=int(ci),
                                         file_index=int(fi))
            cam = _short_cam(vk)
            clips.append({
                "camera": cam,
                "path": str(root / rel),
                "start": float(frm) if frm is not None else None,
                "end": float(to) if to is not None else None,
                "rotate": int(rotations.get(cam, 0)),
            })
        missing = [c["camera"] for c in clips if not Path(c["path"]).exists()]
        if missing and idx == int(df["episode_index"].iloc[0]):
            warnings.append(f"some camera files not found on disk (e.g. {missing}).")
        episodes.append(Episode(idx, length, task, clips))

    if fps:
        warnings.insert(0, f"LeRobot v3.0 dataset: {len(episodes)} episodes, "
                           f"{len(video_keys)} cameras @ {fps}fps, robot={robot_type}.")

    parsed = parse_folder_name(root.name)
    return DatasetInfo(
        root=str(root), folder_name=root.name,
        policy_id=parsed["policy_id"], checkpoint_step=parsed["checkpoint_step"],
        chunk_size=parsed["chunk_size"], episodes=episodes,
        layout="lerobot_v30", robot_type=robot_type,
        fps=int(fps) if fps else None, warnings=warnings,
    )


# ---------------------------------------------------------------------------
# layout A: simple per-episode files
# ---------------------------------------------------------------------------
def _load_json_or_jsonl(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        doc = json.loads(text)
        if isinstance(doc, list):
            return doc
        if isinstance(doc, dict):
            for key in ("episodes", "data", "items"):
                if isinstance(doc.get(key), list):
                    return doc[key]
            return [doc]
    except json.JSONDecodeError:
        pass
    return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


def _first(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _discover_simple_clips(videos_dir: Path, ep_index: int) -> list:
    if not videos_dir.is_dir():
        return []
    clips, seen = [], set()
    for pat in (f"episode_{ep_index}_*", f"episode_{ep_index:06d}_*"):
        for vid in sorted(videos_dir.glob(pat)):
            if vid.suffix.lower() not in (".mp4", ".avi", ".mov", ".mkv"):
                continue
            name = vid.stem
            for prefix in (f"episode_{ep_index}_", f"episode_{ep_index:06d}_"):
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break
            cam = name[len("camera_"):] if name.startswith("camera_") else name
            if cam in seen:
                continue
            seen.add(cam)
            clips.append({"camera": cam, "path": str(vid),
                          "start": None, "end": None, "rotate": 0})
    return clips


def _load_simple(root: Path) -> DatasetInfo:
    warnings: list = []
    meta_dir = root / "meta"
    videos_dir = root / "videos"
    parsed = parse_folder_name(root.name)

    records = []
    for cand in (meta_dir / "episodes.json", meta_dir / "episodes.jsonl"):
        if cand.exists():
            records = _load_json_or_jsonl(cand)
            break
    else:
        warnings.append("meta/episodes.json not found — episodes inferred from videos.")

    episodes: list = []
    if records:
        for i, rec in enumerate(records):
            rec = rec if isinstance(rec, dict) else {}
            idx = int(_first(rec, "episode_index", "index", "episode", default=i))
            length = _first(rec, "length", "num_frames", "n_frames", "frames")
            task = _first(rec, "task", "tasks", "task_instruction", "instruction", default="")
            if isinstance(task, list):
                task = task[0] if task else ""
            episodes.append(Episode(idx, length, str(task),
                                    _discover_simple_clips(videos_dir, idx)))
    else:
        found = set()
        if videos_dir.is_dir():
            for vid in videos_dir.glob("episode_*"):
                m = re.match(r"episode_(\d+)_", vid.stem)
                if m:
                    found.add(int(m.group(1)))
        for idx in sorted(found):
            episodes.append(Episode(idx, None, "",
                                    _discover_simple_clips(videos_dir, idx)))

    if not episodes:
        warnings.append("no episodes discovered (checked meta/ and videos/).")
    no_vid = [e.index for e in episodes if not e.clips]
    if no_vid:
        warnings.append(f"{len(no_vid)} episode(s) have no video files: "
                        f"{no_vid[:10]}{'...' if len(no_vid) > 10 else ''}")

    return DatasetInfo(
        root=str(root), folder_name=root.name,
        policy_id=parsed["policy_id"], checkpoint_step=parsed["checkpoint_step"],
        chunk_size=parsed["chunk_size"], episodes=episodes,
        layout="simple", robot_type=None, warnings=warnings,
    )
