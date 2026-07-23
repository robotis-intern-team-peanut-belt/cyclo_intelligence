"""Merge an episode's camera views into ONE synchronized video.

Why merge server-side: several independent HTML5 <video> elements drift out of
sync. Compositing all cameras into a single file makes them share one timeline,
so playback is frame-locked by construction.

Each input is a "clip": {path, start, end, rotate}. `start`/`end` (seconds) cut a
time-slice out of a concatenated per-camera file (LeRobot v3.0); when both are
None the whole file is used (simple per-episode-file layout). `rotate` (deg)
un-rotates cameras stored rotated (e.g. SG2 wrist cams at 270°).

Prefers a system `ffmpeg`; else uses the static binary from `imageio-ffmpeg`.
Merged files are cached, keyed by inputs + slices + rotation + layout.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

GRID_CELL_W = 480
GRID_CELL_H = 360


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "No ffmpeg available. Install it on PATH, or `pip install "
            "imageio-ffmpeg` inside this tool's venv."
        ) from e


def _norm(clips: list) -> list:
    out = []
    for c in clips:
        out.append({
            "path": str(c["path"]),
            "start": c.get("start"),
            "end": c.get("end"),
            "rotate": int(c.get("rotate", 0)) % 360,
        })
    return out


def _cache_key(clips: list, layout: str, height: int) -> str:
    h = hashlib.sha256()
    h.update(f"{layout}|{height}|v3".encode())
    for c in clips:
        h.update(f"{c['path']}|{c['start']}|{c['end']}|{c['rotate']}".encode())
        try:
            h.update(str(Path(c["path"]).stat().st_mtime_ns).encode())
        except OSError:
            pass
    return h.hexdigest()[:16]


def _rotate_filter(deg: int) -> str:
    """ffmpeg transpose chain to un-rotate a clip stored rotated by `deg`."""
    if deg == 90:
        return "transpose=1,"          # 90° clockwise
    if deg == 180:
        return "transpose=2,transpose=2,"
    if deg == 270:
        return "transpose=2,"          # 90° counter-clockwise
    return ""


def _build_filter(clips: list, layout: str, height: int) -> str:
    n = len(clips)
    rot = [_rotate_filter(c["rotate"]) for c in clips]

    if layout == "grid" and n == 4:
        cw, ch = GRID_CELL_W, GRID_CELL_H
        scaled = "".join(
            f"[{i}:v]{rot[i]}scale={cw}:{ch}:force_original_aspect_ratio=decrease,"
            f"pad={cw}:{ch}:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}];"
            for i in range(4)
        )
        layout_str = "0_0|w0_0|0_h0|w0_h0"
        return scaled + "[v0][v1][v2][v3]" + f"xstack=inputs=4:layout={layout_str}[out]"

    if layout == "vertical":
        scaled = "".join(
            f"[{i}:v]{rot[i]}scale=640:-2:force_original_aspect_ratio=decrease,"
            f"pad=640:ih:(ow-iw)/2:0,setsar=1[v{i}];"
            for i in range(n)
        )
        if n == 1:
            return scaled + "[v0]copy[out]"
        return scaled + "".join(f"[v{i}]" for i in range(n)) + f"vstack=inputs={n}[out]"

    # horizontal (default)
    scaled = "".join(
        f"[{i}:v]{rot[i]}scale=-2:{height}:force_original_aspect_ratio=decrease,"
        f"pad=iw:{height}:0:(oh-ih)/2,setsar=1[v{i}];"
        for i in range(n)
    )
    if n == 1:
        return scaled + "[v0]copy[out]"
    return scaled + "".join(f"[v{i}]" for i in range(n)) + f"hstack=inputs={n}[out]"


def merge_clips(clips: list, cache_dir: str | Path,
                layout: str = "horizontal", height: int = 480) -> Path:
    """Cut + composite the given camera clips into one cached mp4."""
    clips = _norm(clips)
    if not clips:
        raise ValueError("no clips given")
    if layout == "grid" and len(clips) != 4:
        layout = "horizontal"  # grid only defined for 4 cams

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"merged_{_cache_key(clips, layout, height)}.mp4"
    if out.exists() and out.stat().st_size > 0:
        return out

    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error"]
    for c in clips:
        # input-level seek/trim: fast and accurate enough for review playback
        if c["start"] is not None:
            cmd += ["-ss", f"{c['start']}"]
        if c["end"] is not None:
            dur = c["end"] - (c["start"] or 0.0)
            if dur > 0:
                cmd += ["-t", f"{dur}"]
        cmd += ["-i", c["path"]]
    cmd += [
        "-filter_complex", _build_filter(clips, layout, height),
        "-map", "[out]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(
            f"ffmpeg merge failed (rc={proc.returncode}):\n{proc.stderr[-2000:]}"
        )
    return out


# Backwards-compatible thin wrapper (plain file paths, whole-file, no rotation).
def merge_episode_videos(video_paths: list, cache_dir, layout="horizontal",
                         height=480) -> Path:
    clips = [{"path": p, "start": None, "end": None, "rotate": 0}
             for p in video_paths]
    return merge_clips(clips, cache_dir, layout=layout, height=height)
