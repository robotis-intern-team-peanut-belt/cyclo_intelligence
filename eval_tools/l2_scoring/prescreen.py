"""Automatic first-pass success/failure screening for the peanut-in-box task.

Idea (classical CV, no model, no GPU): the "peanut mix" is a vivid yellow snack
packet, far more saturated than the tan cardboard box or the white table. So:

  1. Detect the box ROI (largest tan/cardboard blob) — the box moves per session,
     so it is detected per episode; a human-supplied ROI can override it.
  2. Detect bright-yellow blobs (the packets) by HSV threshold.
  3. Judge: measure yellow area INSIDE the box vs outside over the last frames.
       - plenty inside, mostly inside      -> auto SUCCESS
       - yellow only outside / none at all  -> auto FAILURE
       - small / partial / box not found    -> AMBIGUOUS (send to human)

This is a *pre-screen*: high-confidence cases are auto-labeled, only AMBIGUOUS
(and any the human distrusts) need manual scoring. All thresholds are module
constants so they are trivial to tune for a different object/lighting.

Standalone batch use:
    python -m l2_scoring.prescreen --eval-root <dataset> --out results/<name>_prescreen.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# ---- tunable constants ----------------------------------------------------
# OpenCV HSV: H 0-179, S/V 0-255. Yellow floor sits above the cardboard's
# measured saturation ceiling (~128), so tan is not mistaken for a packet.
YELLOW_LO = np.array([18, 150, 110])
YELLOW_HI = np.array([40, 255, 255])
TAN_LO = np.array([8, 30, 70])
TAN_HI = np.array([28, 150, 210])

MIN_BOX_AREA_FRAC = 0.02     # box contour must cover >=2% of the frame
# The tan blob finds the box FLOOR; the back wall/rim rises above it in the
# image, so packets leaning on the back read as "outside". Grow the counting
# ROI (esp. upward) to include the box opening. Fractions of the detected h/w.
BOX_MARGIN_TOP = 0.55
BOX_MARGIN_SIDE = 0.08
BOX_MARGIN_BOT = 0.08
MIN_YELLOW_PX = 1200         # below this, treat as "no packet visible"
FRAC_IN_SUCCESS = 0.60       # >= this share of yellow inside box -> success-ish
FRAC_IN_FAIL = 0.25          # <= this share inside -> failure-ish (packet outside)

SCORE_SUCCESS = 0.60         # final score bands
SCORE_FAIL = 0.20

LAST_WINDOW_S = 1.2          # sample frames from the final N seconds
N_SAMPLES = 6                # frames sampled near the end (max-yellow wins)

PRED_SUCCESS = "success"
PRED_FAILURE = "failure"
PRED_AMBIGUOUS = "ambiguous"


# ---- primitives -----------------------------------------------------------
def detect_box(bgr) -> tuple | None:
    """Return (x, y, w, h) of the largest tan/cardboard region, or None."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, TAN_LO, TAN_HI)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    H, W = bgr.shape[:2]
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_BOX_AREA_FRAC * H * W:
        return None
    return tuple(int(v) for v in cv2.boundingRect(c))


def yellow_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def expand_box(box: tuple, shape) -> tuple:
    """Grow the box ROI (mostly upward) and clip to the frame."""
    x, y, w, h = box
    H, W = shape[:2]
    x0 = max(0, int(x - BOX_MARGIN_SIDE * w))
    x1 = min(W, int(x + w + BOX_MARGIN_SIDE * w))
    y0 = max(0, int(y - BOX_MARGIN_TOP * h))
    y1 = min(H, int(y + h + BOX_MARGIN_BOT * h))
    return (x0, y0, x1 - x0, y1 - y0)


def judge_frame(bgr, roi: tuple | None = None) -> dict:
    """Measure yellow inside/outside the box for a single frame.

    If `roi` is given (human override) it is used verbatim; otherwise the box is
    auto-detected and grown by the BOX_MARGIN_* margins.
    """
    if roi is not None:
        box = tuple(int(v) for v in roi)
    else:
        raw = detect_box(bgr)
        box = expand_box(raw, bgr.shape) if raw is not None else None
    ym = yellow_mask(bgr)
    total = int(ym.sum() // 255)
    in_box = 0
    if box is not None:
        x, y, w, h = box
        in_box = int(ym[y:y + h, x:x + w].sum() // 255)
    return {"box": box, "total_yellow": total, "in_box": in_box,
            "out_box": total - in_box}


def _score(m: dict) -> float:
    """0 (fail) .. 1 (success). Combines 'is a packet present' with 'is it inside'."""
    total = m["total_yellow"]
    if total < MIN_YELLOW_PX:
        return 0.0                      # no packet seen -> fail-side
    frac_in = m["in_box"] / total
    present = min(1.0, m["in_box"] / MIN_YELLOW_PX)
    return float(frac_in * present)


def annotate(bgr, m: dict):
    """Draw the box ROI (green) and yellow mask (red) for human review."""
    vis = bgr.copy()
    ym = yellow_mask(bgr)
    vis[ym > 0] = (0, 0, 255)
    if m.get("box") is not None:
        x, y, w, h = m["box"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return vis


# ---- episode-level judgment ----------------------------------------------
def _pick_head_clip(clips: list, camera: str | None):
    if camera:
        for c in clips:
            if c["camera"] == camera:
                return c
    for want in ("left_head", "head"):        # prefer a head camera
        for c in clips:
            if want in c["camera"]:
                return c
    return clips[0] if clips else None


def judge_episode(clips: list, camera: str | None = None,
                  roi: tuple | None = None, thumb_path: Path | None = None) -> dict:
    """Sample the final frames of the head-cam clip and produce a prediction."""
    clip = _pick_head_clip(clips, camera)
    if clip is None:
        return {"pred": PRED_AMBIGUOUS, "confidence": 0.0, "reason": "no video clip"}

    cap = cv2.VideoCapture(clip["path"])
    if not cap.isOpened():
        return {"pred": PRED_AMBIGUOUS, "confidence": 0.0, "reason": "cannot open video"}
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1e-6)
    end = clip["end"] if clip["end"] is not None else dur
    start = max((clip["start"] or 0.0), end - LAST_WINDOW_S)
    ts = np.linspace(start, max(start, end - 0.05), N_SAMPLES)

    best = None
    for t in ts:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        m = judge_frame(frame, roi=roi)
        m["_frame"] = frame
        # keep the frame with the most in-box yellow (packet settled / least occluded)
        if best is None or m["in_box"] > best["in_box"]:
            best = m
    cap.release()
    if best is None:
        return {"pred": PRED_AMBIGUOUS, "confidence": 0.0, "reason": "no frames read",
                "camera": clip["camera"]}

    total = best["total_yellow"]
    frac_in = best["in_box"] / total if total else 0.0
    score = round(_score(best), 3)
    reason = None

    if best["box"] is None:
        # can't gate without a box -> always defer to a human
        pred, conf = PRED_AMBIGUOUS, 0.0
        reason = "box not detected (draw an ROI to override)"
    elif total < MIN_YELLOW_PX:
        # absence of yellow is unreliable (packet may be occluded / still held)
        pred, conf = PRED_AMBIGUOUS, 0.0
        reason = "no packet visible near end (occluded? still held?)"
    elif frac_in >= FRAC_IN_SUCCESS:
        pred = PRED_SUCCESS
        conf = round(min(1.0, (frac_in - FRAC_IN_SUCCESS) / (1 - FRAC_IN_SUCCESS)), 3)
    elif frac_in <= FRAC_IN_FAIL:
        pred = PRED_FAILURE
        conf = round(min(1.0, (FRAC_IN_FAIL - frac_in) / max(FRAC_IN_FAIL, 1e-6)), 3)
        reason = "packet detected mostly OUTSIDE the box"
    else:
        pred, conf = PRED_AMBIGUOUS, round(0.5 - abs(frac_in - 0.5), 3)
        reason = "packet partly in / partly out"

    frame = best.pop("_frame")
    if thumb_path is not None:
        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(thumb_path), annotate(frame, best))
    return {
        "pred": pred, "confidence": round(conf, 3), "score": round(score, 3),
        "in_box": best["in_box"], "out_box": best["out_box"],
        "total_yellow": best["total_yellow"], "box": best["box"],
        "camera": clip["camera"], "reason": reason,
    }


def judge_dataset(ds, camera=None, thumbs_dir: Path | None = None) -> dict:
    out = {}
    for ep in ds.episodes:
        tp = (thumbs_dir / f"ep{ep.index}.png") if thumbs_dir else None
        out[ep.index] = judge_episode(ep.clips, camera=camera, thumb_path=tp)
    return out


def main():
    from .dataset import load_dataset
    ap = argparse.ArgumentParser(description="Auto pre-screen peanut-in-box success/fail")
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--results-dir", default="results",
                    help="where the L2 app reads CSV/predictions (default: results)")
    ap.add_argument("--out", default=None,
                    help="predictions JSON (default: <results-dir>/<folder>_prescreen.json)")
    ap.add_argument("--camera", default=None, help="camera key to use (default: a head cam)")
    ap.add_argument("--thumbs", default=None,
                    help="dir for review thumbnails (default: <results-dir>/<folder>_thumbs)")
    ap.add_argument("--no-thumbs", action="store_true", help="skip writing thumbnails")
    args = ap.parse_args()

    ds = load_dataset(args.eval_root)
    # Default names follow the convention the L2 app auto-discovers.
    rdir = Path(args.results_dir)
    args.out = args.out or str(rdir / f"{ds.folder_name}_prescreen.json")
    if args.no_thumbs:
        thumbs = None
    else:
        thumbs = Path(args.thumbs) if args.thumbs else rdir / f"{ds.folder_name}_thumbs"
    preds = judge_dataset(ds, camera=args.camera, thumbs_dir=thumbs)

    counts = {}
    for p in preds.values():
        counts[p["pred"]] = counts.get(p["pred"], 0) + 1
    out = {"dataset": ds.folder_name, "root": ds.root, "camera": args.camera,
           "counts": counts, "predictions": {str(k): v for k, v in preds.items()}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[prescreen] {ds.folder_name}: {counts}  -> {args.out}")
    if thumbs:
        print(f"[prescreen] review thumbnails in {thumbs}")


if __name__ == "__main__":
    main()
