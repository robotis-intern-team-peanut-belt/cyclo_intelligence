"""L2 scoring web app — FastAPI backend.

Run it:
    python -m l2_scoring.app --eval-root /path/to/eval_data --csv results/annotations.csv
or point it at nothing and load a folder from the UI. Then open the printed URL.

Endpoints
    GET  /                         -> the single-page scoring UI
    GET  /api/labels               -> failure codes + y-positions (from failure_codes.py)
    GET  /api/dataset?path=...      -> parse folder + episode metadata (read-only)
    GET  /api/video/{ep}?layout=... -> merged, frame-synced mp4 (supports Range/seek)
    GET  /api/records?run_id&session_id -> existing annotations (session resume)
    POST /api/records              -> upsert one annotation
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import prescreen as ps
from .dataset import DatasetInfo, load_dataset
from .failure_codes import FAILURE_CODES, RESULT_FAILURE, RESULT_SUCCESS, Y_POSITIONS
from .storage import SCHEMA, AnnotationStore
from .video_merge import merge_clips

HERE = Path(__file__).parent
STATIC = HERE / "static"


class AnnotationIn(BaseModel):
    run_id: str
    session_id: str
    policy_id: str = ""
    checkpoint_step: str = ""
    chunk_size: str = ""
    trial_index: int
    init_condition_id: str = ""
    y_position: str = ""
    result: str = ""
    failure_code: str = ""
    notes: str = ""
    video_path: str = ""
    annotated_by: str = ""
    annotated_at: str = ""


class RejudgeIn(BaseModel):
    ep_index: int
    roi: list | None = None  # [x, y, w, h] in the head-cam's pixel space


def create_app(csv_path: str, eval_root: str | None = None,
               cache_dir: str | None = None) -> FastAPI:
    app = FastAPI(title="Cyclo eval scorer (L2)")
    store = AnnotationStore(csv_path)
    cache_dir = cache_dir or str(Path(csv_path).parent / "_video_cache")

    results_dir = Path(csv_path).parent

    # app state
    app.state.store = store
    app.state.cache_dir = cache_dir
    app.state.dataset = None  # type: DatasetInfo | None
    app.state.prescreen = {}          # ep_index(int) -> prediction dict
    app.state.prescreen_thumbs = None  # Path to annotated thumbnails, if any

    def set_dataset(ds: DatasetInfo):
        """Set the active dataset and load its pre-screen predictions if present."""
        app.state.dataset = ds
        app.state.prescreen = {}
        pj = results_dir / f"{ds.folder_name}_prescreen.json"
        if pj.exists():
            try:
                doc = json.loads(pj.read_text())
                app.state.prescreen = {int(k): v for k, v in doc.get("predictions", {}).items()}
            except Exception as e:
                print(f"[warn] failed to read prescreen {pj}: {e}")
        tdir = results_dir / f"{ds.folder_name}_thumbs"
        app.state.prescreen_thumbs = tdir if tdir.is_dir() else None

    app.state.set_dataset = set_dataset

    if eval_root:
        try:
            set_dataset(load_dataset(eval_root))
        except Exception as e:  # pragma: no cover
            print(f"[warn] could not preload eval-root: {e}")

    def _episode(ep_index: int):
        ds: DatasetInfo | None = app.state.dataset
        if ds is None:
            raise HTTPException(400, "no dataset loaded; call /api/dataset first")
        for e in ds.episodes:
            if e.index == ep_index:
                return ds, e
        raise HTTPException(404, f"episode {ep_index} not found")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/labels")
    def labels():
        return {
            "failure_codes": FAILURE_CODES,
            "y_positions": Y_POSITIONS,
            "result_success": RESULT_SUCCESS,
            "result_failure": RESULT_FAILURE,
            "schema": SCHEMA,
        }

    @app.get("/api/current")
    def current():
        """The dataset preloaded via --eval-root, if any (for UI auto-load)."""
        ds: DatasetInfo | None = app.state.dataset
        return ds.to_public() if ds is not None else {}

    @app.get("/api/dataset")
    def dataset(path: str = Query(..., description="eval folder path")):
        try:
            ds = load_dataset(path)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except Exception as e:
            raise HTTPException(400, f"failed to read dataset: {e}")
        set_dataset(ds)
        return ds.to_public()

    @app.get("/api/video/{ep_index}")
    def video(ep_index: int, layout: str = "auto"):
        ds, ep = _episode(ep_index)
        if not ep.clips:
            raise HTTPException(404, f"episode {ep_index} has no video files")
        if layout == "auto":
            layout = "grid" if len(ep.clips) == 4 else "horizontal"
        try:
            merged = merge_clips(ep.clips, app.state.cache_dir, layout=layout)
        except Exception as e:
            raise HTTPException(500, f"video merge failed: {e}")
        # FileResponse handles HTTP Range requests -> seeking/scrubbing works.
        return FileResponse(merged, media_type="video/mp4")

    @app.get("/api/records")
    def records(run_id: str | None = None, session_id: str | None = None):
        if run_id is not None and session_id is not None:
            return store.records_for(run_id, session_id)
        return store.all_records()

    @app.post("/api/records")
    def upsert(rec: AnnotationIn):
        saved = store.upsert(rec.model_dump())
        return JSONResponse(saved)

    @app.get("/api/export")
    def export():
        """Download the full annotation CSV."""
        return FileResponse(store.path, media_type="text/csv",
                            filename=Path(store.path).name)

    # ---- auto pre-screen -------------------------------------------------
    @app.get("/api/prescreen")
    def prescreen():
        """All auto predictions for the loaded dataset (empty if none run)."""
        return {"predictions": {str(k): v for k, v in app.state.prescreen.items()},
                "has_thumbs": app.state.prescreen_thumbs is not None}

    @app.get("/api/prescreen/thumb/{ep_index}")
    def prescreen_thumb(ep_index: int):
        tdir = app.state.prescreen_thumbs
        if tdir is None:
            raise HTTPException(404, "no thumbnails")
        p = tdir / f"ep{ep_index}.png"
        if not p.exists():
            raise HTTPException(404, "thumbnail not found")
        return FileResponse(p, media_type="image/png")

    @app.post("/api/prescreen/rejudge")
    def prescreen_rejudge(body: RejudgeIn):
        """Re-run the auto judgment for one episode, optionally with a manual ROI."""
        ds, ep = _episode(body.ep_index)
        roi = tuple(int(v) for v in body.roi) if body.roi else None
        tdir = app.state.prescreen_thumbs or (results_dir / f"{ds.folder_name}_thumbs")
        tdir.mkdir(parents=True, exist_ok=True)
        app.state.prescreen_thumbs = tdir
        pred = ps.judge_episode(ep.clips, roi=roi,
                                thumb_path=tdir / f"ep{ep.index}.png")
        pred["roi_override"] = list(roi) if roi else None
        app.state.prescreen[ep.index] = pred
        return pred

    return app


def main():
    ap = argparse.ArgumentParser(description="L2 eval scoring web app")
    ap.add_argument("--eval-root", default=os.environ.get("EVAL_ROOT"),
                    help="eval dataset folder to preload (optional; can load in UI)")
    ap.add_argument("--csv", default=os.environ.get("EVAL_CSV", "results/annotations.csv"),
                    help="annotation CSV path (created if missing)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    import uvicorn
    app = create_app(csv_path=args.csv, eval_root=args.eval_root)
    print(f"\n  L2 scorer:  http://{args.host}:{args.port}\n"
          f"  CSV:        {Path(args.csv).resolve()}\n")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
