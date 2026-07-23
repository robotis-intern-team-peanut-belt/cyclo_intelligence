"""CSV persistence for L2 annotations. Enables session resume.

One CSV file holds every annotation across runs/sessions. A record is uniquely
keyed by (run_id, session_id, trial_index) — writing the same key again is an
upsert, so re-scoring an episode overwrites the previous verdict rather than
duplicating it. On startup the frontend calls GET /api/records to repopulate
the UI, which is how "앱 재실행 시 이전 상태 복원" works.

Schema (exact column order required by the L3 aggregator):
    run_id, session_id, policy_id, checkpoint_step, chunk_size,
    trial_index, init_condition_id, y_position, result, failure_code,
    notes, video_path, annotated_by, annotated_at
"""
from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from threading import Lock

SCHEMA = [
    "run_id",
    "session_id",
    "policy_id",
    "checkpoint_step",
    "chunk_size",
    "trial_index",
    "init_condition_id",
    "y_position",
    "result",
    "failure_code",
    "notes",
    "video_path",
    "annotated_by",
    "annotated_at",
]

KEY_FIELDS = ("run_id", "session_id", "trial_index")


def _key(rec: dict):
    return tuple(str(rec.get(k, "")) for k in KEY_FIELDS)


class AnnotationStore:
    """Thread-safe CSV-backed annotation store (single local user)."""

    def __init__(self, csv_path: str | Path):
        self.path = Path(csv_path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._rows: dict = {}  # key tuple -> record dict
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", newline="", encoding="utf-8") as f:
            for rec in csv.DictReader(f):
                # keep only known columns; ignore extras gracefully
                clean = {k: rec.get(k, "") for k in SCHEMA}
                self._rows[_key(clean)] = clean

    def _flush_locked(self) -> None:
        """Atomic write: temp file in the same dir, then os.replace."""
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".csv.tmp")
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=SCHEMA)
                writer.writeheader()
                # stable order: by run_id, session_id, then numeric trial_index
                def sort_key(r):
                    try:
                        ti = int(r.get("trial_index", 0))
                    except (TypeError, ValueError):
                        ti = 0
                    return (r.get("run_id", ""), r.get("session_id", ""), ti)

                for rec in sorted(self._rows.values(), key=sort_key):
                    writer.writerow({k: rec.get(k, "") for k in SCHEMA})
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def upsert(self, record: dict) -> dict:
        clean = {k: ("" if record.get(k) is None else record.get(k, ""))
                 for k in SCHEMA}
        with self._lock:
            self._rows[_key(clean)] = clean
            self._flush_locked()
        return clean

    def all_records(self) -> list:
        with self._lock:
            return list(self._rows.values())

    def records_for(self, run_id: str, session_id: str) -> list:
        with self._lock:
            return [
                r for r in self._rows.values()
                if str(r.get("run_id", "")) == str(run_id)
                and str(r.get("session_id", "")) == str(session_id)
            ]
