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

"""Detect when ACT training loss has plateaued, from a run's train.log."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

PROGRESS_RE = re.compile(r"Training:.*?\|\s*(\d+)/(\d+)")
LOSS_RE = re.compile(r"\bstep:\s*(\d+(?:\.\d+)?)([KMG]?)\b.*?\bloss:\s*([0-9.eE+-]+)")
STEP_MULTIPLIER = {"": 1, "K": 1_000, "M": 1_000_000, "G": 1_000_000_000}


class LossAnalysisError(Exception):
    """The log could not be analysed."""


@dataclass(frozen=True)
class LossPoint:
    step: int
    loss: float


@dataclass(frozen=True)
class IntervalRow:
    step: int
    loss: float
    absolute_change: float | None
    relative_change: float | None
    stable: bool


@dataclass(frozen=True)
class Analysis:
    rows: list[IntervalRow]
    recommended_step: int | None
    threshold: float
    consecutive: int
    metric: str


def _abbreviated_step(value: str, suffix: str) -> int:
    return round(float(value) * STEP_MULTIPLIER[suffix])


def parse_loss_points(text: str) -> list[LossPoint]:
    """Extract (step, loss) from a training log.

    The trainer's raw byte stream is written straight to the log, so tqdm's
    carriage-return redraws put many logical frames on one physical line — hence
    the inner splitlines(). The exact step from the progress bar is preferred
    over lerobot's rounded `step: 10K` label.
    """
    points: dict[int, float] = {}
    latest_progress_step: int | None = None

    for physical_line in text.splitlines(keepends=True):
        for part in physical_line.splitlines():
            progress = list(PROGRESS_RE.finditer(part))
            if progress:
                latest_progress_step = int(progress[-1].group(1))
            for match in LOSS_RE.finditer(part):
                displayed = _abbreviated_step(match.group(1), match.group(2))
                step = latest_progress_step if latest_progress_step is not None else displayed
                points[step] = float(match.group(3))

    return [LossPoint(step, loss) for step, loss in sorted(points.items())]


def read_loss_points(path: Path) -> list[LossPoint]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise LossAnalysisError(f"cannot read {path}: {exc}") from exc
    points = parse_loss_points(text)
    if not points:
        raise LossAnalysisError(f"no loss records found in {path}")
    return points


def select_interval_points(points: list[LossPoint], interval: int) -> list[LossPoint]:
    """Pick the logged point nearest each interval boundary.

    A point is never emitted twice: a duplicate would fabricate a zero-change
    interval that reads as 'stable' and can trigger a premature recommendation.
    """
    if not points:
        raise LossAnalysisError("no loss points")
    last_step = points[-1].step
    selected: list[LossPoint] = []
    for target in range(interval, last_step + 1, interval):
        nearest = min(points, key=lambda p: abs(p.step - target))
        if abs(nearest.step - target) > interval // 2:
            continue
        if selected and nearest.step == selected[-1].step:
            continue
        selected.append(nearest)
    return selected


def analyze(
    points: list[LossPoint],
    *,
    threshold: float = 0.05,
    consecutive: int = 2,
    metric: str = "relative",
) -> Analysis:
    if len(points) < 2:
        raise LossAnalysisError("need at least 2 interval points to compare")

    rows: list[IntervalRow] = []
    stable_run = 0
    recommended: int | None = None

    for index, point in enumerate(points):
        previous = points[index - 1] if index else None
        absolute = abs(point.loss - previous.loss) if previous else None
        relative = (
            absolute / abs(previous.loss)
            if previous and previous.loss != 0 and absolute is not None
            else None
        )
        change = relative if metric == "relative" else absolute
        stable = change is not None and change <= threshold
        stable_run = stable_run + 1 if stable else 0
        if recommended is None and stable_run >= consecutive:
            recommended = point.step
        rows.append(IntervalRow(point.step, point.loss, absolute, relative, stable))

    return Analysis(rows, recommended, threshold, consecutive, metric)


def format_report(analysis: Analysis) -> str:
    lines = [
        "step       loss       abs_change   rel_change   stable",
        "---------- ---------- ------------ ------------ ------",
    ]
    for row in analysis.rows:
        abs_txt = "-" if row.absolute_change is None else f"{row.absolute_change:.6f}"
        rel_txt = "-" if row.relative_change is None else f"{row.relative_change * 100:.2f}%"
        lines.append(
            f"{row.step:>10,d} {row.loss:>10.6f} {abs_txt:>12} {rel_txt:>12} "
            f"{'yes' if row.stable else 'no':>6}"
        )
    condition = (
        f"relative change <= {analysis.threshold * 100:.2f}%"
        if analysis.metric == "relative"
        else f"absolute change <= {analysis.threshold:g}"
    )
    lines.append("")
    lines.append(f"Stable when: {condition}, for {analysis.consecutive} consecutive intervals")
    if analysis.recommended_step is None:
        lines.append("Recommended step: not stabilised yet.")
    else:
        lines.append(f"Recommended step: {analysis.recommended_step:,}")
    return "\n".join(lines)


def write_csv(path: Path, analysis: Analysis) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "loss", "absolute_change", "relative_change", "stable"])
        for row in analysis.rows:
            writer.writerow(
                [row.step, row.loss, row.absolute_change, row.relative_change, row.stable]
            )
