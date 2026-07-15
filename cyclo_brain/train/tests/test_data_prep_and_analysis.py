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

"""Data-prep primitives and loss analysis. Pure: no numpy/pandas needed here."""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from cyclo_train.analysis import loss as loss_mod
from cyclo_train.data_prep import dataset as ds
from cyclo_train.data_prep import download
from cyclo_train.data_prep.errors import DataPrepError


# --------------------------------------------------------------------------- #
# purity: importing the package must not drag in numpy/pandas
# --------------------------------------------------------------------------- #
def test_importing_cyclo_train_needs_no_numpy_or_pandas(monkeypatch):
    """The package is bind-mounted into images that must import it unchanged;
    only pyyaml may be required at import time."""
    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {"numpy", "pandas", "pyarrow", "torch"}:
            raise AssertionError(f"{name} imported at cyclo_train import time")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    for module in [
        "cyclo_train",
        "cyclo_train.cli",
        "cyclo_train.config",
        "cyclo_train.promote",
        "cyclo_train.data_prep",
        "cyclo_train.analysis",
    ]:
        importlib = real_import("importlib")
        importlib.import_module(module)


# --------------------------------------------------------------------------- #
# identity + paths
# --------------------------------------------------------------------------- #
def test_org_prefix_is_stripped_from_the_on_disk_name():
    """Keeping the org would download to dataset/<org>/<name>, which nothing reads."""
    assert ds.local_name("RobotisSW/Task_900004_x_lerobot") == "Task_900004_x_lerobot"
    assert ds.local_name("Task_900004_x_lerobot") == "Task_900004_x_lerobot"
    assert ds.dataset_root(Path("/workspace"), "RobotisSW/ds") == Path("/workspace/dataset/ds")


def test_read_info_falls_back_to_a_bare_info_json(tmp_path):
    (tmp_path / "info.json").write_text('{"codebase_version": "v2.1"}')
    assert ds.read_info(tmp_path)["codebase_version"] == "v2.1"
    assert ds.codebase_version(tmp_path) == "v2.1"


def test_codebase_version_of_a_non_dataset_is_unknown(tmp_path):
    assert ds.codebase_version(tmp_path) == "unknown"


# --------------------------------------------------------------------------- #
# the shared trim plan
# --------------------------------------------------------------------------- #
def _info(names):
    return {
        "features": {
            "observation.state": {"names": list(names), "shape": [len(names)]},
            "action": {"names": list(names), "shape": [len(names)]},
        }
    }


MOBILE = ["j1", "j2", "linear_x", "linear_y", "angular_z"]


def test_trim_selects_by_name_not_position():
    plan = ds.TrimPlan.for_info(_info(MOBILE), require_mobile=True)
    assert plan.indices_by_key["action"] == [0, 1]


def test_trim_updates_names_and_shape():
    info = _info(MOBILE)
    plan = ds.TrimPlan.for_info(info, require_mobile=True)
    plan.apply_to_info(info)
    assert info["features"]["action"]["names"] == ["j1", "j2"]
    assert info["features"]["action"]["shape"] == [2]


def test_trim_refuses_a_dataset_without_the_mobile_tail():
    """Guards against trimming an already-trimmed dataset down again."""
    with pytest.raises(DataPrepError, match="does not end with"):
        ds.TrimPlan.for_info(_info(["j1", "j2", "j3"]), require_mobile=True)


def test_trim_stats_leaves_count_alone():
    plan = ds.TrimPlan.for_info(_info(MOBILE), require_mobile=True)
    stats = {"action": {"mean": [1, 2, 3, 4, 5], "std": [1, 2, 3, 4, 5], "min": [1, 2, 3, 4, 5],
                        "max": [1, 2, 3, 4, 5], "count": [999]}}
    plan.trim_stats_block(stats, "action")
    assert stats["action"]["mean"] == [1, 2]
    assert stats["action"]["count"] == [999]  # length-1: never index-trimmed


def test_trim_stats_raises_a_typed_error_for_a_missing_key():
    plan = ds.TrimPlan.for_info(_info(MOBILE), require_mobile=True)
    with pytest.raises(DataPrepError, match="no entry for"):
        plan.trim_stats_block({}, "action")


def test_ensure_visual_stats_is_setdefault_only():
    info = {"features": {"observation.images.cam": {"dtype": "video"}}}
    stats = {}
    ds.ensure_visual_stats(info, stats)
    assert stats["observation.images.cam"]["mean"] == ds.IMAGENET_MEAN
    stats["observation.images.cam"]["mean"] = "mine"
    ds.ensure_visual_stats(info, stats)
    assert stats["observation.images.cam"]["mean"] == "mine"


# --------------------------------------------------------------------------- #
# download argv
# --------------------------------------------------------------------------- #
def test_download_argv():
    argv = download.build_download_argv("Org/ds", Path("/workspace/dataset/ds"))
    assert argv == [
        "hf", "download", "Org/ds", "--repo-type", "dataset",
        "--local-dir", "/workspace/dataset/ds",
    ]


def test_convert_argv_always_disables_push_to_hub():
    """Upstream defaults push_to_hub to True: omitting the flag can publish."""
    argv = download.build_convert_argv("Org/ds", Path("/ws/dataset/ds"))
    assert "--push-to-hub=false" in argv
    assert argv[:3] == ["python", "-m", download.UPSTREAM_CONVERTER]


def test_prepare_is_idempotent_for_a_v30_dataset(tmp_path, capsys):
    """Re-running must not re-download a v2.1 tree over a converted one."""
    ws = tmp_path
    root = ws / "dataset" / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text('{"codebase_version": "v3.0"}')
    result = download.prepare("Org/ds", ws)
    assert result.version == "v3.0"
    assert result.converted is False
    assert "nothing to do" in capsys.readouterr().out


def test_prepare_dry_run_touches_nothing(tmp_path, capsys):
    result = download.prepare("Org/ds", tmp_path, dry_run=True)
    assert not (tmp_path / "dataset" / "ds").exists()
    assert "would run" in capsys.readouterr().out
    assert result.version == "(dry-run)"


# --------------------------------------------------------------------------- #
# loss analysis
# --------------------------------------------------------------------------- #
LOG = (
    "Training:   1%| | 10/100 [00:01<00:10]INFO step:10 loss:42.7 grdn:8.0\n"
    "Training:   2%| | 20/100 [00:02<00:09]INFO step:20 loss:16.1 grdn:4.0\n"
    "Training:   3%| | 30/100 [00:03<00:08]INFO step:30 loss:11.2 grdn:3.0\n"
)


def test_parse_prefers_the_exact_tqdm_step():
    points = loss_mod.parse_loss_points(LOG)
    assert [(p.step, p.loss) for p in points] == [(10, 42.7), (20, 16.1), (30, 11.2)]


def test_parse_expands_abbreviated_steps():
    text = "INFO step:1.5K loss:2.0\n"
    assert loss_mod.parse_loss_points(text)[0].step == 1500


def test_parse_handles_carriage_return_frames():
    """runner._stream writes raw bytes, so tqdm redraws land on one line."""
    text = "Training: 1%| | 10/100\rTraining: 2%| | 20/100 INFO step:20 loss:5.0\n"
    assert loss_mod.parse_loss_points(text)[0].step == 20


def test_select_never_emits_the_same_point_twice():
    """A duplicate fabricates a zero-change interval that reads as 'stable'."""
    points = [loss_mod.LossPoint(10, 5.0), loss_mod.LossPoint(100, 4.0)]
    selected = loss_mod.select_interval_points(points, 10)
    assert len(selected) == len({p.step for p in selected})


def test_plateau_is_detected_after_consecutive_stable_intervals():
    points = [loss_mod.LossPoint(s, loss) for s, loss in
              [(10, 10.0), (20, 5.0), (30, 4.99), (40, 4.98)]]
    analysis = loss_mod.analyze(points, threshold=0.05, consecutive=2)
    assert analysis.recommended_step == 40


def test_no_plateau_when_loss_keeps_dropping():
    points = [loss_mod.LossPoint(s, loss) for s, loss in
              [(10, 100.0), (20, 50.0), (30, 25.0), (40, 12.0)]]
    assert loss_mod.analyze(points).recommended_step is None


def test_analysis_needs_two_points():
    with pytest.raises(loss_mod.LossAnalysisError):
        loss_mod.analyze([loss_mod.LossPoint(10, 1.0)])


def test_empty_log_raises_a_typed_error(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("nothing useful here\n")
    with pytest.raises(loss_mod.LossAnalysisError, match="no loss records"):
        loss_mod.read_loss_points(log)


def test_report_renders(tmp_path):
    points = [loss_mod.LossPoint(s, loss) for s, loss in [(10, 10.0), (20, 5.0)]]
    text = loss_mod.format_report(loss_mod.analyze(points))
    assert "step" in text and "Recommended step" in text
