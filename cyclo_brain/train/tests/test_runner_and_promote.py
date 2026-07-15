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

"""The stateful layer: run directory, log, manifest, exit codes, promotion.

No GPU and no lerobot: the trainer is stubbed with a plain python subprocess.
"""

from __future__ import annotations

import json
import sys
import textwrap

import pytest

from cyclo_train import config as config_mod
from cyclo_train import manifest, promote as promote_mod, runner
from cyclo_train.cli import main as cli_main
from cyclo_train.config import ConfigError
from cyclo_train.promote import PromoteError


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    (ws / "dataset" / "ds" / "meta").mkdir(parents=True)
    (ws / "dataset" / "ds" / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "total_episodes": 2,
                "total_frames": 10,
                "features": {
                    "action": {"shape": [19], "dtype": "float32"},
                    "observation.state": {"shape": [19], "dtype": "float32"},
                },
            }
        )
    )
    monkeypatch.setattr(config_mod, "WORKSPACE", ws)
    return ws


@pytest.fixture
def experiments(tmp_path, monkeypatch, workspace):
    exp = tmp_path / "experiments"
    exp.mkdir()
    (exp / "demo.yaml").write_text(
        textwrap.dedent(
            """
            name: demo_run
            backend: lerobot
            dataset: {repo_id: Org/ds, root: '${workspace}/dataset/ds'}
            policy: {type: act}
            train: {steps: 1}
            tracking: {backend: none}
            """
        )
    )
    monkeypatch.setattr(config_mod, "EXPERIMENTS_DIR", exp)
    return exp


def _cfg(overrides=None):
    return config_mod.load("demo", overrides or [])


# --------------------------------------------------------------------------- #
# _stream: logging + exit codes
# --------------------------------------------------------------------------- #
def _run_python(code: str, log, *, append=False) -> int:
    return runner._stream([sys.executable, "-c", code], log, append=append)


def test_stream_appends_on_resume_instead_of_truncating(tmp_path):
    """The log of the run being resumed is the artifact you resume in order to
    consult; resuming must not erase it."""
    log = tmp_path / "train.log"
    _run_python("print('FIRST RUN')", log, append=False)
    _run_python("print('SECOND RUN')", log, append=True)
    text = log.read_text()
    assert "FIRST RUN" in text
    assert "SECOND RUN" in text
    assert "resumed" in text


def test_stream_truncates_on_a_fresh_run(tmp_path):
    log = tmp_path / "train.log"
    _run_python("print('OLD')", log, append=False)
    _run_python("print('NEW')", log, append=False)
    assert "OLD" not in log.read_text()


def test_stream_returns_the_child_exit_code(tmp_path):
    assert _run_python("import sys; sys.exit(0)", tmp_path / "a.log") == 0
    assert _run_python("import sys; sys.exit(3)", tmp_path / "b.log") == 3


@pytest.mark.parametrize("sig,expected", [(15, 143), (9, 137), (2, 130)])
def test_stream_converts_signal_deaths_to_shell_convention(tmp_path, sig, expected):
    """Popen reports -N for a signal death; passing that to SystemExit turns
    SIGKILL into 247, which breaks the container's exit-code contract."""
    code = _run_python(
        f"import os, signal; os.kill(os.getpid(), {sig})", tmp_path / f"s{sig}.log"
    )
    assert code == expected


# --------------------------------------------------------------------------- #
# manifest: resume must not rewrite provenance
# --------------------------------------------------------------------------- #
def test_resume_preserves_the_original_provenance(experiments, monkeypatch):
    cfg = _cfg()
    cfg.run_dir.mkdir(parents=True)

    monkeypatch.setenv("CYCLO_GIT_SHA", "aaaaaaaaaaaa")
    first = manifest.write(cfg, ["lerobot-train"], resume=False)

    monkeypatch.setenv("CYCLO_GIT_SHA", "bbbbbbbbbbbb")
    second = manifest.write(cfg, ["lerobot-train", "--resume=true"], resume=True)

    assert second["created_at"] == first["created_at"]
    assert second["code"]["repo_git_sha"] == "aaaaaaaaaaaa"
    assert len(second["resumes"]) == 1
    assert second["resumes"][0]["code"]["repo_git_sha"] == "bbbbbbbbbbbb"


def test_manifest_records_dataset_dims(experiments):
    """The state/action width is the contract with the inference engine."""
    cfg = _cfg()
    cfg.run_dir.mkdir(parents=True)
    data = manifest.write(cfg, [], resume=False)
    assert data["dataset"]["action_shape"] == [19]
    assert data["dataset"]["observation_state_shape"] == [19]


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #
def test_fresh_run_refuses_an_existing_run_dir(experiments):
    cfg = _cfg()
    cfg.run_dir.mkdir(parents=True)
    with pytest.raises(runner.RunError, match="already exists"):
        runner._preflight(cfg, resume=False)


def test_resume_without_a_checkpoint_is_refused(experiments):
    cfg = _cfg()
    cfg.run_dir.mkdir(parents=True)
    with pytest.raises(runner.RunError, match="no saved checkpoint"):
        runner._preflight(cfg, resume=True)


def test_errors_name_the_host_entry_point_not_a_nonexistent_binary(experiments):
    """`cyclo-train` is installed nowhere: the package is bind-mounted read-only
    on PYTHONPATH."""
    cfg = _cfg()
    cfg.run_dir.mkdir(parents=True)
    with pytest.raises(runner.RunError) as exc:
        runner._preflight(cfg, resume=False)
    assert "container.sh train-lerobot" in str(exc.value)


# --------------------------------------------------------------------------- #
# names + promotion containment
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["..", ".", "../evil", "has space", "-leading"])
def test_traversal_and_junk_names_are_rejected(experiments, bad):
    with pytest.raises(ConfigError):
        config_mod.load("demo", [f"name={bad}"])


def _finished_run(ws, name="demo_run", step="000050"):
    ckpt = ws / "runs" / name / "output" / "checkpoints" / step / "pretrained_model"
    ckpt.mkdir(parents=True)
    (ckpt / "model.safetensors").write_text("weights")
    (ckpt / "train_config.json").write_text("{}")
    state = ws / "runs" / name / "output" / "checkpoints" / step / "training_state"
    state.mkdir(parents=True)
    (state / "optimizer_state.safetensors").write_text("huge")
    (ws / "runs" / name / "output" / "checkpoints" / "last").symlink_to(step)
    (ws / "runs" / name / "manifest.json").write_text(json.dumps({"code": {"repo_git_sha": "abc"}}))
    return ws / "runs" / name


def test_promote_publishes_the_engine_layout(workspace):
    _finished_run(workspace)
    dest = promote_mod.promote("demo_run")
    assert (dest / "checkpoints" / "000050" / "pretrained_model" / "model.safetensors").is_file()
    assert (dest / "checkpoints" / "last").is_symlink()
    assert (dest / "checkpoints" / "last").readlink().name == "000050"


def test_promote_excludes_optimizer_state(workspace):
    _finished_run(workspace)
    dest = promote_mod.promote("demo_run")
    assert not (dest / "checkpoints" / "000050" / "training_state").exists()


def test_promote_carries_provenance(workspace):
    _finished_run(workspace)
    dest = promote_mod.promote("demo_run")
    data = json.loads((dest / "manifest.json").read_text())
    assert data["promoted"]["from_run"] == "demo_run"
    assert data["code"]["repo_git_sha"] == "abc"


@pytest.mark.parametrize("bad", ["../../escape", "..", "with/slash"])
def test_promote_refuses_to_write_outside_the_dropbox(workspace, bad):
    _finished_run(workspace)
    with pytest.raises(PromoteError):
        promote_mod.promote("demo_run", dest_name=bad)


def test_promote_refuses_an_unknown_run(workspace):
    with pytest.raises(PromoteError, match="No such run"):
        promote_mod.promote("nope")


def test_promote_needs_force_to_overwrite(workspace):
    _finished_run(workspace)
    promote_mod.promote("demo_run")
    with pytest.raises(PromoteError, match="Already promoted"):
        promote_mod.promote("demo_run")
    promote_mod.promote("demo_run", force=True)


def test_list_reports_the_published_name(workspace):
    _finished_run(workspace)
    promote_mod.promote("demo_run", dest_name="shipped_v1")
    rows = {r["name"]: r for r in promote_mod.list_runs()}
    assert rows["demo_run"]["promoted"] == "shipped_v1"
    assert rows["demo_run"]["last"] == "000050"


# --------------------------------------------------------------------------- #
# CLI argument handling
# --------------------------------------------------------------------------- #
def test_set_before_double_dash_still_parses(experiments, capsys):
    """argparse only keeps a trailing `*` positional when `--` immediately
    follows the experiment, so this must not go through one."""
    rc = cli_main(
        ["run", "demo", "--set", "train.steps=5", "--dry-run", "--", "--policy.n_obs_steps=2"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "--steps=5" in out
    assert "--policy.n_obs_steps=2" in out


def test_resume_rejects_overrides_it_would_silently_ignore(experiments, capsys):
    rc = cli_main(["resume", "demo", "--set", "train.steps=5"])
    assert rc == 2
    assert "would be ignored" in capsys.readouterr().err


def test_resume_allows_selecting_the_run_by_name(experiments, capsys):
    """name= legitimately picks which run to resume, so it is the one override
    that is not silently discarded by the checkpoint's config."""
    rc = cli_main(["resume", "demo", "--set", "name=other_run", "--dry-run"])
    assert rc == 0
    assert "resume 'other_run'" in capsys.readouterr().out
