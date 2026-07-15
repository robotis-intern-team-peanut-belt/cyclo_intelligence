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

"""Config resolution + argv construction.

No GPU, no dataset, no container: the backend only builds strings, which is the
whole point of keeping it free of side effects.
"""

from __future__ import annotations

import textwrap

import pytest

from cyclo_train import config as config_mod
from cyclo_train.backends import get_backend
from cyclo_train.config import ConfigError


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A fake /workspace with a dataset that passes the meta/ check."""
    ws = tmp_path / "ws"
    (ws / "dataset" / "ds" / "meta").mkdir(parents=True)
    monkeypatch.setattr(config_mod, "WORKSPACE", ws)
    return ws


@pytest.fixture
def experiments(tmp_path, monkeypatch, workspace):
    exp = tmp_path / "experiments"
    (exp / "_base").mkdir(parents=True)
    (exp / "_base" / "act.yaml").write_text(
        textwrap.dedent(
            """
            backend: lerobot
            policy:
              type: act
              device: cuda
              kl_weight: 10.0
              dropout: 0.1
            train:
              batch_size: 8
              steps: 80000
            tracking:
              backend: none
              project: cyclo-lerobot
            """
        )
    )
    (exp / "demo.yaml").write_text(
        textwrap.dedent(
            """
            _base: _base/act.yaml
            name: demo_run
            dataset:
              repo_id: Org/ds
              root: ${workspace}/dataset/ds
            """
        )
    )
    monkeypatch.setattr(config_mod, "EXPERIMENTS_DIR", exp)
    return exp


def test_base_merge_and_variable_substitution(experiments, workspace):
    cfg = config_mod.load("demo")
    assert cfg.name == "demo_run"
    assert cfg.policy["kl_weight"] == 10.0  # inherited from _base
    assert cfg.dataset.root == str(workspace / "dataset" / "ds")  # ${workspace} expanded


def test_child_overrides_base(experiments):
    (experiments / "kl5.yaml").write_text(
        "_base: _base/act.yaml\n"
        "name: kl5_run\n"
        "dataset: {repo_id: Org/ds, root: '${workspace}/dataset/ds'}\n"
        "policy: {kl_weight: 5.0}\n"
    )
    cfg = config_mod.load("kl5")
    assert cfg.policy["kl_weight"] == 5.0  # overridden
    assert cfg.policy["type"] == "act"  # deep-merged, not replaced


def test_set_override_is_yaml_typed(experiments):
    cfg = config_mod.load("demo", ["train.steps=100", "name=smoke"])
    assert cfg.train["steps"] == 100
    assert isinstance(cfg.train["steps"], int)
    assert cfg.name == "smoke"


def test_run_dir_is_not_output_dir(experiments):
    """lerobot raises FileExistsError if --output_dir exists, so our metadata must
    live one level above it."""
    cfg = config_mod.load("demo")
    assert cfg.output_dir.parent == cfg.run_dir
    assert cfg.manifest_file.parent == cfg.run_dir


def test_missing_dataset_root_is_rejected(experiments, workspace):
    (experiments / "bad.yaml").write_text(
        "_base: _base/act.yaml\nname: bad\ndataset: {repo_id: Org/x, root: /nope}\n"
    )
    with pytest.raises(ConfigError, match="no meta/"):
        config_mod.load("bad")


def test_unknown_top_level_key_is_rejected(experiments):
    (experiments / "typo.yaml").write_text(
        "_base: _base/act.yaml\nname: t\ndataset: {repo_id: o/x, root: '${workspace}/dataset/ds'}\n"
        "polcy: {}\n"
    )
    with pytest.raises(ConfigError, match="unknown top-level"):
        config_mod.load("typo")


def test_bad_name_rejected(experiments):
    with pytest.raises(ConfigError, match="must match"):
        config_mod.load("demo", ["name=has spaces/slash"])


def test_wandb_requires_project(experiments):
    with pytest.raises(ConfigError, match="tracking.project is required"):
        config_mod.load("demo", ["tracking.backend=wandb", "tracking.project="])


# --------------------------------------------------------------------------- #
# argv
# --------------------------------------------------------------------------- #
def test_argv_matches_the_historical_pipeline(experiments):
    cfg = config_mod.load("demo")
    argv = get_backend("lerobot").build_argv(cfg, wandb_available=False)

    assert argv[0] == "lerobot-train"
    assert "--policy.type=act" in argv
    assert "--policy.device=cuda" in argv
    assert "--policy.kl_weight=10.0" in argv
    assert "--policy.dropout=0.1" in argv
    assert "--batch_size=8" in argv
    assert "--steps=80000" in argv
    assert f"--output_dir={cfg.output_dir}" in argv
    assert "--job_name=demo_run" in argv
    assert "--resume=false" in argv


def test_push_to_hub_is_forced_off_and_cannot_be_re_enabled(experiments):
    """lerobot's own default is push_to_hub=True (configs/policies.py:70). The old
    pipeline relied on every caller remembering to disable it."""
    cfg = config_mod.load("demo", ["policy.push_to_hub=true"])
    argv = get_backend("lerobot").build_argv(cfg, wandb_available=False)
    assert argv.count("--policy.push_to_hub=false") == 1
    assert "--policy.push_to_hub=true" not in argv


def test_booleans_render_lowercase(experiments):
    cfg = config_mod.load("demo", ["policy.use_amp=true"])
    argv = get_backend("lerobot").build_argv(cfg, wandb_available=False)
    assert "--policy.use_amp=true" in argv


def test_none_values_are_omitted(experiments):
    cfg = config_mod.load("demo", ["policy.pretrained_path=null"])
    argv = get_backend("lerobot").build_argv(cfg, wandb_available=False)
    assert not any(a.startswith("--policy.pretrained_path") for a in argv)


def test_pretrained_path_passed_when_set(experiments):
    cfg = config_mod.load("demo", ["policy.pretrained_path=/some/ckpt"])
    argv = get_backend("lerobot").build_argv(cfg, wandb_available=False)
    assert "--policy.pretrained_path=/some/ckpt" in argv


def test_wandb_argv(experiments):
    cfg = config_mod.load(
        "demo", ["tracking.backend=wandb", "tracking.project=p", "tracking.entity=e"]
    )
    argv = get_backend("lerobot").build_argv(cfg, wandb_available=True)
    assert "--wandb.enable=true" in argv
    assert "--wandb.project=p" in argv
    assert "--wandb.entity=e" in argv
    assert "--wandb.disable_artifact=true" in argv


def test_wandb_degrades_when_unavailable(experiments):
    """A missing wandb must never kill an 80k-step run."""
    cfg = config_mod.load("demo", ["tracking.backend=wandb", "tracking.project=p"])
    argv = get_backend("lerobot").build_argv(cfg, wandb_available=False)
    assert "--wandb.enable=false" in argv
    assert not any(a.startswith("--wandb.project") for a in argv)


def test_resume_derives_config_path(experiments):
    """The bug that made `RESUME=true ./train_act.sh` silently unusable:
    lerobot refuses to resume without --config_path (configs/train.py:150-155)."""
    cfg = config_mod.load("demo")
    argv = get_backend("lerobot").build_argv(cfg, resume=True)
    assert argv == [
        "lerobot-train",
        f"--config_path={cfg.last_checkpoint_config}",
        "--resume=true",
    ]
    assert str(cfg.last_checkpoint_config).endswith(
        "output/checkpoints/last/pretrained_model/train_config.json"
    )


def test_config_hash_is_stable_and_sensitive(experiments):
    a = config_mod.load("demo")
    b = config_mod.load("demo")
    c = config_mod.load("demo", ["policy.kl_weight=5.0"])
    assert a.hash() == b.hash()
    assert a.hash() != c.hash()
