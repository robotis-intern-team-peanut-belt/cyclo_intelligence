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

"""Host entry point: `docker/container.sh train-lerobot`.

These live with the training feature rather than in docker/test_container_sh.py
because they pin down the training contract specifically: which service is used,
how the subcommand maps to `cyclo-train`, and that provenance is injected.

Docker is stubbed out — nothing here starts a container.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTAINER_SH = REPO_ROOT / "docker" / "container.sh"

pytestmark = pytest.mark.skipif(
    not CONTAINER_SH.is_file(), reason="container.sh not found (running outside the repo)"
)


@pytest.fixture
def harness(tmp_path):
    """A throwaway repo with container.sh and a docker stub that records calls."""
    docker_dir = tmp_path / "docker"
    docker_dir.mkdir()
    shutil.copy2(CONTAINER_SH, docker_dir / "container.sh")
    (docker_dir / "container.sh").chmod(0o755)
    (docker_dir / "config").mkdir()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "docker"
    # Record argv AND the CYCLO_* provenance env the compose call would inherit.
    stub.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$DOCKER_STUB_LOG"\n'
        'env | grep "^CYCLO_" | sort >> "$DOCKER_STUB_ENV"\n'
        'if [ "$1" = image ] && [ "$2" = inspect ]; then printf "sha256:present\\n"; fi\n'
        "exit 0\n"
    )
    stub.chmod(0o755)

    # tmux must not exist, so --no-tmux is not silently required for every test.
    (bin_dir / "tmux").write_text("#!/bin/sh\nexit 1\n")
    (bin_dir / "tmux").chmod(0o755)

    log = tmp_path / "docker.log"
    envlog = tmp_path / "docker.env"

    def run(*args):
        result = subprocess.run(
            [str(docker_dir / "container.sh"), "train-lerobot", *args],
            cwd=tmp_path,
            env={
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "DOCKER_STUB_LOG": str(log),
                "DOCKER_STUB_ENV": str(envlog),
                "CYCLO_STORAGE_MODE": "local",
            },
            text=True,
            capture_output=True,
        )
        return (
            result,
            log.read_text() if log.exists() else "",
            envlog.read_text() if envlog.exists() else "",
        )

    return run


def test_requires_an_experiment(harness):
    result, _, _ = harness("--no-tmux")
    assert result.returncode != 0
    assert "no experiment given" in result.stderr


def test_runs_the_train_service_as_a_batch_job(harness):
    """Must use `compose run --rm` on the profile-gated lerobot_train service —
    never `docker exec` into the inference container the way train_act.sh did."""
    result, calls, _ = harness("act_smoke", "--no-tmux")
    assert result.returncode == 0, result.stderr
    assert "--profile train run --rm lerobot_train" in calls
    assert "python3 -m cyclo_train run act_smoke" in calls
    assert "exec lerobot_server" not in calls


def test_forwards_set_overrides(harness):
    _, calls, _ = harness("act_smoke", "--no-tmux", "--set", "train.steps=100")
    assert "python3 -m cyclo_train run act_smoke --set train.steps=100" in calls


def test_resume_uses_the_resume_subcommand(harness):
    _, calls, _ = harness("act_smoke", "--resume", "--no-tmux")
    assert "python3 -m cyclo_train resume act_smoke" in calls


def test_promote_passes_the_run_name(harness):
    """Regression: --promote used to drop the run name before reaching the CLI."""
    result, calls, _ = harness("--promote", "my_run")
    assert result.returncode == 0, result.stderr
    assert "python3 -m cyclo_train promote my_run" in calls


def test_promote_without_a_name_errors(harness):
    result, _, _ = harness("--promote")
    assert result.returncode != 0
    assert "--promote needs a run name" in result.stderr


def test_list_runs_without_an_experiment(harness):
    result, calls, _ = harness("--list")
    assert result.returncode == 0, result.stderr
    assert "python3 -m cyclo_train list" in calls


def test_injects_git_provenance(harness):
    """The repo's .git is deliberately not mounted, so the container cannot work
    these out for itself — container.sh must export them for the manifest."""
    _, _, envlog = harness("act_smoke", "--no-tmux")
    assert "CYCLO_GIT_SHA=" in envlog
    assert "CYCLO_GIT_DIRTY=" in envlog
    assert "CYCLO_LEROBOT_SHA=" in envlog
    assert "CYCLO_TRAIN_IMAGE=robotis/lerobot-train:" in envlog


def test_data_prep_subcommands_dispatch_directly(harness):
    """`train-lerobot download <repo>` must reach the CLI as `download`, not be
    mistaken for an experiment name."""
    for sub, arg, expect in [
        ("download", "Org/ds", "cyclo_train download Org/ds"),
        ("trim-mobile", "ds", "cyclo_train trim-mobile ds"),
        ("convert", "ds", "cyclo_train convert ds"),
        ("analyze-loss", "my_run", "cyclo_train analyze-loss my_run"),
    ]:
        result, calls, _ = harness(sub, arg)
        assert result.returncode == 0, result.stderr
        assert expect in calls, f"{sub}: {calls!r}"


def test_an_experiment_named_like_nothing_special_still_runs(harness):
    _, calls, _ = harness("act_smoke", "--no-tmux")
    assert "cyclo_train run act_smoke" in calls


def test_subcommand_flags_are_forwarded(harness):
    _, calls, _ = harness("trim-mobile", "ds", "--force")
    assert "cyclo_train trim-mobile ds --force" in calls
