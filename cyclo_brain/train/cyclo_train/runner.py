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

"""Run execution: guards, run directory, manifest, and a live-streamed trainer."""

from __future__ import annotations

import datetime as dt
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from . import manifest
from .backends import get_backend
from .config import WORKSPACE, ExperimentConfig

# The package is bind-mounted read-only and never pip-installed, so the console
# script does not exist anywhere. Every hint must name the host entry point.
LAUNCH = "./docker/container.sh train-lerobot"


class RunError(Exception):
    """A run that cannot start."""


def _echo(msg: str) -> None:
    print(f"[cyclo-train] {msg}", flush=True)


def _host_path(path: Path) -> str:
    """Render a container path the way the operator sees it on the host."""
    host_ws = os.environ.get("CYCLO_HOST_WORKSPACE")
    if not host_ws:
        return str(path)
    try:
        return str(Path(host_ws) / path.relative_to(WORKSPACE))
    except ValueError:
        return str(path)


def _wandb_available() -> bool:
    return importlib.util.find_spec("wandb") is not None


def _preflight(cfg: ExperimentConfig, *, resume: bool) -> None:
    if resume:
        if not cfg.run_dir.is_dir():
            raise RunError(
                f"Cannot resume: no run at {_host_path(cfg.run_dir)}\n"
                f"Start it with: {LAUNCH} {cfg.name}"
            )
        if not cfg.last_checkpoint_config.is_file():
            raise RunError(
                f"Cannot resume '{cfg.name}': it has no saved checkpoint yet.\n"
                f"Expected: {_host_path(cfg.last_checkpoint_config)}\n"
                "A run is only resumable once it reaches its first train.save_freq step."
            )
        return

    if cfg.run_dir.exists():
        raise RunError(
            f"A run named '{cfg.name}' already exists.\n"
            f"  continue it:    {LAUNCH} {cfg.name} --resume\n"
            f"  new run:        {LAUNCH} {cfg.name} --set name=<new_name>\n"
            f"  discard it:     sudo rm -rf {_host_path(cfg.run_dir)}"
        )
    if shutil.which(get_backend(cfg.backend).executable) is None:
        raise RunError(
            f"{get_backend(cfg.backend).executable!r} is not on PATH.\n"
            f"Training must run in the training container: {LAUNCH} <experiment>"
        )


def _resolve_tracking(cfg: ExperimentConfig, *, strict: bool) -> bool:
    if cfg.tracking.backend != "wandb":
        return False
    if _wandb_available():
        if not os.environ.get("WANDB_API_KEY") and cfg.tracking.mode == "online":
            _echo(
                "WARNING: tracking.backend=wandb and mode=online, but WANDB_API_KEY is unset.\n"
                "         Put it in docker/.env, or set tracking.mode=offline."
            )
        return True
    msg = (
        "tracking.backend=wandb but the wandb package is not installed here.\n"
        f"Run training through {LAUNCH}, which uses the training image."
    )
    if strict:
        raise RunError(msg)
    _echo(f"WARNING: {msg}")
    _echo("Continuing with local logs only; training is unaffected.")
    return False


def _stream(argv: list[str], log_file: Path, *, append: bool) -> int:
    """Run argv, echoing output live to stdout and to log_file.

    Reads raw byte chunks rather than lines because tqdm redraws progress with
    carriage returns and no newline.
    """
    mode = "ab" if append else "wb"
    with log_file.open(mode) as log:
        if append:
            log.write(f"\n===== resumed {dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ} =====\n".encode())
            log.flush()
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        assert proc.stdout is not None
        fd = proc.stdout.fileno()
        try:
            while True:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                log.write(chunk)
                log.flush()
        except KeyboardInterrupt:
            proc.terminate()
        finally:
            proc.stdout.close()
        code = proc.wait()

    # Popen reports a signal death as a negative number; SystemExit would turn
    # -9 into 247. Convert to the shell's 128+N convention.
    return code if code >= 0 else 128 - code


def run(
    cfg: ExperimentConfig,
    *,
    resume: bool = False,
    dry_run: bool = False,
    strict_tracking: bool = False,
    extra_args: list[str] | None = None,
) -> int:
    """Execute (or preview) a training run. Returns the trainer's exit code."""
    backend = get_backend(cfg.backend)
    wandb_ok = _resolve_tracking(cfg, strict=strict_tracking)
    argv = backend.build_argv(cfg, resume=resume, wandb_available=wandb_ok)
    if extra_args:
        argv += extra_args

    if dry_run:
        _echo(f"DRY RUN — {'resume' if resume else 'fresh run'} '{cfg.name}'")
        _echo(f"run dir:    {_host_path(cfg.run_dir)}")
        _echo(f"output dir: {cfg.output_dir}")
        print("\n" + " \\\n  ".join(argv) + "\n")
        return 0

    _preflight(cfg, resume=resume)

    cfg.run_dir.mkdir(parents=True, exist_ok=resume)
    cfg.resolved_config_file.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    meta = manifest.write(cfg, argv, resume=resume)

    _echo(f"{'Resuming' if resume else 'Starting'} run '{cfg.name}' ({cfg.backend})")
    _echo(f"dataset:  {cfg.dataset.root}")
    _echo(f"run dir:  {_host_path(cfg.run_dir)}")
    _echo(f"log:      {_host_path(cfg.log_file)}")
    _echo(f"tracking: {'wandb → ' + cfg.tracking.project if wandb_ok else 'local logs only'}")
    _echo(f"config:   {meta['config_hash']}  code: {meta['code']['repo_git_sha'][:12]}")
    print()

    code = _stream(argv, cfg.log_file, append=resume)

    print()
    if code == 0:
        _echo(f"Training finished OK. Checkpoints: {_host_path(cfg.output_dir / 'checkpoints')}")
        _echo(f"Publish it for inference:  {LAUNCH} --promote {cfg.name}")
    elif code >= 128:
        _echo(f"Training stopped by signal {code - 128}. Log: {_host_path(cfg.log_file)}")
        if cfg.last_checkpoint_config.is_file():
            _echo(f"It is resumable:  {LAUNCH} {cfg.name} --resume")
        else:
            _echo("No checkpoint was saved yet, so there is nothing to resume from.")
    else:
        _echo(f"Training FAILED (exit {code}). Log: {_host_path(cfg.log_file)}")
    return code
