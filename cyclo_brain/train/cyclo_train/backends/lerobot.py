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

"""LeRobot backend: builds the upstream ``lerobot-train`` command line.

The lerobot submodule tracks upstream, so it is wrapped and never patched.
"""

from __future__ import annotations

from ..config import ExperimentConfig
from .base import TrainerBackend, fmt_value, register


@register
class LeRobotBackend(TrainerBackend):
    name = "lerobot"
    executable = "lerobot-train"

    def build_argv(
        self,
        cfg: ExperimentConfig,
        *,
        resume: bool = False,
        wandb_available: bool = True,
    ) -> list[str]:
        if resume:
            return self._resume_argv(cfg)
        return self._fresh_argv(cfg, wandb_available)

    # ------------------------------------------------------------------ #
    def _resume_argv(self, cfg: ExperimentConfig) -> list[str]:
        """lerobot refuses to resume without --config_path, so it is derived from
        the run's own checkpoints/last symlink.

        Deliberately minimal: on resume lerobot rebuilds the config from the
        checkpoint's train_config.json, so any flag passed here would layer on top
        and diverge from the original run.
        """
        return [
            self.executable,
            f"--config_path={cfg.last_checkpoint_config}",
            "--resume=true",
        ]

    def _fresh_argv(self, cfg: ExperimentConfig, wandb_available: bool) -> list[str]:
        argv = [
            self.executable,
            f"--dataset.repo_id={cfg.dataset.repo_id}",
            f"--dataset.root={cfg.dataset.root}",
            f"--output_dir={cfg.output_dir}",
            f"--job_name={cfg.name}",
        ]
        argv += self._policy_argv(cfg)
        argv += self._train_argv(cfg)
        argv += self._tracking_argv(cfg, wandb_available)
        argv.append("--resume=false")
        return argv

    def _policy_argv(self, cfg: ExperimentConfig) -> list[str]:
        policy = dict(cfg.policy)
        argv = [f"--policy.type={policy.pop('type')}"]

        # accelerate picks the device itself and reads policy.device only to force
        # CPU, so a device index here would be ignored. CUDA_VISIBLE_DEVICES is the
        # GPU selector.
        argv.append(f"--policy.device={policy.pop('device', 'cuda')}")

        pretrained = policy.pop("pretrained_path", None)
        if pretrained:
            argv.append(f"--policy.pretrained_path={pretrained}")

        # Any remaining key becomes --policy.<k>=<v>.
        policy.pop("push_to_hub", None)  # forced below
        for key in sorted(policy):
            rendered = fmt_value(policy[key])
            if rendered is not None:
                argv.append(f"--policy.{key}={rendered}")

        # lerobot defaults push_to_hub to True, so this is forced off
        # unconditionally rather than left to each caller. Config cannot
        # re-enable it.
        argv.append("--policy.push_to_hub=false")
        return argv

    def _train_argv(self, cfg: ExperimentConfig) -> list[str]:
        argv = []
        for key in sorted(cfg.train):
            rendered = fmt_value(cfg.train[key])
            if rendered is not None:
                argv.append(f"--{key}={rendered}")
        return argv

    def _tracking_argv(self, cfg: ExperimentConfig, wandb_available: bool) -> list[str]:
        if cfg.tracking.backend != "wandb" or not wandb_available:
            return ["--wandb.enable=false"]

        argv = [
            "--wandb.enable=true",
            f"--wandb.project={cfg.tracking.project}",
            f"--wandb.mode={cfg.tracking.mode}",
            # Checkpoints are already on disk; uploading them as artifacts on
            # every save is pure cost.
            "--wandb.disable_artifact=true",
        ]
        if cfg.tracking.entity:
            argv.append(f"--wandb.entity={cfg.tracking.entity}")
        if cfg.tracking.notes:
            argv.append(f"--wandb.notes={cfg.tracking.notes}")
        return argv
