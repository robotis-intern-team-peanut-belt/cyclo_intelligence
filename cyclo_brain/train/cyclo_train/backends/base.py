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

"""TrainerBackend contract.

A backend's only job is to turn an ExperimentConfig into an argv. It never runs
anything, touches the filesystem, or imports the training library — which makes
argv construction unit-testable without a GPU, a dataset, or a container.

Adding a policy backend (e.g. GR00T) = one module implementing this ABC + one
base YAML. No changes to the runner, CLI, manifest, or promote.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..config import ExperimentConfig


def fmt_value(value: Any) -> str | None:
    """Render a YAML scalar the way draccus (lerobot's CLI parser) expects.

    Returns None for values that should be omitted entirely.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        # draccus/lerobot dumps these as lowercase; str(True) == "True" would
        # be parsed, but we keep the wire format consistent with upstream.
        return "true" if value else "false"
    return str(value)


class TrainerBackend(ABC):
    """Builds the command line for one training library."""

    #: config `backend:` value this class handles
    name: str
    #: executable expected on PATH inside the training image
    executable: str

    @abstractmethod
    def build_argv(
        self,
        cfg: ExperimentConfig,
        *,
        resume: bool = False,
        wandb_available: bool = True,
    ) -> list[str]:
        """Return the full argv for a training run."""


_REGISTRY: dict[str, type[TrainerBackend]] = {}


def register(cls: type[TrainerBackend]) -> type[TrainerBackend]:
    _REGISTRY[cls.name] = cls
    return cls


def get_backend(name: str) -> TrainerBackend:
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown training backend {name!r}. Known: {known}")
    return _REGISTRY[name]()
