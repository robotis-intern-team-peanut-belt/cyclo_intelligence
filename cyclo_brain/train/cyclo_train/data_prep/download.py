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

"""Fetch a dataset from the HuggingFace Hub and bring it to LeRobot v3.0.

Argv construction is kept separate from execution so it can be tested without a
network, a hub token, or the hub CLI.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .dataset import codebase_version, dataset_root
from .errors import DataPrepError

UPSTREAM_CONVERTER = "lerobot.scripts.convert_dataset_v21_to_v30"


@dataclass(frozen=True)
class PreparedDataset:
    root: Path
    version: str
    backup_root: Path | None = None
    converted: bool = False


def build_download_argv(repo_id: str, root: Path) -> list[str]:
    return ["hf", "download", repo_id, "--repo-type", "dataset", "--local-dir", str(root)]


def build_convert_argv(repo_id: str, root: Path) -> list[str]:
    # push_to_hub must be passed explicitly: upstream defaults it to True in both
    # the function signature and the CLI, so omitting it can publish a dataset.
    return [
        "python",
        "-m",
        UPSTREAM_CONVERTER,
        f"--repo-id={repo_id}",
        f"--root={root}",
        "--push-to-hub=false",
    ]


def _run(argv: list[str]) -> None:
    print("[cyclo-train] $ " + " ".join(argv), flush=True)
    result = subprocess.run(argv)
    if result.returncode != 0:
        raise DataPrepError(f"command failed ({result.returncode}): {' '.join(argv)}")


def prepare(
    repo_id: str,
    workspace: Path,
    *,
    convert: bool = True,
    dry_run: bool = False,
) -> PreparedDataset:
    """Download `repo_id` into <workspace>/dataset/<name> and convert if v2.1.

    Idempotent: an existing v3.0 dataset is left alone rather than re-downloaded
    into (and then clobbered by a re-conversion of) the converted tree.
    """
    root = dataset_root(workspace, repo_id)

    if root.exists():
        existing = codebase_version(root)
        if existing == "v3.0":
            print(f"[cyclo-train] Already v3.0, nothing to do: {root}")
            return PreparedDataset(root=root, version="v3.0")
        if existing != "unknown":
            print(f"[cyclo-train] Existing dataset is {existing}; re-downloading into {root}")

    download_argv = build_download_argv(repo_id, root)
    convert_argv = build_convert_argv(repo_id, root)

    if dry_run:
        print("[cyclo-train] would run:")
        print("  " + " ".join(download_argv))
        print("  " + " ".join(convert_argv) + "   # only if the download is v2.1")
        return PreparedDataset(root=root, version="(dry-run)")

    root.mkdir(parents=True, exist_ok=True)
    _run(download_argv)

    version = codebase_version(root)
    print(f"[cyclo-train] Downloaded {repo_id} ({version}) -> {root}")

    if version == "v3.0":
        return PreparedDataset(root=root, version=version)
    if version != "v2.1":
        print(f"[cyclo-train] WARNING: unrecognised codebase_version {version!r}; not converting.")
        return PreparedDataset(root=root, version=version)
    if not convert:
        return PreparedDataset(root=root, version=version)

    _run(convert_argv)

    # The upstream converter works IN PLACE: it moves the original aside to
    # <name>_old and moves the new tree back onto <name>. There is no <name>_v30.
    backup = root.with_name(root.name + "_old")
    return PreparedDataset(
        root=root,
        version=codebase_version(root),
        backup_root=backup if backup.exists() else None,
        converted=True,
    )
