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

"""cyclo-train command line.

Invoked through `docker/container.sh train-lerobot`, which starts the training
container and injects git provenance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config as config_mod
from . import promote as promote_mod
from . import runner
from .analysis.loss import LossAnalysisError
from .config import ConfigError
from .data_prep.errors import DataPrepError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cyclo-train",
        description="Config-driven policy training for Cyclo Intelligence.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_experiment(p: argparse.ArgumentParser) -> None:
        p.add_argument("experiment", help="experiment name (experiments/<name>.yaml) or a path")

    def add_set(p: argparse.ArgumentParser, help_text: str) -> None:
        p.add_argument(
            "--set", dest="overrides", action="append", default=[], metavar="K=V", help=help_text
        )

    p_validate = sub.add_parser("validate", help="resolve and check an experiment")
    add_experiment(p_validate)
    add_set(p_validate, "override a config key")

    p_run = sub.add_parser("run", help="start a training run")
    add_experiment(p_run)
    add_set(
        p_run,
        "override any config key, e.g. --set train.steps=100 --set name=probe. "
        "Quote exponents: --set policy.optimizer_lr=1.0e-5",
    )
    p_run.add_argument("--dry-run", action="store_true", help="print the trainer argv and exit")
    p_run.add_argument(
        "--strict-tracking",
        action="store_true",
        help="fail if tracking.backend is unavailable instead of degrading",
    )

    p_resume = sub.add_parser("resume", help="continue a run from its last checkpoint")
    add_experiment(p_resume)
    add_set(p_resume, "only --set name=<run> is accepted when resuming")
    p_resume.add_argument("--dry-run", action="store_true")

    sub.add_parser("list", help="list runs in the workspace")

    p_download = sub.add_parser(
        "download", help="fetch a dataset from the HuggingFace Hub (converts v2.1 -> v3.0)"
    )
    p_download.add_argument("repo_id", help="e.g. RobotisSW/Task_900004_..._lerobot")
    p_download.add_argument("--no-convert", action="store_true", help="download only")
    p_download.add_argument("--dry-run", action="store_true")

    p_convert = sub.add_parser(
        "convert", help="convert a local v2.1 dataset to v3.0 (writes <name>_v30)"
    )
    p_convert.add_argument("dataset", help="dataset name under workspace/dataset, or a path")
    p_convert.add_argument("--no-mobile", action="store_true", help="also drop the 3 mobile DOFs")
    p_convert.add_argument("--force", action="store_true", help="replace an existing output")

    p_trim = sub.add_parser(
        "trim-mobile", help="drop the 3 mobile DOFs from a v3.0 dataset (writes <name>_no_mobile)"
    )
    p_trim.add_argument("dataset", help="dataset name under workspace/dataset, or a path")
    p_trim.add_argument("--force", action="store_true", help="replace an existing output")

    p_loss = sub.add_parser("analyze-loss", help="detect a loss plateau in a run's log")
    p_loss.add_argument("run", help="run name (see `list`), or a path to a log file")
    p_loss.add_argument("--interval", type=int, default=10_000, help="step interval (default 10000)")
    p_loss.add_argument("--threshold", type=float, default=0.05)
    p_loss.add_argument("--consecutive", type=int, default=2)
    p_loss.add_argument("--metric", choices=("relative", "absolute"), default="relative")
    p_loss.add_argument("--csv", default=None, help="also write the table to this path")

    p_promote = sub.add_parser("promote", help="publish a checkpoint to the inference dropbox")
    p_promote.add_argument("name", help="run name (see `list`)")
    p_promote.add_argument("--step", default="last", help="checkpoint step, or 'last' (default)")
    p_promote.add_argument("--dest-name", default=None, help="publish under a different name")
    p_promote.add_argument("--backend", default="lerobot")
    p_promote.add_argument("--force", action="store_true", help="overwrite an existing promotion")

    return parser


def _cmd_validate(args, _extra) -> int:
    import yaml

    cfg = config_mod.load(args.experiment, args.overrides)
    print(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    print(f"# config hash : {cfg.hash()}")
    print(f"# run dir     : {cfg.run_dir}")
    print(f"# dropbox     : {cfg.dropbox_dir}")
    print("OK", file=sys.stderr)
    return 0


def _cmd_run(args, extra) -> int:
    cfg = config_mod.load(args.experiment, args.overrides)
    return runner.run(
        cfg,
        dry_run=args.dry_run,
        strict_tracking=args.strict_tracking,
        extra_args=extra,
    )


def _cmd_resume(args, extra) -> int:
    # On resume lerobot rebuilds the config from the checkpoint's
    # train_config.json, so any other override would be silently discarded while
    # still being written into resolved_config.yaml and the manifest.
    bad = [o for o in args.overrides if not o.startswith("name=")]
    if bad:
        raise ConfigError(
            "resume takes its config from the checkpoint, so these would be ignored: "
            + ", ".join(bad)
            + "\nOnly --set name=<run> is accepted. To change hyperparameters, start a new "
            "experiment with policy.pretrained_path pointing at the checkpoint."
        )
    if extra:
        raise ConfigError(f"resume does not forward extra trainer flags: {' '.join(extra)}")
    cfg = config_mod.load(args.experiment, args.overrides)
    return runner.run(cfg, resume=True, dry_run=args.dry_run)


def _cmd_list(_args, _extra) -> int:
    rows = promote_mod.list_runs()
    if not rows:
        print("No runs yet.")
        return 0
    width = max(len(r["name"]) for r in rows + [{"name": "RUN"}])
    print(f"{'RUN':<{width}}  {'LAST':>8}  {'CKPTS':>5}  {'RESUMED':>7}  PUBLISHED AS")
    for row in rows:
        print(
            f"{row['name']:<{width}}  {str(row['last'] or '-'):>8}  {len(row['steps']):>5}  "
            f"{row['resumes']:>7}  {row['promoted'] or '-'}"
        )
    return 0


def _cmd_promote(args, _extra) -> int:
    dest = promote_mod.promote(
        args.name,
        step=args.step,
        dest_name=args.dest_name,
        backend=args.backend,
        force=args.force,
    )
    print(f"[cyclo-train] Promoted {args.name} -> {dest}")
    print("[cyclo-train] The inference engine can now LOAD this model.")
    return 0


def _dataset_path(ref: str) -> Path:
    """Accept a dataset name under workspace/dataset, or a path."""
    candidate = Path(ref)
    if candidate.is_dir():
        return candidate.resolve()
    return config_mod.WORKSPACE / "dataset" / ref


def _cmd_download(args, _extra) -> int:
    from .data_prep import download as download_mod

    result = download_mod.prepare(
        args.repo_id,
        config_mod.WORKSPACE,
        convert=not args.no_convert,
        dry_run=args.dry_run,
    )
    print(f"[cyclo-train] dataset: {result.root}  ({result.version})")
    if result.backup_root:
        print(f"[cyclo-train] v2.1 backup kept at: {result.backup_root}")
    print(f"[cyclo-train] use this in an experiment:\n  dataset:\n    root: {result.root}")
    return 0


def _cmd_convert(args, _extra) -> int:
    from .data_prep import convert_v21

    src = _dataset_path(args.dataset)
    dst = src.with_name(src.name + "_v30")
    result = convert_v21.convert(src, dst, no_mobile=args.no_mobile, force=args.force)
    print(
        f"[cyclo-train] converted -> {result.dst}\n"
        f"[cyclo-train] episodes={result.episodes} frames={result.frames} dims={result.kept_dims}"
    )
    return 0


def _cmd_trim_mobile(args, _extra) -> int:
    from .data_prep import no_mobile

    src = _dataset_path(args.dataset)
    dst = src.with_name(src.name + "_no_mobile")
    result = no_mobile.trim_mobile(src, dst, force=args.force)
    print(
        f"[cyclo-train] trimmed {result.original_dims} -> {result.kept_dims} dims: {result.dst}\n"
        f"[cyclo-train] per-episode stats rewritten: {result.episodes_meta_trimmed}"
    )
    return 0


def _cmd_analyze_loss(args, _extra) -> int:
    from .analysis import loss as loss_mod

    path = Path(args.run)
    if not path.is_file():
        path = config_mod.WORKSPACE / "runs" / args.run / "train.log"
    points = loss_mod.read_loss_points(path)
    selected = loss_mod.select_interval_points(points, args.interval)
    analysis = loss_mod.analyze(
        selected,
        threshold=args.threshold,
        consecutive=args.consecutive,
        metric=args.metric,
    )
    print(loss_mod.format_report(analysis))
    if args.csv:
        loss_mod.write_csv(Path(args.csv), analysis)
        print(f"\nCSV written: {args.csv}")
    return 0


_COMMANDS = {
    "validate": _cmd_validate,
    "run": _cmd_run,
    "resume": _cmd_resume,
    "list": _cmd_list,
    "promote": _cmd_promote,
    "download": _cmd_download,
    "convert": _cmd_convert,
    "trim-mobile": _cmd_trim_mobile,
    "analyze-loss": _cmd_analyze_loss,
}


def main(argv: list[str] | None = None) -> int:
    # parse_known_args, not a trailing positional: argparse only keeps a `*`
    # positional when `--` follows the experiment immediately, so any flag before
    # it (--set, --dry-run) would make the whole invocation fail.
    args, extra = build_parser().parse_known_args(argv)
    try:
        return _COMMANDS[args.command](args, extra)
    except (ConfigError, runner.RunError, promote_mod.PromoteError, DataPrepError,
            LossAnalysisError) as exc:
        print(f"[cyclo-train] error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
