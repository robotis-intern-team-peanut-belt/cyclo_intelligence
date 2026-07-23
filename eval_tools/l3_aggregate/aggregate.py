#!/usr/bin/env python3
"""L3 — aggregate L2 annotation CSVs into interpretable eval statistics.

Reads one or more CSVs in the L2 schema and produces:
  1. Noise floor      — |success-rate diff| across sessions of the SAME policy.
  2. Condition table   — success rate + Wilson 95% CI per
                          (y_position x chunk_size x checkpoint_step); flags n<8.
  3. McNemar test      — paired policy-vs-policy on init_condition_id (inner join);
                          logs unmatched rows; warns if b+c<10 (underpowered).
  4. Checkpoint curve  — step vs success rate + CI band (full curve + last-5 max).
  5. F-code heatmap    — failure_code x condition crosstab.

Tables print to the terminal; figures are written to --outdir as PNGs.

Usage:
    python aggregate.py a.csv b.csv --outdir out
    python aggregate.py *.csv --policy-a policy_A --policy-b policy_B
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUCCESS = "success"
FAILURE = "failure"
Z95 = 1.959963984540054  # z for 95% two-sided
MIN_CELL_N = 8            # Wilson table small-sample flag
MCNEMAR_MIN = 10          # b+c below this -> underpowered

REQUIRED_COLS = [
    "run_id", "session_id", "policy_id", "checkpoint_step", "chunk_size",
    "trial_index", "init_condition_id", "y_position", "result", "failure_code",
]


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
def load_csvs(paths) -> pd.DataFrame:
    frames = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            sys.exit(f"[error] CSV not found: {p}")
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            sys.exit(f"[error] {p} missing columns: {missing}")
        df["__source"] = p.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # normalize
    df["result"] = df["result"].str.strip().str.lower()
    df["success"] = (df["result"] == SUCCESS).astype(int)
    # keep only scored rows (result present)
    scored = df["result"].isin([SUCCESS, FAILURE])
    dropped = (~scored).sum()
    if dropped:
        print(f"[info] dropping {dropped} unscored/blank-result row(s).")
    df = df[scored].copy()

    df["checkpoint_step"] = pd.to_numeric(df["checkpoint_step"], errors="coerce")
    df["chunk_size"] = pd.to_numeric(df["chunk_size"], errors="coerce")
    return df


# ----------------------------------------------------------------------------
# stats helpers
# ----------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = Z95):
    """Wilson score interval. Returns (p_hat, lo, hi)."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def mcnemar_pvalue(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value (binomial on the discordant pairs)."""
    n = b + c
    if n == 0:
        return float("nan")
    try:
        from scipy.stats import binomtest
        return float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
    except Exception:
        # pure-python exact two-sided binomial fallback
        from math import comb
        k = min(b, c)
        tail = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
        return min(1.0, 2 * tail)


# ----------------------------------------------------------------------------
# 1. noise floor
# ----------------------------------------------------------------------------
def noise_floor(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("1) NOISE FLOOR — success-rate spread across sessions of the same policy")
    print("=" * 70)
    rows = []
    per_session = (df.groupby(["policy_id", "session_id"])["success"]
                     .agg(["mean", "count"]).reset_index())
    for policy, g in per_session.groupby("policy_id"):
        g = g.sort_values("session_id")
        if len(g) < 2:
            print(f"  [{policy}] only {len(g)} session — need >=2 to estimate noise floor.")
            continue
        rates = g["mean"].values
        spread = float(rates.max() - rates.min())
        print(f"  [{policy}] {len(g)} sessions:")
        for _, r in g.iterrows():
            print(f"      session={r['session_id']:<12} rate={r['mean']*100:6.1f}%  (n={int(r['count'])})")
        print(f"      -> max abs diff (noise floor) = {spread*100:.1f} pp")
        # pairwise abs diffs
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                rows.append({
                    "policy_id": policy,
                    "session_a": g.iloc[i]["session_id"],
                    "session_b": g.iloc[j]["session_id"],
                    "rate_a": g.iloc[i]["mean"],
                    "rate_b": g.iloc[j]["mean"],
                    "abs_diff": abs(g.iloc[i]["mean"] - g.iloc[j]["mean"]),
                })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 2. condition table with Wilson CI
# ----------------------------------------------------------------------------
def condition_table(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("2) CONDITION SUCCESS RATE + WILSON 95% CI  (y_position x chunk_size x checkpoint_step)")
    print("=" * 70)
    keys = ["y_position", "chunk_size", "checkpoint_step"]
    rows = []
    for vals, g in df.groupby(keys, dropna=False):
        n = len(g)
        k = int(g["success"].sum())
        p, lo, hi = wilson_ci(k, n)
        rows.append({
            "y_position": vals[0], "chunk_size": vals[1], "checkpoint_step": vals[2],
            "n": n, "k_success": k,
            "rate": round(p, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "low_n_flag": "⚠ n<8" if n < MIN_CELL_N else "",
        })
    out = pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(out.to_string(index=False))
    flagged = (out["low_n_flag"] != "").sum()
    if flagged:
        print(f"  [warn] {flagged} cell(s) have n<{MIN_CELL_N} — treat those CIs as unreliable.")
    return out


# ----------------------------------------------------------------------------
# 3. McNemar paired test
# ----------------------------------------------------------------------------
def _collapse_condition(df: pd.DataFrame, policy: str) -> pd.Series:
    """One binary outcome per init_condition_id for a policy (mean>=0.5=success)."""
    g = df[df["policy_id"] == policy]
    by = g.groupby("init_condition_id")["success"]
    counts = by.count()
    multi = counts[counts > 1]
    if len(multi):
        print(f"  [info] policy '{policy}': {len(multi)} condition(s) had >1 trial; "
              f"collapsed by majority (mean>=0.5).")
    return (by.mean() >= 0.5).astype(int)


def mcnemar(df: pd.DataFrame, policy_a: str, policy_b: str, outdir: Path):
    print("\n" + "=" * 70)
    print(f"3) McNEMAR — paired on init_condition_id:  '{policy_a}'  vs  '{policy_b}'")
    print("=" * 70)

    a = _collapse_condition(df, policy_a)
    b_ = _collapse_condition(df, policy_b)
    if a.empty or b_.empty:
        print("  [error] one of the policies has no rows — skipping McNemar.")
        return None

    # inner join (intersection) of init_condition_id
    common = a.index.intersection(b_.index)
    only_a = a.index.difference(b_.index)
    only_b = b_.index.difference(a.index)
    print(f"  matched conditions (inner join): {len(common)}")
    if len(only_a):
        print(f"  [warn] {len(only_a)} condition(s) only in '{policy_a}' (dropped): "
              f"{sorted(only_a)[:10]}{'...' if len(only_a)>10 else ''}")
    if len(only_b):
        print(f"  [warn] {len(only_b)} condition(s) only in '{policy_b}' (dropped): "
              f"{sorted(only_b)[:10]}{'...' if len(only_b)>10 else ''}")
    if len(common) == 0:
        print("  [error] no shared init_condition_id — cannot run McNemar.")
        return None

    aa = a.loc[common]
    bb = b_.loc[common]
    # b = A success & B fail ; c = A fail & B success
    b_count = int(((aa == 1) & (bb == 0)).sum())
    c_count = int(((aa == 0) & (bb == 1)).sum())
    both = int(((aa == 1) & (bb == 1)).sum())
    neither = int(((aa == 0) & (bb == 0)).sum())
    p = mcnemar_pvalue(b_count, c_count)

    print(f"\n  contingency (paired, n={len(common)}):")
    print(f"                         {policy_b}=success   {policy_b}=fail")
    print(f"    {policy_a}=success        {both:>6}          {b_count:>6}")
    print(f"    {policy_a}=fail           {c_count:>6}          {neither:>6}")
    print(f"\n  b (A>B) = {b_count}   c (B>A) = {c_count}   b+c = {b_count+c_count}")
    print(f"  exact two-sided p-value = {p:.4g}")
    if b_count + c_count < MCNEMAR_MIN:
        print(f"  [warn] b+c={b_count+c_count} < {MCNEMAR_MIN} — UNDERPOWERED, interpret with caution.")

    return {"policy_a": policy_a, "policy_b": policy_b, "n_matched": len(common),
            "b": b_count, "c": c_count, "both": both, "neither": neither, "pvalue": p}


# ----------------------------------------------------------------------------
# 4. checkpoint curve
# ----------------------------------------------------------------------------
def checkpoint_curve(df: pd.DataFrame, outdir: Path):
    print("\n" + "=" * 70)
    print("4) CHECKPOINT CURVE — step vs success rate (+95% CI band)")
    print("=" * 70)
    sub = df.dropna(subset=["checkpoint_step"])
    if sub.empty:
        print("  [skip] no numeric checkpoint_step values.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for policy, g in sub.groupby("policy_id"):
        per = (g.groupby("checkpoint_step")["success"]
                 .agg(["sum", "count"]).reset_index()
                 .sort_values("checkpoint_step"))
        ci = [wilson_ci(int(r["sum"]), int(r["count"])) for _, r in per.iterrows()]
        steps = per["checkpoint_step"].values
        rate = np.array([c[0] for c in ci])
        lo = np.array([c[1] for c in ci])
        hi = np.array([c[2] for c in ci])
        line, = ax.plot(steps, rate, marker="o", label=str(policy))
        ax.fill_between(steps, lo, hi, alpha=0.18, color=line.get_color())

        last5 = per.tail(5)
        l5_rate = (last5["sum"] / last5["count"])
        best_idx = l5_rate.idxmax()
        full_best_idx = (per["sum"] / per["count"]).idxmax()
        print(f"  [{policy}] full-curve best: step={int(per.loc[full_best_idx,'checkpoint_step'])} "
              f"rate={100*(per.loc[full_best_idx,'sum']/per.loc[full_best_idx,'count']):.1f}%  |  "
              f"last-5 max: step={int(per.loc[best_idx,'checkpoint_step'])} "
              f"rate={100*l5_rate.loc[best_idx]:.1f}%")
    ax.set_xlabel("checkpoint_step"); ax.set_ylabel("success rate")
    ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("Checkpoint success rate (Wilson 95% CI)")
    path = outdir / "checkpoint_curve.png"
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  -> saved {path}")


# ----------------------------------------------------------------------------
# 5. F-code x condition heatmap
# ----------------------------------------------------------------------------
def fcode_heatmap(df: pd.DataFrame, dim: str, outdir: Path):
    print("\n" + "=" * 70)
    print(f"5) FAILURE-CODE x {dim} CROSSTAB (failures only)")
    print("=" * 70)
    fails = df[(df["result"] == FAILURE) & (df["failure_code"] != "")]
    if fails.empty:
        print("  [skip] no failure rows with a failure_code.")
        return
    ct = pd.crosstab(fails["failure_code"], fails[dim].astype(str))
    ct = ct.sort_index()
    print(ct.to_string())

    fig, ax = plt.subplots(figsize=(1.4 + 1.1 * ct.shape[1], 1 + 0.5 * ct.shape[0]))
    im = ax.imshow(ct.values, cmap="magma", aspect="auto")
    ax.set_xticks(range(ct.shape[1])); ax.set_xticklabels(ct.columns, rotation=30, ha="right")
    ax.set_yticks(range(ct.shape[0])); ax.set_yticklabels(ct.index)
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = ct.values[i, j]
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v < ct.values.max() * 0.6 else "black", fontsize=10)
    ax.set_xlabel(dim); ax.set_ylabel("failure_code")
    ax.set_title(f"Failure code x {dim}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path = outdir / f"fcode_by_{dim}.png"
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  -> saved {path}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="L3 eval aggregation")
    ap.add_argument("csvs", nargs="+", help="one or more L2 annotation CSVs")
    ap.add_argument("--outdir", default="l3_out", help="figure output dir")
    ap.add_argument("--policy-a", default=None, help="McNemar policy A (default: auto)")
    ap.add_argument("--policy-b", default=None, help="McNemar policy B (default: auto)")
    ap.add_argument("--heatmap-dim", default="y_position",
                    choices=["y_position", "chunk_size", "checkpoint_step",
                             "init_condition_id", "policy_id"],
                    help="condition axis for the F-code crosstab")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = load_csvs(args.csvs)
    print(f"[info] loaded {len(df)} scored rows across policies: "
          f"{sorted(df['policy_id'].unique())}")

    noise_floor(df)
    condition_table(df)

    # pick policies for McNemar
    policies = sorted(df["policy_id"].unique())
    pa, pb = args.policy_a, args.policy_b
    if pa is None or pb is None:
        if len(policies) == 2:
            pa, pb = policies
        else:
            print("\n[3) McNemar] need exactly 2 policies or --policy-a/--policy-b; "
                  f"found {len(policies)}: {policies} — skipping.")
            pa = pb = None
    if pa and pb:
        mcnemar(df, pa, pb, outdir)

    checkpoint_curve(df, outdir)
    fcode_heatmap(df, args.heatmap_dim, outdir)
    print(f"\n[done] figures in {outdir.resolve()}")


if __name__ == "__main__":
    main()
