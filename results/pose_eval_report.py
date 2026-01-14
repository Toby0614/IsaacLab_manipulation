#!/usr/bin/env python3
"""
Pose corruption evaluation report generator (poe3.pdf).

Consumes pose-eval outputs in a directory (default: results/pose_eval_final):
  - *_pose_all_summaries.json (preferred)
  - or *_pose_*_variant{1,2}_summary.json (fallback)

Produces figures and tables:
  - Success heatmaps (condition x duration) per (policy, mode, variant)
  - AUC (robustness score) per (policy, mode, variant)
  - L50 per (policy, mode, variant) computed from success-vs-duration curves

Notes on L50:
  - poe3.pdf defines L50 as the duration where success drops to 50% of CLEAN success.
  - Many of your eval sweeps do NOT include a "clean" (duration=0) point.
  - This script therefore defaults to using the smallest evaluated duration as a proxy baseline
    (configurable via --baseline_duration, or using max success via --baseline_kind max).
  - It also reports "censored" cases where success never crosses 50% within the tested durations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _try_import_plotting():
    # Keep imports local so users without plotting deps can still compute tables.
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt  # noqa: WPS433

    try:
        import seaborn as sns  # type: ignore  # noqa: WPS433
    except Exception:
        sns = None
    return plt, sns


def _safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s).strip("_")


def _read_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


@dataclass(frozen=True)
class Summary:
    base_policy: str  # e.g., "M2_policy"
    mode: str  # hard/freeze/noise/delay/mixed
    variant: int  # 1 or 2
    conditions: list[str]  # phases or onset steps as strings
    durations: list[int]
    success_matrix: list[list[float]]  # rows=conditions, cols=durations
    source_file: str


def _parse_policy_and_mode(policy_name: str) -> tuple[str, str]:
    """
    Expects policy_name like:
      - M2_policy_pose_delay
      - M4_pose_pose_noise
    """
    m = re.match(r"^(?P<base>.+)_pose_(?P<mode>[^_]+)$", policy_name)
    if not m:
        # fallback: assume last token is mode, base is the rest
        parts = policy_name.split("_")
        if len(parts) >= 2:
            return "_".join(parts[:-1]), parts[-1]
        return policy_name, "unknown"
    return m.group("base"), m.group("mode")


def _load_summaries_from_all_summaries(path: Path) -> list[Summary]:
    raw = _read_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"Expected list in {path}, got: {type(raw)}")
    out: list[Summary] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        policy_name = str(item.get("policy_name", ""))
        base_policy, mode = _parse_policy_and_mode(policy_name)
        variant = int(item.get("variant"))
        conditions = [str(x) for x in item.get("conditions", [])]
        durations = [int(x) for x in item.get("durations", [])]
        success_matrix = item.get("success_matrix", [])
        out.append(
            Summary(
                base_policy=base_policy,
                mode=mode,
                variant=variant,
                conditions=conditions,
                durations=durations,
                success_matrix=success_matrix,
                source_file=str(path),
            )
        )
    return out


def _load_summaries_from_summary_jsons(input_dir: Path) -> list[Summary]:
    out: list[Summary] = []
    for p in sorted(input_dir.glob("*_pose_*_variant*_summary.json")):
        item = _read_json(p)
        if not isinstance(item, dict):
            continue
        policy_name = str(item.get("policy_name", ""))
        base_policy, mode = _parse_policy_and_mode(policy_name)
        variant = int(item.get("variant"))
        conditions = [str(x) for x in item.get("conditions", [])]
        durations = [int(x) for x in item.get("durations", [])]
        success_matrix = item.get("success_matrix", [])
        out.append(
            Summary(
                base_policy=base_policy,
                mode=mode,
                variant=variant,
                conditions=conditions,
                durations=durations,
                success_matrix=success_matrix,
                source_file=str(p),
            )
        )
    return out


def load_summaries(input_dir: Path) -> list[Summary]:
    all_sum = sorted(input_dir.glob("*_pose_all_summaries.json"))
    if all_sum:
        summaries: list[Summary] = []
        for p in all_sum:
            summaries.extend(_load_summaries_from_all_summaries(p))
        return summaries
    return _load_summaries_from_summary_jsons(input_dir)


def iter_rows(summaries: Iterable[Summary]) -> Iterable[dict[str, Any]]:
    for s in summaries:
        if len(s.success_matrix) != len(s.conditions):
            raise ValueError(
                f"Bad matrix shape in {s.source_file}: "
                f"rows={len(s.success_matrix)} vs conditions={len(s.conditions)}"
            )
        for i, cond in enumerate(s.conditions):
            row = s.success_matrix[i]
            if len(row) != len(s.durations):
                raise ValueError(
                    f"Bad matrix shape in {s.source_file}: "
                    f"cols={len(row)} vs durations={len(s.durations)}"
                )
            for j, dur in enumerate(s.durations):
                yield {
                    "policy": s.base_policy,
                    "mode": s.mode,
                    "variant": s.variant,
                    "condition": cond,
                    "duration": int(dur),
                    "success": float(row[j]),
                }


def compute_auc(durations: np.ndarray, success: np.ndarray) -> float:
    # Robustness score: normalized integral of success over duration.
    if durations.size < 2:
        return float(np.nan)
    x = durations.astype(np.float64)
    y = success.astype(np.float64)
    # numpy compatibility: np.trapezoid is newer; np.trapz exists in older versions.
    try:
        area = float(np.trapezoid(y=y, x=x))  # type: ignore[attr-defined]
    except Exception:
        area = float(np.trapz(y=y, x=x))
    denom = float(x[-1] - x[0])
    if denom <= 0:
        return float(np.nan)
    return float(area / denom)


@dataclass(frozen=True)
class L50Result:
    l50: float | None
    censored: bool
    baseline_success: float
    threshold: float


def compute_l50(
    durations: np.ndarray,
    success: np.ndarray,
    *,
    baseline_kind: str,
    baseline_duration: int | None,
) -> L50Result:
    """
    Find the duration where success falls to <= 50% of baseline.
    Linear interpolation between grid points when a crossing occurs.
    """
    x = durations.astype(np.float64)
    y = success.astype(np.float64)
    if x.size == 0:
        return L50Result(l50=None, censored=True, baseline_success=float("nan"), threshold=float("nan"))

    if baseline_kind == "min_duration":
        if baseline_duration is not None and baseline_duration in set(int(v) for v in x.tolist()):
            baseline = float(y[np.where(x == float(baseline_duration))[0][0]])
        else:
            baseline = float(y[0])
    elif baseline_kind == "max":
        baseline = float(np.nanmax(y))
    else:
        raise ValueError(f"Unknown baseline_kind: {baseline_kind}")

    thr = 0.5 * baseline

    # If already below threshold at smallest duration, L50 is that duration.
    if y[0] <= thr:
        return L50Result(l50=float(x[0]), censored=False, baseline_success=baseline, threshold=thr)

    # Walk forward until we cross.
    for i in range(1, x.size):
        if y[i] <= thr:
            # interpolate between (x[i-1], y[i-1]) and (x[i], y[i])
            x0, y0 = float(x[i - 1]), float(y[i - 1])
            x1, y1 = float(x[i]), float(y[i])
            if y1 == y0:
                return L50Result(l50=float(x1), censored=False, baseline_success=baseline, threshold=thr)
            t = (thr - y0) / (y1 - y0)  # y0 > thr, y1 <= thr => t in (0,1]
            l50 = x0 + t * (x1 - x0)
            return L50Result(l50=float(l50), censored=False, baseline_success=baseline, threshold=thr)

    # Never crossed within tested range.
    return L50Result(l50=None, censored=True, baseline_success=baseline, threshold=thr)


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _pivot_matrix(rows: list[dict[str, Any]], conditions: list[str], durations: list[int]) -> np.ndarray:
    # rows must include exactly one (condition,duration) each
    cond_to_i = {c: i for i, c in enumerate(conditions)}
    dur_to_j = {int(d): j for j, d in enumerate(durations)}
    mat = np.full((len(conditions), len(durations)), np.nan, dtype=np.float64)
    for r in rows:
        i = cond_to_i[str(r["condition"])]
        j = dur_to_j[int(r["duration"])]
        mat[i, j] = float(r["success"])
    return mat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir",
        type=str,
        default="results/pose_eval_final",
        help="Directory with pose-eval json/csv outputs (default: results/pose_eval_final).",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default="results/pose_eval_report",
        help="Output directory for figures/tables.",
    )
    ap.add_argument(
        "--baseline_kind",
        type=str,
        default="min_duration",
        choices=["min_duration", "max"],
        help="How to choose baseline success for L50 when clean (duration=0) is unavailable.",
    )
    ap.add_argument(
        "--baseline_duration",
        type=int,
        default=None,
        help="If provided and present, use this duration as baseline success (only for baseline_kind=min_duration).",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure DPI.",
    )
    ap.add_argument(
        "--no_plots",
        action="store_true",
        default=False,
        help="Only compute tables (no matplotlib/seaborn required).",
    )

    args = ap.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = load_summaries(input_dir)
    if not summaries:
        raise SystemExit(f"No pose-eval summaries found in: {input_dir}")

    # Collect long rows (no pandas dependency).
    long_rows = list(iter_rows(summaries))

    # Index unique groups.
    policies = sorted({r["policy"] for r in long_rows})
    modes = sorted({r["mode"] for r in long_rows})
    variants = sorted({int(r["variant"]) for r in long_rows})

    # Compute per-(policy,mode,variant,condition) L50 and AUC.
    l50_rows: list[list[Any]] = []
    auc_rows: list[list[Any]] = []

    # For plots, we create per-summary heatmaps, plus aggregate bar charts per mode/variant.
    if not args.no_plots:
        plt, sns = _try_import_plotting()
    else:
        plt, sns = None, None

    # Helper: filter rows for a given group.
    def _select(policy: str, mode: str, variant: int) -> list[dict[str, Any]]:
        return [r for r in long_rows if r["policy"] == policy and r["mode"] == mode and int(r["variant"]) == int(variant)]

    # We want "conditions/durations" from the Summary object to preserve ordering.
    # Build a lookup for each (policy,mode,variant) to its canonical ordering.
    ordering: dict[tuple[str, str, int], tuple[list[str], list[int]]] = {}
    for s in summaries:
        key = (s.base_policy, s.mode, int(s.variant))
        if key not in ordering:
            ordering[key] = (list(s.conditions), list(s.durations))

    # Compute metrics and per-group plots.
    for policy in policies:
        for mode in modes:
            for variant in variants:
                key = (policy, mode, int(variant))
                if key not in ordering:
                    continue
                conds, durs = ordering[key]
                rows_g = _select(policy, mode, variant)
                if not rows_g:
                    continue

                # Heatmap of success itself.
                mat = _pivot_matrix(rows_g, conds, durs)
                if plt is not None:
                    fig, ax = plt.subplots(figsize=(0.9 * len(durs) + 2.2, 0.5 * len(conds) + 2.0))
                    title = f"{policy} | mode={mode} | variant={variant}"
                    if sns is not None:
                        sns.heatmap(
                            mat,
                            ax=ax,
                            vmin=0.0,
                            vmax=1.0,
                            cmap="viridis",
                            annot=False,
                            cbar=True,
                            xticklabels=[str(d) for d in durs],
                            yticklabels=conds,
                        )
                    else:
                        im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
                        fig.colorbar(im, ax=ax)
                        ax.set_xticks(range(len(durs)))
                        ax.set_xticklabels([str(d) for d in durs])
                        ax.set_yticks(range(len(conds)))
                        ax.set_yticklabels(conds)
                    ax.set_title(title)
                    ax.set_xlabel("duration (steps)")
                    ax.set_ylabel("condition")
                    fig.tight_layout()
                    fig_path = (
                        output_dir
                        / "heatmaps"
                        / _safe_name(mode)
                        / f"heatmap_success_{_safe_name(policy)}_{mode}_v{variant}.png"
                    )
                    fig_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(fig_path, dpi=int(args.dpi))
                    plt.close(fig)

                # Per-condition metrics.
                for cond in conds:
                    xs = np.array([r["duration"] for r in rows_g if r["condition"] == cond], dtype=np.float64)
                    ys = np.array([r["success"] for r in rows_g if r["condition"] == cond], dtype=np.float64)
                    # Ensure sorted by duration
                    order_idx = np.argsort(xs)
                    xs = xs[order_idx]
                    ys = ys[order_idx]

                    auc = compute_auc(xs, ys)
                    auc_rows.append([policy, mode, variant, cond, auc])

                    l50r = compute_l50(
                        xs,
                        ys,
                        baseline_kind=str(args.baseline_kind),
                        baseline_duration=int(args.baseline_duration) if args.baseline_duration is not None else None,
                    )
                    max_dur = float(xs[-1]) if xs.size > 0 else float("nan")
                    # For aggregation/plots, it's often better to treat "never crossed within tested range"
                    # as a right-censored value at the max tested duration, instead of dropping it.
                    l50_censored_as_max = float(l50r.l50) if l50r.l50 is not None else float(max_dur)
                    l50_rows.append(
                        [
                            policy,
                            mode,
                            variant,
                            cond,
                            l50r.l50 if l50r.l50 is not None else "",
                            int(l50r.censored),
                            l50r.baseline_success,
                            l50r.threshold,
                            l50_censored_as_max,
                            max_dur,
                        ]
                    )

    # Write tables.
    _write_csv(
        output_dir / "tables" / "auc_per_condition.csv",
        ["policy", "mode", "variant", "condition", "auc"],
        auc_rows,
    )
    _write_csv(
        output_dir / "tables" / "l50_per_condition.csv",
        [
            "policy",
            "mode",
            "variant",
            "condition",
            "l50",
            "censored",
            "baseline_success",
            "threshold_50pct",
            "l50_censored_as_max",
            "max_duration_tested",
        ],
        l50_rows,
    )

    # Aggregate tables per (policy,mode,variant)
    def _group_mean(rows: list[list[Any]], idx_key: list[int], idx_val: int) -> dict[tuple, float]:
        buckets: dict[tuple, list[float]] = {}
        for r in rows:
            key = tuple(r[i] for i in idx_key)
            v = r[idx_val]
            try:
                fv = float(v)
            except Exception:
                continue
            if math.isnan(fv):
                continue
            buckets.setdefault(key, []).append(fv)
        return {k: float(np.mean(vs)) if vs else float("nan") for k, vs in buckets.items()}

    auc_mean = _group_mean(auc_rows, [0, 1, 2], 4)
    l50_mean_observed = _group_mean([r for r in l50_rows if r[4] != ""], [0, 1, 2], 4)
    l50_mean_censored_as_max = _group_mean(l50_rows, [0, 1, 2], 8)

    agg_rows: list[list[Any]] = []
    for policy in policies:
        for mode in modes:
            for variant in variants:
                key = (policy, mode, variant)
                if key in auc_mean or key in l50_mean_observed or key in l50_mean_censored_as_max:
                    agg_rows.append(
                        [
                            policy,
                            mode,
                            variant,
                            auc_mean.get(key, ""),
                            l50_mean_observed.get(key, ""),
                            l50_mean_censored_as_max.get(key, ""),
                        ]
                    )

    _write_csv(
        output_dir / "tables" / "auc_l50_mean_by_group.csv",
        [
            "policy",
            "mode",
            "variant",
            "auc_mean_over_conditions",
            "l50_mean_observed_only",
            "l50_mean_censored_as_max",
        ],
        agg_rows,
    )

    # Aggregate plots: bars for AUC and L50 (mean over conditions) per mode/variant.
    if plt is not None:
        # Build helper arrays
        def _plot_bar(metric: str, values: dict[tuple, float], out_name: str) -> None:
            for variant in variants:
                for mode in modes:
                    xs = []
                    ys = []
                    for policy in policies:
                        key = (policy, mode, variant)
                        if key not in values:
                            continue
                        xs.append(policy)
                        ys.append(values[key])
                    if not xs:
                        continue
                    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(xs)), 4.2))
                    ax.bar(xs, ys, color="steelblue")
                    ax.set_title(f"{metric} (mean over conditions) | mode={mode} | variant={variant}")
                    ax.set_ylabel(metric)
                    ax.set_xlabel("policy")
                    ax.set_ylim(0.0, 1.05 if metric.lower().startswith("auc") else None)
                    ax.grid(axis="y", alpha=0.3)
                    fig.tight_layout()
                    p = output_dir / "summary" / _safe_name(mode) / f"{out_name}_{mode}_v{variant}.png"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(p, dpi=int(args.dpi))
                    plt.close(fig)

        _plot_bar("AUC", auc_mean, "bar_auc_mean")
        _plot_bar("L50 (observed only)", l50_mean_observed, "bar_l50_mean_observed_only")
        _plot_bar("L50 (censored->max)", l50_mean_censored_as_max, "bar_l50_mean_censored_as_max")

    # Write a small run manifest.
    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "policies": policies,
        "modes": modes,
        "variants": variants,
        "baseline_kind": args.baseline_kind,
        "baseline_duration": args.baseline_duration,
        "notes": "If L50 looks surprising, confirm baseline choice and whether duration=0 (clean) is available.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[OK] Wrote report to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


