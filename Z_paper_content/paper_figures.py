"""Paper figures: P / Delta T invariance, ablation bars, dataset stability.

python -m Z_paper_content.paper_figures p   <p_sweep.json>   <out.pdf>
python -m Z_paper_content.paper_figures t   <t_sweep.json>   <out.pdf>
python -m Z_paper_content.paper_figures ablation <out.pdf> <metric s2t|s2s> LABEL=results.json ...
python -m Z_paper_content.paper_figures dataset <npz_dir> <name> <out.pdf> [--xmax X]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter

from Z_paper_content import paper_style as ps


def _sweep(path: str) -> tuple[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    data = json.loads(Path(path).read_text())
    series = {}
    for label, s in data["series"].items():
        xs = np.array([r["value"] for r in s["rows"]], dtype=float)
        vals = np.array([r["s2t"] for r in s["rows"]], dtype=float) * 100.0
        series[label] = (xs, vals.mean(axis=1), vals.std(axis=1, ddof=1) if vals.shape[1] > 1 else np.zeros(len(xs)))
    return data["kind"], series


def plot_sweep(path: str, out: str) -> None:
    kind, series = _sweep(path)
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(ps.HALF_WIDTH, 2.0))
    for label, (xs, m, sd) in series.items():
        color, marker = ps.model_style(label)
        ax.fill_between(xs, m - 2 * sd, m + 2 * sd, color=color, alpha=0.15, linewidth=0)
        ax.plot(xs, m, color=color, marker=marker, markeredgecolor=ps.SURFACE, markeredgewidth=0.6, label=label)
    if kind == "t":
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"Inference $\Delta T$ (fs)")
        ax.axvline(3000, color=ps.INK_MUTED, linewidth=0.6, linestyle=(0, (2, 2)), zorder=0)
        ax.annotate("train", (3000, 1.0), xycoords=("data", "axes fraction"), xytext=(-2, -2), textcoords="offset points", ha="right", va="top", color=ps.INK_SECONDARY, fontsize=7)
    else:
        ax.set_xlabel(r"Inference $P$")
        ax.set_xticks(sorted({int(x) for s in series.values() for x in s[0]}))
        ax.axvline(8, color=ps.INK_MUTED, linewidth=0.6, linestyle=(0, (2, 2)), zorder=0)
        ax.annotate("train", (8, 1.0), xycoords=("data", "axes fraction"), xytext=(2, -2), textcoords="offset points", ha="left", va="top", color=ps.INK_SECONDARY, fontsize=7)
        lo = min((m - 2 * sd).min() for _, m, sd in series.values())
        ax.set_ylim(bottom=max(0.0, lo * 0.8))
    ax.set_ylabel(r"S2T MSE ($\times 10^{-2}$)")
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.legend(loc="upper left", ncol=len(series), bbox_to_anchor=(0.0, 1.16), columnspacing=1.0, handletextpad=0.4)
    ps.save(fig, out)


def plot_ablation(out: str, metric: str, specs: list[str]) -> None:
    key = f"{metric}_test_loss"
    rows = []
    for spec in specs:
        label, path = spec.split("=", 1)
        runs = json.loads(Path(path).read_text())["single_run_results"]
        v = np.array([r[key] for r in runs]) * 100.0
        rows.append((label, v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0))
    rows.sort(key=lambda r: r[1])
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(ps.HALF_WIDTH, 0.28 * len(rows) + 0.45))
    y = np.arange(len(rows))
    colors = [ps.BLUE if r[0] == "ATOM" else "#a9b6c6" for r in rows]
    ax.barh(y, [r[1] for r in rows], height=0.62, color=colors, edgecolor=ps.SURFACE, linewidth=1.0)
    ax.errorbar([r[1] for r in rows], y, xerr=[2 * r[2] for r in rows], fmt="none", ecolor=ps.INK_SECONDARY, elinewidth=0.7, capsize=1.8, capthick=0.7)
    xmax = max(r[1] + 2 * r[2] for r in rows)
    for yi, (label, m, sd) in zip(y, rows):
        ax.annotate(f"{m:.2f}", (m + 2 * sd, yi), xytext=(3, 0), textcoords="offset points", va="center", ha="left", fontsize=7, color=ps.INK_SECONDARY)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, xmax * 1.15)
    ax.set_xlabel(rf"{metric.upper()} MSE ($\times 10^{{-2}}$)")
    ax.invert_yaxis()
    ps.save(fig, out)


def plot_dataset(npz_dir: str, name: str, out: str, xmax: float | None = None) -> None:
    pos_var, step_var, mols = [], [], []
    for f in sorted(Path(npz_dir).glob("*.npz")):
        d = np.load(f)
        arr = d["coords"] if "coords" in d.files else d["R"]
        if arr.ndim != 3 or arr.shape[2] != 3:
            continue
        mols.append(f.stem.split("_")[-1])
        pos_var.append(float(np.var(arr)))
        norms = np.linalg.norm(np.diff(arr, axis=0), axis=2)
        sv = float(np.mean(np.var(norms, axis=0)))
        if name in ("md17", "rmd17"):
            sv *= 2.0  # 0.5 fs MD17 step vs 1 fs TG80 step
        step_var.append(sv)
    x, yv = np.array(pos_var), np.array(step_var)
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(ps.HALF_WIDTH, 2.1))
    ax.scatter(x, yv, s=18, color=ps.BLUE, edgecolor=ps.SURFACE, linewidth=0.6, zorder=3)
    ax.set_xscale("log")
    if xmax:
        ax.set_xlim(1e-1, xmax)
    ax.axvline(np.median(x), color=ps.INK_MUTED, linewidth=0.6, linestyle=(0, (2, 2)))
    ax.axhline(np.median(yv), color=ps.INK_MUTED, linewidth=0.6, linestyle=(0, (2, 2)))
    zx = np.abs(x - np.median(x)) / (x.std() or 1)
    zy = np.abs(yv - np.median(yv)) / (yv.std() or 1)
    z = np.maximum(zx, zy)
    placed: list[float] = []
    span = (yv.max() - yv.min()) or 1.0
    for i in [i for i in np.argsort(-z)[:3] if z[i] > 2]:
        if any(abs(yv[i] - p) < 0.08 * span for p in placed):
            continue
        placed.append(yv[i])
        ax.annotate(mols[i].capitalize(), (x[i], yv[i]), xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=7, color=ps.INK)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax.set_xlabel(r"Position variance (Å$^2$)")
    ax.set_ylabel(r"Per-step displacement var. (Å$^2$)")
    ps.save(fig, out)


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    if cmd in ("p", "t"):
        plot_sweep(a[0], a[1])
    elif cmd == "ablation":
        plot_ablation(a[0], a[1], a[2:])
    elif cmd == "dataset":
        xm = float(a[a.index("--xmax") + 1]) if "--xmax" in a else None
        plot_dataset(a[0], a[1], a[2], xm)
