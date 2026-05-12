#!/usr/bin/env python3
"""
ablation_grapher.py — focused comparison plots for the run_ablation.sh experiments.

Run from anywhere:
    uv run python eval_logs/ablation_grapher.py

Outputs (saved to eval_logs/):
    ablation_overall.png      — success rate across all 9 experiments
    ablation_multiwheels.png  — residual vs pure RL as fault count increases
    ablation_sameside.png     — same-side vs random fault placement
    ablation_jitter.png       — constant vs jittered fault

For detailed per-alpha / per-wheel breakdowns, use the existing eval_grapher.py:
    cd eval_logs
    uv run python eval_grapher.py \\
        --experiment fault_w1=fault_w1-eval_log.csv \\
        --experiment fault_w2=fault_w2-eval_log.csv \\
        --experiment fault_w3=fault_w3-eval_log.csv
"""

import csv
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent

# ── Experiment registry ───────────────────────────────────────────────────────
EXPS = {
    "fault_w1":          "fault_w1-eval_log.csv",
    "fault_w2":          "fault_w2-eval_log.csv",
    "fault_w3":          "fault_w3-eval_log.csv",
    "fault_w2_sameside": "fault_w2_sameside-eval_log.csv",
    "fault_w3_sameside": "fault_w3_sameside-eval_log.csv",
    "fault_constant":    "fault_constant-eval_log.csv",
    "fault_jitter":      "fault_jitter-eval_log.csv",
    "pure_w2":           "pure_w2-eval_log.csv",
    "pure_w3":           "pure_w3-eval_log.csv",
}

COLORS = {
    "residual": "#d95f02",
    "pure":     "#1b9e77",
    "sameside": "#e7298a",
    "jitter":   "#7570b3",
}

# ── Data loading ──────────────────────────────────────────────────────────────
def _load(name):
    path = SCRIPT_DIR / EXPS[name]
    if not path.exists():
        return None
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "success": int(float(row["Success"])),
                    "reward":  float(row["Total Reward"]),
                    "steps":   float(row["Steps"]),
                })
            except (KeyError, ValueError):
                continue
    return rows or None

def _sr(name):
    rows = data[name]
    if not rows:
        return math.nan
    return sum(r["success"] for r in rows) / len(rows) * 100.0

def _avg(name, key):
    rows = data[name]
    if not rows:
        return math.nan
    return sum(r[key] for r in rows) / len(rows)

data = {name: _load(name) for name in EXPS}

# ── Print summary table ───────────────────────────────────────────────────────
print(f"\n{'Experiment':<22} {'N':>5}  {'Success%':>9}  {'Avg Reward':>11}  {'Avg Steps':>10}")
print("─" * 63)
for name in EXPS:
    rows = data[name]
    if rows:
        print(f"{name:<22} {len(rows):>5}  {_sr(name):>8.1f}%  "
              f"{_avg(name, 'reward'):>11.1f}  {_avg(name, 'steps'):>10.1f}")
    else:
        print(f"{name:<22} {'—':>5}  {'missing':>9}  {'—':>11}  {'—':>10}")
print()

# ── Shared helpers ────────────────────────────────────────────────────────────
def _label_bars(ax, bars):
    for bar in bars:
        v = bar.get_height()
        if not math.isnan(v):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 1.2,
                f"{v:.1f}%",
                ha="center", va="bottom", fontsize=9,
            )

def _finish(ax, title, ylabel="Success Rate (%)", ylim=(0, 110)):
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.25)

def _save(fig, filename):
    out = SCRIPT_DIR / filename
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)

# ── Figure 1: Overall success rate across all 9 experiments ──────────────────
fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)

# Color each bar by its category
bar_colors = (
    [COLORS["residual"]] * 3 +           # fault_w1/w2/w3
    [COLORS["sameside"]] * 2 +            # w2_sameside/w3_sameside
    [COLORS["residual"], COLORS["jitter"]] +  # constant/jitter
    [COLORS["pure"]] * 2                  # pure_w2/w3
)
names  = list(EXPS.keys())
values = [_sr(n) for n in names]
bars = ax.bar(names, values, color=bar_colors, alpha=0.9)
_label_bars(ax, bars)
ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)

from matplotlib.patches import Patch
legend_handles = [
    Patch(color=COLORS["residual"], label="Residual RL"),
    Patch(color=COLORS["pure"],     label="Pure RL"),
    Patch(color=COLORS["sameside"], label="Same-side placement"),
    Patch(color=COLORS["jitter"],   label="Jitter"),
]
ax.legend(handles=legend_handles, fontsize=9)
_finish(ax, "Ablation: Overall Success Rate Across All Experiments")
_save(fig, "ablation_overall.png")

# ── Figure 2: Multi-wheel fault — residual vs pure ────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
x = np.arange(2)
w = 0.3
res_vals  = [_sr("fault_w2"), _sr("fault_w3")]
pure_vals = [_sr("pure_w2"),  _sr("pure_w3")]
b1 = ax.bar(x - w / 2, res_vals,  w, label="Residual RL (fault-trained)", color=COLORS["residual"], alpha=0.9)
b2 = ax.bar(x + w / 2, pure_vals, w, label="Pure RL (fault-trained)",     color=COLORS["pure"],     alpha=0.9)
_label_bars(ax, b1)
_label_bars(ax, b2)

# Single-wheel baseline as a dashed reference line
w1 = _sr("fault_w1")
if not math.isnan(w1):
    ax.axhline(w1, color=COLORS["residual"], linestyle="--", linewidth=1.5,
               label=f"Residual 1-wheel baseline: {w1:.1f}%")

ax.set_xticks(x)
ax.set_xticklabels(["2 Faulted Wheels", "3 Faulted Wheels"])
ax.legend(fontsize=9)
_finish(ax, "Multi-Wheel Fault: Residual vs Pure RL")
_save(fig, "ablation_multiwheels.png")

# ── Figure 3: Same-side vs random placement ───────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
x = np.arange(2)
rand_vals = [_sr("fault_w2"),           _sr("fault_w3")]
side_vals = [_sr("fault_w2_sameside"),  _sr("fault_w3_sameside")]
b1 = ax.bar(x - w / 2, rand_vals, w, label="Random placement", color=COLORS["residual"], alpha=0.9)
b2 = ax.bar(x + w / 2, side_vals, w, label="Same-side",        color=COLORS["sameside"], alpha=0.9)
_label_bars(ax, b1)
_label_bars(ax, b2)
ax.set_xticks(x)
ax.set_xticklabels(["2 Faulted Wheels", "3 Faulted Wheels"])
ax.legend(fontsize=9)
_finish(ax, "Fault Placement: Same-Side vs Random (Residual RL)")
_save(fig, "ablation_sameside.png")

# ── Figure 4: Jitter robustness ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
labels = ["Constant fault\n(no jitter)", "Jittered fault\n(σ=0.1)"]
vals   = [_sr("fault_constant"), _sr("fault_jitter")]
bars = ax.bar(labels, vals, color=[COLORS["residual"], COLORS["jitter"]], width=0.45, alpha=0.9)
_label_bars(ax, bars)
_finish(ax, "Fault Jitter Robustness — 1 Wheel (Residual RL)")
_save(fig, "ablation_jitter.png")

print("\nAll plots saved to eval_logs/")
