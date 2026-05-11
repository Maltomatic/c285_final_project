#!/usr/bin/env python3
"""
history_grapher.py — compare success rate across history window sizes.

Run from anywhere:
    uv run python eval_logs/history_grapher.py

Inputs (from eval_logs/):
    fault_k1-eval_log.csv   — k=1  (no history)
    fault_k3-eval_log.csv   — k=3  (short history, ~60ms)
    fault_w1-eval_log.csv   — k=5  (baseline, trained as 'fault')
    fault_k10-eval_log.csv  — k=10 (long history, ~200ms)

Outputs (saved to eval_logs/):
    history_success.png  — success rate vs window size
    history_reward.png   — average episode reward vs window size
    history_steps.png    — average episode steps vs window size
"""

import csv
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent

EXPS = [
    {"k": 1,  "name": "fault_k1",  "csv": "fault_k1-eval_log.csv",  "label": "k=1\n(no history)"},
    {"k": 3,  "name": "fault_k3",  "csv": "fault_k3-eval_log.csv",  "label": "k=3\n(~60ms)"},
    {"k": 5,  "name": "fault_w1",  "csv": "fault_w1-eval_log.csv",  "label": "k=5\n(baseline)"},
    {"k": 10, "name": "fault_k10", "csv": "fault_k10-eval_log.csv", "label": "k=10\n(~200ms)"},
]

COLOR_BAR   = "#d95f02"
COLOR_LINE  = "#1b9e77"
COLOR_BASE  = "#7570b3"


def _load(csv_name):
    path = SCRIPT_DIR / csv_name
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


def _sr(rows):
    if not rows:
        return math.nan
    return sum(r["success"] for r in rows) / len(rows) * 100.0


def _avg(rows, key):
    if not rows:
        return math.nan
    return sum(r[key] for r in rows) / len(rows)


for exp in EXPS:
    exp["data"] = _load(exp["csv"])

# ── Print summary table ───────────────────────────────────────────────────────
print(f"\n{'k':>4}  {'Experiment':<14} {'N':>5}  {'Success%':>9}  {'Avg Reward':>11}  {'Avg Steps':>10}")
print("─" * 60)
for exp in EXPS:
    d = exp["data"]
    if d:
        print(f"{exp['k']:>4}  {exp['name']:<14} {len(d):>5}  {_sr(d):>8.1f}%  "
              f"{_avg(d, 'reward'):>11.1f}  {_avg(d, 'steps'):>10.1f}")
    else:
        print(f"{exp['k']:>4}  {exp['name']:<14} {'—':>5}  {'missing':>9}  {'—':>11}  {'—':>10}")
print()

ks     = [e["k"]          for e in EXPS]
labels = [e["label"]       for e in EXPS]
srs    = [_sr(e["data"])   for e in EXPS]
rwds   = [_avg(e["data"], "reward") for e in EXPS]
steps  = [_avg(e["data"], "steps")  for e in EXPS]

baseline_idx = next((i for i, e in enumerate(EXPS) if e["name"] == "fault_w1"), None)


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


def _save(fig, filename):
    out = SCRIPT_DIR / filename
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


# ── Figure 1: Success rate ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
colors = [COLOR_BASE if e["name"] == "fault_w1" else COLOR_BAR for e in EXPS]
bars = ax.bar(labels, srs, color=colors, alpha=0.9, width=0.55)
_label_bars(ax, bars)

if baseline_idx is not None and not math.isnan(srs[baseline_idx]):
    ax.axhline(srs[baseline_idx], color=COLOR_BASE, linestyle="--", linewidth=1.5,
               label=f"k=5 baseline: {srs[baseline_idx]:.1f}%")
    ax.legend(fontsize=9)

ax.set_title("History Window Ablation — Success Rate")
ax.set_ylabel("Success Rate (%)")
ax.set_ylim(0, 110)
ax.grid(True, axis="y", alpha=0.25)
_save(fig, "history_success.png")

# ── Figure 2: Avg reward ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
valid = [(e["label"], r) for e, r in zip(EXPS, rwds) if not math.isnan(r)]
if valid:
    lbs, rs = zip(*valid)
    ax.plot(range(len(lbs)), rs, marker="o", color=COLOR_LINE, linewidth=2)
    ax.set_xticks(range(len(lbs)))
    ax.set_xticklabels(lbs, fontsize=9)
    for i, (lb, r) in enumerate(zip(lbs, rs)):
        ax.text(i, r + abs(r) * 0.02, f"{r:.1f}", ha="center", va="bottom", fontsize=9)
ax.set_title("History Window Ablation — Average Episode Reward")
ax.set_ylabel("Average Total Reward")
ax.grid(True, axis="y", alpha=0.25)
_save(fig, "history_reward.png")

# ── Figure 3: Avg steps (episode length) ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
valid = [(e["label"], s) for e, s in zip(EXPS, steps) if not math.isnan(s)]
if valid:
    lbs, ss = zip(*valid)
    ax.plot(range(len(lbs)), ss, marker="s", color=COLOR_BAR, linewidth=2)
    ax.set_xticks(range(len(lbs)))
    ax.set_xticklabels(lbs, fontsize=9)
    for i, (lb, s) in enumerate(zip(lbs, ss)):
        ax.text(i, s + max(ss) * 0.02, f"{s:.0f}", ha="center", va="bottom", fontsize=9)
ax.set_title("History Window Ablation — Average Episode Length")
ax.set_ylabel("Average Steps to Termination")
ax.grid(True, axis="y", alpha=0.25)
_save(fig, "history_steps.png")

print("\nAll plots saved to eval_logs/")
