import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


BAR_COLORS = {
    "fault": "#d95f02",
    "baseline": "#1b9e77",
}


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _resolve_column(fieldnames, candidates):
    norm_map = {_normalize(col): col for col in fieldnames}
    for cand in candidates:
        key = _normalize(cand)
        if key in norm_map:
            return norm_map[key]
    raise KeyError(f"Could not find any of columns: {candidates}. Found: {fieldnames}")


def _read_eval_log(path: Path):
    rows = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")

        alpha_col = _resolve_column(reader.fieldnames, ["Fault Alpha", "fault_alpha", "alpha"])
        wheel_col = _resolve_column(
            reader.fieldnames,
            ["Damaged Wheel", "damaged_wheel", "wheel"],
        )
        steps_col = _resolve_column(reader.fieldnames, ["Steps", "step"])
        reward_col = _resolve_column(reader.fieldnames, ["Total Reward", "total_reward", "reward"])
        success_col = _resolve_column(reader.fieldnames, ["Success", "success"])

        for row in reader:
            try:
                alpha = float(row[alpha_col])
                wheel = int(float(row[wheel_col]))
                steps = float(row[steps_col])
                reward = float(row[reward_col])
                success = int(float(row[success_col]))
            except (TypeError, ValueError):
                continue

            rows.append(
                {
                    "alpha": alpha,
                    "wheel": wheel,
                    "steps": steps,
                    "reward": reward,
                    "success": 1 if success else 0,
                }
            )

    return rows


def _summarize_by(rows, key_name):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key_name]].append(row)

    summary = {}
    for key, bucket in grouped.items():
        count = len(bucket)
        success_bucket = [r for r in bucket if r["success"] == 1]
        fail_bucket = [r for r in bucket if r["success"] == 0]

        summary[key] = {
            "count": count,
            "avg_reward": sum(r["reward"] for r in bucket) / count if count else math.nan,
            "success_rate": (sum(r["success"] for r in bucket) / count * 100.0) if count else math.nan,
            "avg_steps_overall": sum(r["steps"] for r in bucket) / count if count else math.nan,
            "avg_steps_success": (
                sum(r["steps"] for r in success_bucket) / len(success_bucket)
                if success_bucket
                else math.nan
            ),
            "avg_steps_fail": (
                sum(r["steps"] for r in fail_bucket) / len(fail_bucket) if fail_bucket else math.nan
            ),
        }

    return summary


def _category_labels(categories, fault_summary, baseline_summary, formatter):
    labels = []
    for cat in categories:
        fault_count = fault_summary.get(cat, {}).get("count", 0)
        base_count = baseline_summary.get(cat, {}).get("count", 0)
        labels.append(f"{formatter(cat)}\nF:{fault_count} B:{base_count}")
    return labels


def _plot_two_bar_chart(ax, categories, labels, fault_vals, baseline_vals, title, ylabel):
    x = list(range(len(categories)))
    width = 0.36

    ax.bar(
        [i - width / 2 for i in x],
        fault_vals,
        width=width,
        label="Fault",
        color=BAR_COLORS["fault"],
        alpha=0.9,
    )
    ax.bar(
        [i + width / 2 for i in x],
        baseline_vals,
        width=width,
        label="Baseline",
        color=BAR_COLORS["baseline"],
        alpha=0.9,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")


def _plot_steps_breakdown(ax, categories, labels, fault_summary, baseline_summary, title):
    x = list(range(len(categories)))
    width = 0.13
    offsets = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]

    fault_overall = [fault_summary.get(cat, {}).get("avg_steps_overall", math.nan) for cat in categories]
    fault_success = [fault_summary.get(cat, {}).get("avg_steps_success", math.nan) for cat in categories]
    fault_fail = [fault_summary.get(cat, {}).get("avg_steps_fail", math.nan) for cat in categories]

    base_overall = [baseline_summary.get(cat, {}).get("avg_steps_overall", math.nan) for cat in categories]
    base_success = [baseline_summary.get(cat, {}).get("avg_steps_success", math.nan) for cat in categories]
    base_fail = [baseline_summary.get(cat, {}).get("avg_steps_fail", math.nan) for cat in categories]

    ax.bar([i + offsets[0] * width for i in x], fault_overall, width=width, color="#fc8d62", label="Fault Overall")
    ax.bar([i + offsets[1] * width for i in x], fault_success, width=width, color="#e6550d", label="Fault Success")
    ax.bar(
        [i + offsets[2] * width for i in x],
        fault_fail,
        width=width,
        color="#a63603",
        label="Fault Not Success",
    )

    ax.bar(
        [i + offsets[3] * width for i in x],
        base_overall,
        width=width,
        color="#a1d99b",
        label="Baseline Overall",
    )
    ax.bar(
        [i + offsets[4] * width for i in x],
        base_success,
        width=width,
        color="#31a354",
        label="Baseline Success",
    )
    ax.bar(
        [i + offsets[5] * width for i in x],
        base_fail,
        width=width,
        color="#006d2c",
        label="Baseline Not Success",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel("Average Steps")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        fontsize=8,
        ncol=3,
        frameon=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot evaluation comparisons between fault and baseline runs"
    )
    parser.add_argument("--fault-log", default="fault-eval_log.csv", help="Path to fault eval CSV")
    parser.add_argument(
        "--baseline-log",
        default="baseline-eval_log.csv",
        help="Path to baseline eval CSV",
    )
    parser.add_argument(
        "--alpha-out",
        default="eval_alpha_comparison.png",
        help="Output image path for alpha charts",
    )
    parser.add_argument(
        "--wheel-out",
        default="eval_wheel_comparison.png",
        help="Output image path for damaged-wheel charts",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not display plot window")
    args = parser.parse_args()

    fault_path = Path(args.fault_log)
    baseline_path = Path(args.baseline_log)

    if not fault_path.exists():
        raise FileNotFoundError(f"Fault eval log not found: {fault_path}")
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline eval log not found: {baseline_path}")

    fault_rows = _read_eval_log(fault_path)
    baseline_rows = _read_eval_log(baseline_path)

    fault_alpha = _summarize_by(fault_rows, "alpha")
    base_alpha = _summarize_by(baseline_rows, "alpha")
    fault_wheel = _summarize_by(fault_rows, "wheel")
    base_wheel = _summarize_by(baseline_rows, "wheel")

    alpha_categories = sorted(set(fault_alpha) | set(base_alpha))
    wheel_categories = sorted(set(fault_wheel) | set(base_wheel))

    alpha_labels = _category_labels(alpha_categories, fault_alpha, base_alpha, lambda a: f"alpha={a:g}")
    wheel_labels = _category_labels(wheel_categories, fault_wheel, base_wheel, lambda w: f"wheel={w}")

    alpha_fig, alpha_axes = plt.subplots(3, 1, figsize=(14, 14), constrained_layout=True)

    _plot_two_bar_chart(
        alpha_axes[0],
        alpha_categories,
        alpha_labels,
        [fault_alpha.get(a, {}).get("avg_reward", math.nan) for a in alpha_categories],
        [base_alpha.get(a, {}).get("avg_reward", math.nan) for a in alpha_categories],
        "Average Reward Over Fault Alpha",
        "Average Reward",
    )

    _plot_two_bar_chart(
        alpha_axes[1],
        alpha_categories,
        alpha_labels,
        [fault_alpha.get(a, {}).get("success_rate", math.nan) for a in alpha_categories],
        [base_alpha.get(a, {}).get("success_rate", math.nan) for a in alpha_categories],
        "Success Rate Over Fault Alpha",
        "Success Rate (%)",
    )

    _plot_steps_breakdown(
        alpha_axes[2],
        alpha_categories,
        alpha_labels,
        fault_alpha,
        base_alpha,
        "Average Steps Over Fault Alpha (Overall / Success / Not Success)",
    )

    wheel_fig, wheel_axes = plt.subplots(3, 1, figsize=(14, 14), constrained_layout=True)

    _plot_two_bar_chart(
        wheel_axes[0],
        wheel_categories,
        wheel_labels,
        [fault_wheel.get(w, {}).get("avg_reward", math.nan) for w in wheel_categories],
        [base_wheel.get(w, {}).get("avg_reward", math.nan) for w in wheel_categories],
        "Average Reward Over Damaged Wheel",
        "Average Reward",
    )

    _plot_two_bar_chart(
        wheel_axes[1],
        wheel_categories,
        wheel_labels,
        [fault_wheel.get(w, {}).get("success_rate", math.nan) for w in wheel_categories],
        [base_wheel.get(w, {}).get("success_rate", math.nan) for w in wheel_categories],
        "Success Rate Over Damaged Wheel",
        "Success Rate (%)",
    )

    _plot_steps_breakdown(
        wheel_axes[2],
        wheel_categories,
        wheel_labels,
        fault_wheel,
        base_wheel,
        "Average Steps Over Damaged Wheel (Overall / Success / Not Success)",
    )

    alpha_out_path = Path(args.alpha_out)
    alpha_out_path.parent.mkdir(parents=True, exist_ok=True)
    alpha_fig.savefig(alpha_out_path, dpi=170)
    print(f"Saved: {alpha_out_path}")

    wheel_out_path = Path(args.wheel_out)
    wheel_out_path.parent.mkdir(parents=True, exist_ok=True)
    wheel_fig.savefig(wheel_out_path, dpi=170)
    print(f"Saved: {wheel_out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
