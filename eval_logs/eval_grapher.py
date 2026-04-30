import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


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

def _plot_overall_success_chart(ax, experiment_names, success_rates, color_map):
    x = list(range(len(experiment_names)))
    colors = [color_map[name] for name in experiment_names]
    bars = ax.bar(x, success_rates, color=colors, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(experiment_names)
    ax.set_title("Overall Success Rate by Experiment")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.25)

    for bar, rate in zip(bars, success_rates):
        if math.isnan(rate):
            label = "n/a"
        else:
            label = f"{rate:.1f}%"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5 if not math.isnan(rate) else 1.5,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )



def _category_labels(categories, experiment_summaries, formatter):
    labels = []
    for cat in categories:
        total_count = sum(summary.get(cat, {}).get("count", 0) for summary in experiment_summaries.values())
        labels.append(f"{formatter(cat)}\nn={total_count}")
    return labels


def _blend_towards_white(color, mix_factor):
    r, g, b = mcolors.to_rgb(color)
    return (
        r * (1.0 - mix_factor) + mix_factor,
        g * (1.0 - mix_factor) + mix_factor,
        b * (1.0 - mix_factor) + mix_factor,
    )


def _experiment_color_map(experiment_names):
    color_map = {}
    tab20 = plt.get_cmap("tab20")
    palette = [tab20(i) for i in range(20)]
    palette_idx = 0
    for name in experiment_names:
        if name in BAR_COLORS:
            color_map[name] = BAR_COLORS[name]
            continue
        color_map[name] = palette[palette_idx % len(palette)]
        palette_idx += 1
    return color_map


def _plot_grouped_bar_chart(ax, categories, labels, series_by_experiment, title, ylabel, color_map):
    x = list(range(len(categories)))
    num_experiments = max(1, len(series_by_experiment))
    width = min(0.8 / num_experiments, 0.36)

    for idx, (name, values) in enumerate(series_by_experiment.items()):
        offset = (idx - (num_experiments - 1) / 2) * width
        ax.bar(
            [i + offset for i in x],
            values,
            width=width,
            label=name,
            color=color_map[name],
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        fontsize=8,
        ncol=max(1, min(5, len(series_by_experiment))),
        frameon=True,
    )


def _plot_failure_step_density(ax, rows_by_experiment, color_map, max_step, step_bin):
    for name, rows in rows_by_experiment.items():
        step_counts = defaultdict(int)
        for row in rows:
            if row["success"] == 1:
                continue

            step = row["steps"]
            if step < 0 or step > max_step:
                continue

            # Bin steps so nearby failures can be viewed as one dot.
            binned_step = int(step // step_bin) * step_bin
            step_counts[binned_step] += 1

        if not step_counts:
            continue

        xs = sorted(step_counts.keys())
        ys = [step_counts[x] for x in xs]
        ax.scatter(xs, ys, s=24, alpha=0.8, color=color_map[name], label=name)

    ax.set_title("Failure Density Over Steps")
    ax.set_xlabel("Step")
    ax.set_ylabel("Failure Count")
    ax.set_xlim(0, max_step)
    ax.grid(True, alpha=0.25)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        fontsize=8,
        ncol=3,
        frameon=True,
    )


def _plot_steps_breakdown(ax, categories, labels, summaries_by_experiment, title, color_map):
    x = list(range(len(categories)))
    metric_keys = [
        ("avg_steps_overall", "Overall", 0.25),
        ("avg_steps_success", "Success", 0.05),
        ("avg_steps_fail", "Not Success", 0.45),
    ]
    num_series = max(1, len(summaries_by_experiment) * len(metric_keys))
    width = min(0.8 / num_series, 0.16)

    series_idx = 0
    for name, summary in summaries_by_experiment.items():
        base_color = color_map[name]
        for metric_key, metric_label, lighten in metric_keys:
            values = [summary.get(cat, {}).get(metric_key, math.nan) for cat in categories]
            offset = (series_idx - (num_series - 1) / 2) * width
            ax.bar(
                [i + offset for i in x],
                values,
                width=width,
                color=_blend_towards_white(base_color, lighten),
                label=f"{name} {metric_label}",
            )
            series_idx += 1

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel("Average Steps")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        fontsize=8,
        ncol=max(3, min(6, len(summaries_by_experiment) * 3)),
        frameon=True,
    )


def _parse_experiment_arg(spec):
    if "=" not in spec:
        raise ValueError(f"Invalid --experiment value '{spec}'. Expected NAME=PATH")
    name, path = spec.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"Invalid --experiment value '{spec}'. Expected NAME=PATH")
    return name, Path(path)


def _resolve_experiment_paths(args):
    experiment_paths = {}
    if args.experiment:
        for spec in args.experiment:
            name, path = _parse_experiment_arg(spec)
            if name in experiment_paths:
                raise ValueError(f"Duplicate experiment name provided: {name}")
            experiment_paths[name] = path
    else:
        experiment_paths["fault"] = Path(args.fault_log)
        experiment_paths["baseline"] = Path(args.baseline_log)

    return experiment_paths


def main():
    parser = argparse.ArgumentParser(
        description="Plot evaluation comparisons for one or more experiments"
    )
    parser.add_argument("--fault-log", default="fault-eval_log.csv", help="Path to fault eval CSV")
    parser.add_argument(
        "--baseline-log",
        default="baseline-eval_log.csv",
        help="Path to baseline eval CSV",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Experiment input in NAME=PATH format. Repeat for multiple experiments. "
            "If omitted, --fault-log and --baseline-log are used."
        ),
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
    parser.add_argument(
        "--overall-out",
        default="eval_overall_success_comparison.png",
        help="Output image path for overall success-rate chart",
    )
    parser.add_argument(
        "--failure-out",
        default="eval_failure_step_density.png",
        help="Output image path for failure step density chart",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=2000,
        help="Maximum step shown on failure density x-axis",
    )
    parser.add_argument(
        "--failure-step-bin",
        type=int,
        default=1,
        help="Step-bin width for failure density points (1 means exact step)",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not display plot window")
    args = parser.parse_args()

    if args.max_step <= 0:
        raise ValueError("--max-step must be > 0")
    if args.failure_step_bin <= 0:
        raise ValueError("--failure-step-bin must be > 0")

    experiment_paths = _resolve_experiment_paths(args)

    for name, path in experiment_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Eval log not found for experiment '{name}': {path}")

    rows_by_experiment = {name: _read_eval_log(path) for name, path in experiment_paths.items()}
    alpha_summary_by_experiment = {
        name: _summarize_by(rows, "alpha") for name, rows in rows_by_experiment.items()
    }
    wheel_summary_by_experiment = {
        name: _summarize_by(rows, "wheel") for name, rows in rows_by_experiment.items()
    }
    color_map = _experiment_color_map(list(experiment_paths.keys()))
    overall_success_rates = {
        name: (sum(row["success"] for row in rows) / len(rows) * 100.0 if rows else math.nan)
        for name, rows in rows_by_experiment.items()
    }


    alpha_categories = sorted(
        set().union(*(summary.keys() for summary in alpha_summary_by_experiment.values()))
    )
    wheel_categories = sorted(
        set().union(*(summary.keys() for summary in wheel_summary_by_experiment.values()))
    )

    alpha_labels = _category_labels(alpha_categories, alpha_summary_by_experiment, lambda a: f"alpha={a:g}")
    wheel_labels = _category_labels(wheel_categories, wheel_summary_by_experiment, lambda w: f"wheel={w}")

    alpha_fig, alpha_axes = plt.subplots(3, 1, figsize=(14, 14), constrained_layout=True)

    _plot_grouped_bar_chart(
        alpha_axes[0],
        alpha_categories,
        alpha_labels,
        {
            name: [summary.get(a, {}).get("avg_reward", math.nan) for a in alpha_categories]
            for name, summary in alpha_summary_by_experiment.items()
        },
        "Average Reward Over Fault Alpha",
        "Average Reward",
        color_map,
    )

    _plot_grouped_bar_chart(
        alpha_axes[1],
        alpha_categories,
        alpha_labels,
        {
            name: [summary.get(a, {}).get("success_rate", math.nan) for a in alpha_categories]
            for name, summary in alpha_summary_by_experiment.items()
        },
        "Success Rate Over Fault Alpha",
        "Success Rate (%)",
        color_map,
    )

    _plot_steps_breakdown(
        alpha_axes[2],
        alpha_categories,
        alpha_labels,
        alpha_summary_by_experiment,
        "Average Steps Over Fault Alpha (Overall / Success / Not Success)",
        color_map,
    )

    overall_fig, overall_ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
    overall_experiment_names = list(experiment_paths.keys())
    _plot_overall_success_chart(
        overall_ax,
        overall_experiment_names,
        [overall_success_rates[name] for name in overall_experiment_names],
        color_map,
    )

    failure_fig, failure_ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
    _plot_failure_step_density(
        failure_ax,
        rows_by_experiment,
        color_map,
        max_step=args.max_step,
        step_bin=args.failure_step_bin,
    )

    wheel_fig, wheel_axes = plt.subplots(3, 1, figsize=(14, 14), constrained_layout=True)

    _plot_grouped_bar_chart(
        wheel_axes[0],
        wheel_categories,
        wheel_labels,
        {
            name: [summary.get(w, {}).get("avg_reward", math.nan) for w in wheel_categories]
            for name, summary in wheel_summary_by_experiment.items()
        },
        "Average Reward Over Damaged Wheel",
        "Average Reward",
        color_map,
    )

    _plot_grouped_bar_chart(
        wheel_axes[1],
        wheel_categories,
        wheel_labels,
        {
            name: [summary.get(w, {}).get("success_rate", math.nan) for w in wheel_categories]
            for name, summary in wheel_summary_by_experiment.items()
        },
        "Success Rate Over Damaged Wheel",
        "Success Rate (%)",
        color_map,
    )

    _plot_steps_breakdown(
        wheel_axes[2],
        wheel_categories,
        wheel_labels,
        wheel_summary_by_experiment,
        "Average Steps Over Damaged Wheel (Overall / Success / Not Success)",
        color_map,
    )

    alpha_out_path = Path(args.alpha_out)
    alpha_out_path.parent.mkdir(parents=True, exist_ok=True)
    alpha_fig.savefig(str(alpha_out_path), dpi=170)
    print(f"Saved: {alpha_out_path}")

    wheel_out_path = Path(args.wheel_out)
    wheel_out_path.parent.mkdir(parents=True, exist_ok=True)
    overall_out_path = Path(args.overall_out)
    overall_out_path.parent.mkdir(parents=True, exist_ok=True)
    overall_fig.savefig(str(overall_out_path), dpi=170)
    print(f"Saved: {overall_out_path}")

    failure_out_path = Path(args.failure_out)
    failure_out_path.parent.mkdir(parents=True, exist_ok=True)
    failure_fig.savefig(str(failure_out_path), dpi=170)
    print(f"Saved: {failure_out_path}")

    wheel_fig.savefig(str(wheel_out_path), dpi=170)
    print(f"Saved: {wheel_out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")


