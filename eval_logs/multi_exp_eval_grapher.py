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

        steps_col = _resolve_column(reader.fieldnames, ["Steps", "step"])
        success_col = _resolve_column(reader.fieldnames, ["Success", "success"])

        for row in reader:
            try:
                steps = float(row[steps_col])
                success = int(float(row[success_col]))
            except (TypeError, ValueError):
                continue

            rows.append(
                {
                    "steps": steps,
                    "success": 1 if success else 0,
                }
            )

    return rows


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


def _plot_overall_success_chart(ax, experiment_names, success_rates, color_map, experiment_name=""):
    x = list(range(len(experiment_names)))
    colors = [color_map[name] for name in experiment_names]
    bars = ax.bar(x, success_rates, color=colors, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(experiment_names)
    title = f"Overall Success Rate by Model ({experiment_name})" if experiment_name else "Overall Success Rate by Model"
    ax.set_title(title)
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.25)

    for bar, rate in zip(bars, success_rates):
        label = "n/a" if math.isnan(rate) else f"{rate:.1f}%"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5 if not math.isnan(rate) else 1.5,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _plot_failure_step_density(ax, rows_by_model, color_map, max_step, step_bin, experiment_name=""):
    for model_name, rows in rows_by_model.items():
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
        ax.scatter(xs, ys, s=24, alpha=0.8, color=color_map[model_name], label=model_name)

    title = f"Failure Density Over Steps ({experiment_name})" if experiment_name else "Failure Density Over Steps"
    ax.set_title(title)
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


def _infer_model_name(csv_path: Path, experiment_name: str) -> str:
    filename = csv_path.name
    core = filename
    if core.endswith("-eval_log.csv"):
        core = core[: -len("-eval_log.csv")]
    elif core.endswith(".csv"):
        core = core[:-4]

    suffix = f"_{experiment_name}"
    if core.endswith(suffix):
        model_name = core[: -len(suffix)]
    else:
        marker_idx = core.find(suffix)
        model_name = core[:marker_idx] if marker_idx > 0 else core

    return model_name or core


def _dedupe_name(base_name: str, used_names: set[str]) -> str:
    if base_name not in used_names:
        used_names.add(base_name)
        return base_name

    idx = 2
    while True:
        candidate = f"{base_name}_{idx}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        idx += 1


def _parse_csv_args(csv_args, experiment_name: str):
    """
    Parse CSV arguments. Each arg can be:
      - a path (infers model name from filename)
      - model_name=path (explicit override)
    
    Returns dict: {model_name -> Path}
    """
    if not csv_args:
        raise ValueError("Provide at least one --csv argument")
    
    result = {}
    for arg in csv_args:
        if "=" in arg:
            model_name, csv_path = arg.split("=", 1)
            model_name = model_name.strip()
            csv_path = Path(csv_path.strip())
            if not model_name or not csv_path:
                raise ValueError(f"Invalid CSV argument: {arg}. Expected model_name=path")
        else:
            csv_path = Path(arg)
            model_name = _infer_model_name(csv_path, experiment_name)
        
        if model_name in result:
            raise ValueError(f"Duplicate model name: {model_name}")
        result[model_name] = csv_path
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate model-level eval charts for a single experiment name and a list of CSV files"
        )
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="Experiment suffix in filenames, e.g. jitter, jitter_2faults, 3faults",
    )
    parser.add_argument(
        "--csv",
        action="append",
        default=[],
        metavar="CSV_PATH_OR_NAME_OVERRIDE",
        help=(
            "Path to an eval CSV for this experiment. Can be a plain path or 'model_name=path'. "
            "Repeat for multiple files."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Directory to write output image",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Override output path for combined chart",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=2000,
        help="Maximum step shown on failure density x-axis",
    )
    parser.add_argument(
        "--step-bin",
        type=int,
        default=1,
        help="Step-bin width for failure density points",
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.max_step <= 0:
        raise ValueError("--max-step must be > 0")
    if args.step_bin <= 0:
        raise ValueError("--step-bin must be > 0")

    csv_map = _parse_csv_args(args.csv, args.experiment_name)
    
    for model_name, path in csv_map.items():
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

    rows_by_model = {name: _read_eval_log(path) for name, path in csv_map.items()}
    print(f"Loaded {len(rows_by_model)} model(s)")
    for model_name in sorted(rows_by_model.keys()):
        print(f"  {model_name}")

    model_names = sorted(rows_by_model.keys())
    color_map = _experiment_color_map(model_names)
    overall_success_rates = {
        name: (sum(row["success"] for row in rows) / len(rows) * 100.0 if rows else math.nan)
        for name, rows in rows_by_model.items()
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = (
        Path(args.out)
        if args.out
        else out_dir / f"{args.experiment_name}_comparison.png"
    )

    # Create a single figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), constrained_layout=True)
    
    _plot_overall_success_chart(
        ax1,
        model_names,
        [overall_success_rates[name] for name in model_names],
        color_map,
        experiment_name=args.experiment_name,
    )
    
    _plot_failure_step_density(
        ax2,
        rows_by_model,
        color_map,
        max_step=args.max_step,
        step_bin=args.step_bin,
        experiment_name=args.experiment_name,
    )
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=170)
    print(f"Saved: {out_path}")
    
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
