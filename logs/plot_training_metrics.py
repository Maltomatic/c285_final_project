import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt

N_PTS = 500

def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _resolve_column(fieldnames, candidates):
    norm_map = {_normalize(col): col for col in fieldnames}
    for cand in candidates:
        key = _normalize(cand)
        if key in norm_map:
            return norm_map[key]
    raise KeyError(f"Could not find any of columns: {candidates}. Found: {fieldnames}")


def _sample_series(x_values, *series, max_points=N_PTS):
    if len(x_values) <= max_points or max_points <= 0:
        return (x_values, *series)

    last_idx = len(x_values) - 1
    # Uniformly sample indices while always including first and last points.
    indices = sorted({round(i * last_idx / (max_points - 1)) for i in range(max_points)})

    sampled_x = [x_values[i] for i in indices]
    sampled_series = [[values[i] for i in indices] for values in series]
    return (sampled_x, *sampled_series)


def _read_training_log(path: Path):
    global_steps = []
    c1_losses = []
    c2_losses = []
    actor_losses = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")

        step_col = _resolve_column(reader.fieldnames, ["Global step", "global_step", "step"])
        c1_col = _resolve_column(reader.fieldnames, ["Critic1 Loss", "critic1_loss", "c1_loss"])
        c2_col = _resolve_column(reader.fieldnames, ["Critic2 Loss", "critic2_loss", "c2_loss"])
        actor_col = _resolve_column(reader.fieldnames, ["Actor Loss", "actor_loss"])

        for row in reader:
            try:
                step = float(row[step_col])
                c1 = float(row[c1_col])
                c2 = float(row[c2_col])
                actor = float(row[actor_col])
            except (TypeError, ValueError):
                continue

            global_steps.append(step)
            c1_losses.append(c1)
            c2_losses.append(c2)
            actor_losses.append(actor)

    # Keep line continuity sane if rows are out of order.
    order = sorted(range(len(global_steps)), key=lambda i: global_steps[i])
    global_steps = [global_steps[i] for i in order]
    c1_losses = [c1_losses[i] for i in order]
    c2_losses = [c2_losses[i] for i in order]
    actor_losses = [actor_losses[i] for i in order]

    return global_steps, c1_losses, c2_losses, actor_losses


def _read_episode_log(path: Path):
    episodes = []
    steps_per_episode = []
    rewards = []
    discounted_rewards = []
    expected_qs = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")

        episode_col = _resolve_column(reader.fieldnames, ["Episode", "episode"])
        steps_col = _resolve_column(reader.fieldnames, ["Steps", "step"])
        reward_col = _resolve_column(reader.fieldnames, ["Total Reward", "reward", "total_reward"])
        discounted_col = _resolve_column(
            reader.fieldnames,
            ["Discounted Reward", "discounted_reward", "discounted"],
        )
        expected_q_col = _resolve_column(
                reader.fieldnames,
                ["Expected Q", "expected_q", "q", "estimated_q"],
            )
        for row in reader:
            try:
                episode = int(float(row[episode_col]))
                steps = float(row[steps_col])
                reward = float(row[reward_col])
                discounted = float(row[discounted_col])
                expected_q = float(row[expected_q_col])
            except (TypeError, ValueError):
                continue

            episodes.append(episode)
            steps_per_episode.append(steps)
            rewards.append(reward)
            discounted_rewards.append(discounted)
            expected_qs.append(expected_q)

    # Sort by episode index so asynchronous logs are plotted in temporal order.
    order = sorted(range(len(episodes)), key=lambda i: episodes[i])
    episodes = [episodes[i] for i in order]
    steps_per_episode = [steps_per_episode[i] for i in order]
    rewards = [rewards[i] for i in order]
    discounted_rewards = [discounted_rewards[i] for i in order]
    expected_qs = [expected_qs[i] for i in order]

    return episodes, steps_per_episode, rewards, discounted_rewards, expected_qs


def main():
    parser = argparse.ArgumentParser(description="Plot TD3 training metrics from CSV logs")
    parser.add_argument(
        "prefix",
        nargs="?",
        default="",
        help="Optional filename prefix (e.g. 'fault' -> fault-training_log.csv)",
    )
    parser.add_argument("--training-log", default="", help="Path to training log CSV")
    parser.add_argument("--episode-log", default="", help="Path to episode log CSV")
    parser.add_argument("--outdir", default=".", help="Directory to write output PNGs")
    args = parser.parse_args()

    prefix = args.prefix.strip()
    file_prefix = f"{prefix}-" if prefix else ""
    training_name = f"{file_prefix}training_log.csv"
    episode_name = f"{file_prefix}episode_returns.csv"

    training_log = Path(args.training_log) if args.training_log else Path(training_name)
    episode_log = Path(args.episode_log) if args.episode_log else Path(episode_name)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not training_log.exists():
        raise FileNotFoundError(f"Training log not found: {training_log}")
    if not episode_log.exists():
        raise FileNotFoundError(f"Episode log not found: {episode_log}")

    steps, c1, c2, actor = _read_training_log(training_log)
    episodes, steps_per_episode, rewards, discounted, expected_qs = _read_episode_log(episode_log)

    steps, c1, c2, actor = _sample_series(steps, c1, c2, actor)
    episodes, steps_per_episode, rewards, discounted, expected_qs = _sample_series(
        episodes,
        steps_per_episode,
        rewards,
        discounted,
        expected_qs,
    )

    # Graph 1: losses over global steps
    plt.figure(figsize=(10, 6))
    plt.plot(steps, c1, label="Critic1 Loss", linewidth=1.5)
    plt.plot(steps, c2, label="Critic2 Loss", linewidth=1.5)
    plt.plot(steps, actor, label="Actor Loss", linewidth=1.5)
    plt.xlabel("Global Steps")
    plt.ylabel("Loss")
    plt.title("Losses vs Global Steps")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    losses_path = outdir / f"{file_prefix}losses_vs_global_steps.png"
    plt.savefig(losses_path, dpi=160)
    plt.close()

    # Graph 2: rewards over episodes
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, label="Total Reward", linewidth=1.5)
    plt.plot(episodes, discounted, label="Discounted Reward", linewidth=1.5)
    plt.plot(episodes, expected_qs, label="Expected Q (episode start)", linewidth=1.5)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards and Expected Q vs Episode")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    rewards_path = outdir / f"{file_prefix}rewards_vs_episode.png"
    plt.savefig(rewards_path, dpi=160)
    plt.close()

    # Graph 3: steps over episodes
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, steps_per_episode, label="Episode Steps", linewidth=1.5)
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Steps vs Episode")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    steps_path = outdir / f"{file_prefix}steps_vs_episode.png"
    plt.savefig(steps_path, dpi=160)
    plt.close()

    # Show all three plots in one interactive 2x2 window after saving individual PNGs.
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
    ax_ul, ax_ur = axes[0]
    ax_ll, ax_lr = axes[1]

    ax_ul.plot(steps, c1, label="Critic1 Loss", linewidth=1.5)
    ax_ul.plot(steps, c2, label="Critic2 Loss", linewidth=1.5)
    ax_ul.plot(steps, actor, label="Actor Loss", linewidth=1.5)
    ax_ul.set_xlabel("Global Steps")
    ax_ul.set_ylabel("Loss")
    ax_ul.set_title("Losses vs Global Steps")
    ax_ul.grid(True, alpha=0.25)
    ax_ul.legend()

    ax_ur.plot(episodes, rewards, label="Total Reward", linewidth=1.5)
    ax_ur.plot(episodes, discounted, label="Discounted Reward", linewidth=1.5)
    ax_ur.plot(episodes, expected_qs, label="Expected Q (episode start)", linewidth=1.5)
    ax_ur.set_xlabel("Episode")
    ax_ur.set_ylabel("Reward")
    ax_ur.set_title("Rewards and Expected Q vs Episode")
    ax_ur.grid(True, alpha=0.25)
    ax_ur.legend()

    ax_ll.plot(episodes, steps_per_episode, label="Episode Steps", linewidth=1.5)
    ax_ll.set_xlabel("Episode")
    ax_ll.set_ylabel("Steps")
    ax_ll.set_title("Steps vs Episode")
    ax_ll.grid(True, alpha=0.25)
    ax_ll.legend()

    # Keep lower-right quadrant intentionally empty.
    ax_lr.axis("off")

    plt.show()

    print(f"Saved: {losses_path}")
    print(f"Saved: {rewards_path}")
    print(f"Saved: {steps_path}")


if __name__ == "__main__":
    main()
