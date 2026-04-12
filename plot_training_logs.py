import argparse
import ast
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="valid")


def parse_losses(raw: str) -> dict:
    if not raw or raw.strip() in {"", "None"}:
        return {}
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, SyntaxError):
        pass
    return {}


def load_log(csv_path: Path):
    episodes = []
    rewards = []
    critic1 = []
    critic2 = []
    actor = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ep = int(row["Episode"])
                rew = float(row["Reward"])
            except (KeyError, TypeError, ValueError):
                continue

            losses = parse_losses(row.get("Losses", ""))
            episodes.append(ep)
            rewards.append(rew)
            critic1.append(float(losses.get("critic1_loss", np.nan)))
            critic2.append(float(losses.get("critic2_loss", np.nan)))
            actor.append(float(losses.get("actor_loss", np.nan)))

    if not episodes:
        raise ValueError(f"No valid rows found in {csv_path}")

    return (
        np.asarray(episodes, dtype=np.int64),
        np.asarray(rewards, dtype=np.float64),
        np.asarray(critic1, dtype=np.float64),
        np.asarray(critic2, dtype=np.float64),
        np.asarray(actor, dtype=np.float64),
    )


def main():
    parser = argparse.ArgumentParser(description="Plot training reward/loss from training_log.csv")
    parser.add_argument("--csv", default="training_log.csv", help="Path to training log CSV")
    parser.add_argument("--window", type=int, default=25, help="Moving average window")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    episodes, rewards, critic1, critic2, actor = load_log(csv_path)

    reward_sm = moving_average(rewards, args.window)
    loss1_sm = moving_average(critic1[np.isfinite(critic1)], args.window)
    loss2_sm = moving_average(critic2[np.isfinite(critic2)], args.window)
    actor_sm = moving_average(actor[np.isfinite(actor)], args.window)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(episodes, rewards, alpha=0.25, label="reward")
    if len(reward_sm) > 0:
        x_reward = episodes[args.window - 1 :] if len(rewards) >= args.window else episodes
        axes[0].plot(x_reward, reward_sm, linewidth=2, label=f"reward MA({args.window})")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Training Reward")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    finite_c1 = np.isfinite(critic1)
    finite_c2 = np.isfinite(critic2)
    finite_a = np.isfinite(actor)

    axes[1].plot(episodes[finite_c1], critic1[finite_c1], alpha=0.2, label="critic1_loss")
    axes[1].plot(episodes[finite_c2], critic2[finite_c2], alpha=0.2, label="critic2_loss")
    axes[1].plot(episodes[finite_a], actor[finite_a], alpha=0.2, label="actor_loss")

    if len(loss1_sm) > 0 and np.sum(finite_c1) >= args.window:
        axes[1].plot(episodes[finite_c1][args.window - 1 :], loss1_sm, linewidth=2, label=f"critic1 MA({args.window})")
    if len(loss2_sm) > 0 and np.sum(finite_c2) >= args.window:
        axes[1].plot(episodes[finite_c2][args.window - 1 :], loss2_sm, linewidth=2, label=f"critic2 MA({args.window})")
    if len(actor_sm) > 0 and np.sum(finite_a) >= args.window:
        axes[1].plot(episodes[finite_a][args.window - 1 :], actor_sm, linewidth=2, label=f"actor MA({args.window})")

    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training Losses")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
