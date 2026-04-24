import csv
import os
import gymnasium as gym
from envs.rewards import tracking_reward, sparse_reward, eval_reward
from envs.env_configs import RENDER_TRAINING, DEBUG, EVAL

csv_path = 'training_log.csv'
csv_eps_log_path = 'episode_returns.csv'
checkpoint_base_path = 'td3_checkpoint'

def make_env(rwd = "tracking", id=0, no_fault=False, pure_rl=False):
    def _init():
        return gym.make("SixWheelSkidSteer-v0",
                        render_mode="human" if RENDER_TRAINING else None,
                        reward_fn=eval_reward if EVAL else tracking_reward if rwd == "tracking" else sparse_reward if rwd == "sparse" else None,
                        env_id=id,
                        no_fault=no_fault,
                        pure_rl=pure_rl)
    return _init

def logging(episode, global_steps, critic1_loss, critic2_loss, actor_loss):
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as log_file:
            log_writer = csv.writer(log_file)
            log_writer.writerow(['Episode', 'Global step', 'Critic1 Loss', 'Critic2 Loss', 'Actor Loss'])
    with open(csv_path, mode='a', newline='') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow([episode, global_steps, critic1_loss, critic2_loss, actor_loss])

def eps_logging(episode, steps, total_reward, discounted_reward, expected_q):
    if not os.path.exists(csv_eps_log_path):
        with open(csv_eps_log_path, mode='w', newline='') as log_file:
            log_writer = csv.writer(log_file)
            log_writer.writerow(['Episode', 'Steps', 'Total Reward', 'Discounted Reward', 'Expected Q'])
    with open(csv_eps_log_path, mode='a', newline='') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow([episode, steps, total_reward, discounted_reward, expected_q])

def extract_eps_info(infos, key, idx, default=None):
    if not isinstance(infos, dict) or key not in infos:
        return default
    values = infos[key]
    mask_key = f"_{key}"
    if mask_key in infos:
        try:
            if not bool(infos[mask_key][idx]):
                return default
        except Exception:
            pass
    try:
        return values[idx]
    except Exception:
        return values