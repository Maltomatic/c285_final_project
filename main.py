import argparse
from typing import Any, Optional
import numpy as np
import torch, envs, gymnasium as gym
from envs.six_wheel_env import SixWheelEnv
from controllers.TD3_controller import Agent as TD3Agent
from controllers.PPO_controller import Agent as PPOAgent
import time, csv, multiprocessing, threading

from controllers.utils.model_configs import DECAY_INTERVAL, DISCOUNT, G_STEPS, RWD_FN
from envs.utils import env_helpers
from envs.utils.env_helpers import logging, eps_logging, make_env, extract_eps_info
from envs.env_configs import DEBUG, EVAL_EPISODES, GPU_THREAD, _ACTION_DIM, EVAL
import envs.env_configs as env_config

def print_d(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

def main():
    parser = argparse.ArgumentParser(description="Train RL agent with parallel environments")
    parser.add_argument("--algo", type=str, default="td3", choices=["td3", "ppo"], help="RL algorithm to use")
    parser.add_argument("--no-fault", action="store_true", help="Disable injected wheel faults (NO_FAULT mode)")
    parser.add_argument("--no-op", action="store_true", default=False, help="Disable agent actions and use zero actions instead")
    parser.add_argument("--pure", action="store_true", default=False, help="Pure RL, no residuals (use ZeroAllocator)")
    parser.add_argument("--ft", action="store_true", default=False, help="Use fine tuning training process")
    parser.add_argument("--eval", action="store_true", help="Enable eval mode")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment prefix for logs/checkpoints")
    parser.add_argument("--ckpt-name", type=str, default=None, help="Checkppoint name prefix for experiment")
    parser.add_argument("--num-envs", type=int, default=max(1, multiprocessing.cpu_count()-1), help="Number of parallel environments to launch")
    args = parser.parse_args()

    algo = str(args.algo).lower().strip()
    no_fault = bool(args.no_fault)
    no_op = bool(args.no_op)
    pure = bool(args.pure)
    ft = bool(args.ft)
    eval_mode = bool(args.eval)
    exp_prefix = args.exp_name.strip() if args.exp_name else ("normal" if no_fault else "fault")
    ckpt_prefix = args.ckpt_name.strip() if args.ckpt_name else exp_prefix

    env_config.NUM_ENVS = max(1, int(args.num_envs))
    env_config.NO_OP = no_op
    env_config.PURE_RL = pure
    env_config.NO_FAULT = no_fault
    env_config.FINE_TUNE = ft
    env_config.EVAL = eval_mode

    if pure:
        exp_prefix += "_pure"
        ckpt_prefix += "_pure"
    if ft:
        exp_prefix += "_ft"
        ckpt_prefix += "_ft"
    # ensure ft only if pure
    assert not ft or pure, "Fine-tuning mode only applies if using pure RL (no residuals). Please set --pure if using --ft."
    #
    env_helpers.csv_path = f"{exp_prefix}-training_log.csv"
    env_helpers.csv_eps_log_path = f"{exp_prefix}-episode_returns.csv"
    eval_csv_path = f"eval_logs/{exp_prefix}-eval_log.csv"
    checkpoint_base_path = f"{ckpt_prefix}-{algo}_checkpoint"

    print(f"Launching {env_config.NUM_ENVS} parallel environments.")
    envs = gym.vector.AsyncVectorEnv([
        make_env(RWD_FN if not EVAL else 'eval', i, no_fault=no_fault, pure_rl=pure)
        for i in range(env_config.NUM_ENVS)
    ])
    agent_cls = PPOAgent if algo == "ppo" else TD3Agent
    agent: Any = agent_cls(num_envs=env_config.NUM_ENVS, checkpoint_path=checkpoint_base_path)
    print(f"GPU thread enabled: {GPU_THREAD}")
    print(f"Algorithm: {algo.upper()}")
    print(f"Reward function: {RWD_FN if not EVAL else 'eval'}")
    print(f"NO_FAULT: {no_fault}")
    print(f"Agent NO-OP mode: {env_config.NO_OP}")
    print(f"Residual mode: {not env_config.PURE_RL}")
    print(f"Fine-tuning mode: {env_config.FINE_TUNE}")
    print(f"Experiment prefix: {exp_prefix}")
    print(f"Training on device: {agent.device}, env action device: {'GPU' if agent.device.type == 'cuda' else 'CPU'}")

    if EVAL:
        print(f"EVAL mode enabled. Running {EVAL_EPISODES} evaluation episodes.")
        with open(eval_csv_path, mode='w', newline='') as eval_file:
            writer = csv.writer(eval_file)
            writer.writerow(['Episode', 'Env', 'Steps', 'Total Reward', 'Damaged Wheel', 'Fault Alpha', 'Success'])

            obs, _ = envs.reset()
            obs_t = torch.stack([agent.parse_obs(obs[i], env_id=i)[1] for i in range(env_config.NUM_ENVS)])
            episode_rewards = np.zeros(env_config.NUM_ENVS, dtype=float)
            episode_steps = np.zeros(env_config.NUM_ENVS, dtype=int)
            completed = 0
            success_count = 0

            while completed < EVAL_EPISODES:
                if env_config.NO_OP:
                    action = np.zeros((env_config.NUM_ENVS, _ACTION_DIM), dtype=np.float32)
                else:
                    action = agent.make_decision(obs_t, explore=False).detach().cpu().numpy()
                obs, reward, terminated, truncated, infos = envs.step(action)
                # input(f"step {episode_steps[0]}")
                next_obs_t = torch.stack([agent.parse_obs(obs[i], env_id=i)[1] for i in range(env_config.NUM_ENVS)])
                for i in range(env_config.NUM_ENVS):
                    if completed >= EVAL_EPISODES:
                        break
                    episode_rewards[i] += reward[i]
                    episode_steps[i] += 1
                    done = bool(terminated[i] or truncated[i])
                    if not done:
                        continue

                    fin_steps = extract_eps_info(infos, 'step', i, -1)
                    success = bool(extract_eps_info(infos, 'success', i, False))
                    damaged_wheel = extract_eps_info(infos, 'fault_wheel_idx', i, -1)
                    fault_alpha = extract_eps_info(infos, 'fault_alpha', i, np.nan)
                    completed += 1
                    success_count += int(success)

                    writer.writerow([
                        completed,
                        i,
                        fin_steps,
                        episode_rewards[i],
                        int(damaged_wheel) if damaged_wheel is not None else -1,
                        float(fault_alpha) if fault_alpha is not None else np.nan,
                        int(success)
                    ])

                    if completed % 50 == 0 or completed == EVAL_EPISODES:
                        print(
                            f"[EVAL] Episode {completed}/{EVAL_EPISODES} | env {i} | "
                            f"reward {episode_rewards[i]:.3f} | wheel {damaged_wheel} | "
                            f"alpha {float(fault_alpha):.2f} | success {success} | steps {fin_steps}"
                        )

                    episode_rewards[i] = 0.0
                    episode_steps[i] = 0
                    agent.hist_reset_single_env(i)
                    next_obs_t[i] = agent.parse_obs(obs[i], env_id=i)[1]

                obs_t = next_obs_t

        success_rate = success_count / max(1, EVAL_EPISODES)
        print(f"EVAL completed. Success rate: {success_count}/{EVAL_EPISODES} = {success_rate:.2%}")
        print(f"Eval log saved to: {eval_csv_path}")
        envs.close()
        return

    train_thread = None
    train_losses = [None]

    use_gpu_thread = GPU_THREAD and algo == "td3"
    if use_gpu_thread:
        def train_fn():
            loss = agent.train_step()
            # print_d(f"Training step completed with losses: {loss}")
            train_losses[0] = loss

    # use log to find last episode
    last_episode = 0
    try:
        reader = csv.reader(open(env_helpers.csv_eps_log_path, mode='r'))
        for row in reader:
            if row and row[0] != 'Episode':
                last_episode = max(last_episode, int(row[0]))
        episode = last_episode
    except Exception as e:
        print(e)
        print("No existing log file found, starting from episode 0.")
    
    global_step = 0
    try:
        reader = csv.reader(open(env_helpers.csv_path, mode='r'))
        for row in reader:
            if row and row[1] != 'Global step':
                global_step = max(global_step, int(row[1]))
    except Exception as e:
        print(e)
        print("No existing log file found, starting from global step 0.")
    episode_step = np.zeros(env_config.NUM_ENVS, dtype=int)
    episode_cnt = np.arange(env_config.NUM_ENVS, dtype=int) + last_episode + env_config.NUM_ENVS
    episode = episode_cnt.min()
    episode_rewards = np.zeros(env_config.NUM_ENVS, dtype=float)
    episode_discounted_rewards = np.zeros(env_config.NUM_ENVS, dtype=float)

    def estimate_metric(obs_state: torch.Tensor) -> float:
        if algo == "ppo":
            return agent.estimate_value(obs_state)
        with torch.no_grad():
            obs = obs_state.to(agent.device).unsqueeze(0)
            act = torch.clamp(agent.actor(obs), -agent.max_action, agent.max_action)
            assess = torch.cat([obs, act], dim=1)
            q1 = agent.critic1(assess)
            q2 = agent.critic2(assess)
            q = torch.min(q1, q2)
        return float(q.item())

    try:
        obs, _ = envs.reset() # stacked, all obs
        obs_t = torch.stack([agent.parse_obs(obs[i], env_id=i)[1] for i in range(env_config.NUM_ENVS)])
        critic_estimated_q = np.zeros(env_config.NUM_ENVS, dtype=float)
        
        # Benchmarking: report every 10 seconds
        bench_start = time.time()
        bench_last_check = bench_start
        bench_last_step = 0

        # per-step timing
        step_time = 0
        action_time = 0
        loop_time = 0
        bt0, bt1, bt2, bt3 = 0, 0, 0, 0
        bench_steps = 100
        
        while global_step < G_STEPS:
            # sync train thread
            if use_gpu_thread:
                if train_thread is not None:
                    train_thread.join()
                train_thread = threading.Thread(target=train_fn, daemon=True)
                train_thread.start()
            
            # make move
            if env_config.NO_OP:
                action = np.zeros((env_config.NUM_ENVS, _ACTION_DIM), dtype=np.float32)
            else:
                action = agent.make_decision(obs_t, explore=True, steps = global_step).detach().cpu().numpy()
            bt3 = time.perf_counter()
            if bt0 != 0:
                step_time += bt1 - bt0
                action_time += bt2 - bt1
                loop_time += bt3 - bt0
                bench_steps -= 1
                if bench_steps <= 0:
                    bench_steps = 100
                    # print_d(f"Per-step timing: {step_time/bench_steps:.4f}s step, {action_time/bench_steps:.4f}s action, {loop_time/bench_steps:.4f}s per loop")
                    step_time = 0
                    action_time = 0
                    loop_time = 0

            bt0 = time.perf_counter()
            obs, reward, terminated, truncated, _ = envs.step(action)
            bt1 = time.perf_counter()
            next_obs_t = torch.stack([agent.parse_obs(obs[i], env_id=i)[1] for i in range(env_config.NUM_ENVS)])
            # process transition, dones per env
            for i in range(env_config.NUM_ENVS):
                act = action[i]
                # print_d(f"Env {i} obs: errs {obs[i][0:5]}, \n\tcurr-wheels {obs[i][5:11]}, \n\tbase-wheels {obs[i][11:17]}, \n\t\tprev-dev {obs[i][17:]}")
                # print_d(f"Made deviation {act} in environment {i} at episode {episode}, step {episode_step[i]} -- base action {obs[i][-12:-6]}")
                done = terminated[i] or truncated[i]
                rwd = reward[i]
                episode_rewards[i] += reward[i]
                episode_discounted_rewards[i] += (DISCOUNT ** episode_step[i]) * reward[i]
                if algo == "ppo":
                    agent.save_transition(obs_t[i], act, rwd, done, next_obs_t[i], env_id=i)
                else:
                    agent.save_transition(obs_t[i], act, rwd, done, next_obs_t[i])
                episode_step[i] += 1
                if episode_step[i] == 100:
                    critic_estimated_q[i] = estimate_metric(obs_t[i])
                if done:        
                    eps_logging(
                        episode_cnt[i],
                        episode_step[i],
                        episode_rewards[i],
                        episode_discounted_rewards[i],
                        critic_estimated_q[i],
                    )
                    agent.hist_reset_single_env(i)
                    next_obs_t[i] = agent.parse_obs(obs[i], env_id=i)[1]
                    critic_estimated_q[i] = 0.0
                    print(f"Episode {episode_cnt[i]} finished in environment {i} with total reward {episode_rewards[i]:.3f} and discounted reward {episode_discounted_rewards[i]:.3f} after {episode_step[i]} steps.")    
                    episode_step[i] = 0
                    episode_cnt[i] += env_config.NUM_ENVS
                    episode_discounted_rewards[i] = 0.0
                    episode_rewards[i] = 0.0
            # train, log, update
            bt2 = time.perf_counter()
            losses = train_losses[0] if use_gpu_thread else agent.train_step()
            
            if (global_step > DECAY_INTERVAL and (global_step // DECAY_INTERVAL >  (global_step - env_config.NUM_ENVS) // DECAY_INTERVAL )):
                print(f"Episode {episode}, global step {global_step}, Losses: {losses}")
                if algo == "td3":
                    agent.decay_epsilon()
                    print(f"\tReplay buffer size: {len(agent.replay)}  |  Epsilon: {agent.epsilon:.3f}")
                else:
                    print(f"\tRollout size: {len(agent.rollout)} transitions")
                if losses and algo == "td3":
                    logging(episode, global_step, losses['critic1_loss'], losses['critic2_loss'], losses['actor_loss'])
                elif losses and algo == "ppo":
                    logging(episode, global_step, losses['value_loss'], losses['policy_loss'], losses['entropy'])
            obs_t = next_obs_t
            global_step += env_config.NUM_ENVS
            episode = episode_cnt.min()

            # Benchmark: every half min, report throughput
            now = time.time()
            if now - bench_last_check >= 30.0:
                elapsed = now - bench_last_check
                steps_delta = global_step - bench_last_step
                iter_per_sec = steps_delta / elapsed if elapsed > 0 else 0
                print(f"\t[BENCH] {iter_per_sec:.1f} iter/s")
                bench_last_check = now
                bench_last_step = global_step
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print("Exception during training:", e)
        raise e
    finally:
        print("Saving checkpoint...")
        agent.checkpoint_save(path=checkpoint_base_path)
        print("Checkpoint saved to:", checkpoint_base_path)
        envs.close()
    

if __name__ == "__main__":
    main()