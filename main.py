import argparse
import numpy as np
import torch, envs, gymnasium as gym
from envs.six_wheel_env import SixWheelEnv
from controllers.TD3_controller import Agent
import time, csv, multiprocessing, threading

from controllers.utils.model_configs import DECAY_INTERVAL, DISCOUNT, G_STEPS, RWD_FN
from envs.utils.env_helpers import logging, eps_logging,\
    csv_path, csv_eps_log_path, checkpoint_base_path,\
    make_env, extract_eps_info
from envs.env_configs import DEBUG, EVAL, EVAL_EPISODES, GPU_THREAD, NO_OP, _ACTION_DIM

def print_d(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

def main():
    parser = argparse.ArgumentParser(description="Train TD3 with parallel environments")
    parser.add_argument("--no-fault", action="store_true", help="Disable injected wheel faults (NO_FAULT mode)")
    parser.add_argument(
        "--no-op",
        dest="no_op",
        action="store_true",
        default=False,
        help="Disable agent actions and use zero actions instead",
    )
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment prefix for logs/checkpoints")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=max(1, multiprocessing.cpu_count()-1),
        help="Number of parallel environments to launch",
    )
    args = parser.parse_args()

    no_fault = bool(args.no_fault)
    no_op = bool(args.no_op)
    exp_prefix = args.exp_name.strip() if args.exp_name else ("normal" if no_fault else "fault")

    global NUM_ENVS, csv_path, csv_eps_log_path, checkpoint_base_path, NO_OP

    NUM_ENVS = max(1, int(args.num_envs))
    NO_OP = no_op
    csv_path = f"{exp_prefix}-training_log.csv"
    csv_eps_log_path = f"{exp_prefix}-episode_returns.csv"
    checkpoint_base_path = f"{exp_prefix}-td3_checkpoint"
    eval_csv_path = f"{exp_prefix}-eval_log.csv"

    print(f"Launching {NUM_ENVS} parallel environments.")
    envs = gym.vector.AsyncVectorEnv([make_env(RWD_FN, i, no_fault=no_fault) for i in range(NUM_ENVS)])
    agent = Agent(num_envs=NUM_ENVS, checkpoint_path=checkpoint_base_path)
    print(f"GPU thread enabled: {GPU_THREAD}")
    print(f"Reward function: {RWD_FN if not EVAL else 'eval'}")
    print(f"NO_FAULT: {no_fault}")
    print(f"Agent NO-OP mode: {NO_OP}")
    print(f"Experiment prefix: {exp_prefix}")
    print(f"Training on device: {agent.device}, env action device: {'GPU' if agent.device.type == 'cuda' else 'CPU'}")

    if EVAL:
        print(f"EVAL mode enabled. Running {EVAL_EPISODES} evaluation episodes.")
        with open(eval_csv_path, mode='w', newline='') as eval_file:
            writer = csv.writer(eval_file)
            writer.writerow(['Episode', 'Env', 'Steps', 'Total Reward', 'Damaged Wheel', 'Fault Alpha', 'Success'])

            obs, _ = envs.reset()
            obs_t = torch.stack([agent.parse_obs(obs[i], env_id=i)[1] for i in range(NUM_ENVS)])
            episode_rewards = np.zeros(NUM_ENVS, dtype=float)
            episode_steps = np.zeros(NUM_ENVS, dtype=int)
            completed = 0
            success_count = 0

            while completed < EVAL_EPISODES:
                if NO_OP:
                    action = np.zeros((NUM_ENVS, _ACTION_DIM), dtype=np.float32)
                else:
                    action = agent.make_decision(obs_t, explore=False).detach().cpu().numpy()
                obs, reward, terminated, truncated, infos = envs.step(action)
                next_obs_t = torch.stack([agent.parse_obs(obs[i], env_id=i)[1] for i in range(NUM_ENVS)])

                for i in range(NUM_ENVS):
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

    if GPU_THREAD:
        def train_fn():
            loss = agent.train_step()
            # print_d(f"Training step completed with losses: {loss}")
            train_losses[0] = loss

    # use log to find last episode
    last_episode = 0
    try:
        reader = csv.reader(open(csv_eps_log_path, mode='r'))
        for row in reader:
            if row and row[0] != 'Episode':
                last_episode = max(last_episode, int(row[0]))
        episode = last_episode
    except Exception as e:
        print(e)
        print("No existing log file found, starting from episode 0.")
    
    global_step = 0
    try:
        reader = csv.reader(open(csv_path, mode='r'))
        for row in reader:
            if row and row[1] != 'Global step':
                global_step = max(global_step, int(row[1]))
    except Exception as e:
        print(e)
        print("No existing log file found, starting from global step 0.")
    episode_step = np.zeros(NUM_ENVS, dtype=int)
    episode_cnt = np.arange(NUM_ENVS, dtype=int) + last_episode + NUM_ENVS
    episode = episode_cnt.min()
    episode_rewards = np.zeros(NUM_ENVS, dtype=float)
    episode_discounted_rewards = np.zeros(NUM_ENVS, dtype=float)

    def estimate_q(obs_state: torch.Tensor) -> float:
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
        obs_t = torch.stack([agent.parse_obs(obs[i], env_id=i)[1] for i in range(NUM_ENVS)])
        critic_estimated_q = np.zeros(NUM_ENVS, dtype=float)
        
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
            if GPU_THREAD:
                if train_thread is not None:
                    train_thread.join()
                train_thread = threading.Thread(target=train_fn, daemon=True)
                train_thread.start()
            
            # make move
            if NO_OP:
                action = np.zeros((NUM_ENVS, _ACTION_DIM), dtype=np.float32)
            else:
                action = agent.make_decision(obs_t, explore=True).detach().cpu().numpy()
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
            next_obs_t = torch.stack([agent.parse_obs(obs[i], env_id=i)[1] for i in range(NUM_ENVS)])
            # process transition, dones per env
            for i in range(NUM_ENVS):
                act = action[i]
                # print_d(f"Env {i} obs: errs {obs[i][0:5]}, \n\tcurr-wheels {obs[i][5:11]}, \n\tbase-wheels {obs[i][11:17]}, \n\t\tprev-dev {obs[i][17:]}")
                # print_d(f"Made deviation {act} in environment {i} at episode {episode}, step {episode_step[i]} -- base action {obs[i][-12:-6]}")
                done = terminated[i] or truncated[i]
                rwd = reward[i]
                episode_rewards[i] += reward[i]
                episode_discounted_rewards[i] += (DISCOUNT ** episode_step[i]) * reward[i]
                agent.save_transition(obs_t[i], act, rwd, done, next_obs_t[i])
                episode_step[i] += 1
                if episode_step[i] == 100:
                    critic_estimated_q[i] = estimate_q(obs_t[i])
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
                    episode_cnt[i] += NUM_ENVS
                    episode_discounted_rewards[i] = 0.0
                    episode_rewards[i] = 0.0
            # train, log, update
            bt2 = time.perf_counter()
            losses = train_losses[0] if GPU_THREAD else agent.train_step()
            
            if (global_step > DECAY_INTERVAL and (global_step // DECAY_INTERVAL >  (global_step - NUM_ENVS) // DECAY_INTERVAL )):
                # print("eps decay")
                agent.decay_epsilon()
                print(f"Episode {episode}, global step {global_step}, Losses: {losses}")
                print(f"\tReplay buffer size: {len(agent.replay)}  |  Epsilon: {agent.epsilon:.3f}")
                if losses:
                    logging(episode, global_step, losses['critic1_loss'], losses['critic2_loss'], losses['actor_loss'])
            obs_t = next_obs_t
            global_step += NUM_ENVS
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
    finally:
        print("Saving checkpoint...")
        agent.checkpoint_save(path=checkpoint_base_path)    
        envs.close()
    

if __name__ == "__main__":
    main()