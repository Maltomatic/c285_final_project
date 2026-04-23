from collections import deque
import copy

import numpy as np
import torch
import torch.nn as nn

from envs.env_configs import STACK_OBS_DIM, _ACTION_DIM as ACT_DIM, _ACTION_CLIP


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.policy_mean = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        self.value = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

    def dist(self, obs: torch.Tensor):
        mean = self.policy_mean(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)


class RolloutBuffer:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.reset()

    def reset(self):
        self.obs = []
        self.actions = []
        self.logprobs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.next_values = []

    def add(self, obs, action, logprob, value, reward, done, next_value):
        self.obs.append(obs)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_values.append(next_value)

    def __len__(self):
        return len(self.obs)


class Agent(nn.Module):
    def __init__(self, num_envs: int = 1, checkpoint_path: str = "ppo_checkpoint"):
        super().__init__()
        self.num_envs = num_envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_action = float(_ACTION_CLIP)

        self.model = ActorCritic(STACK_OBS_DIM, ACT_DIM).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_coef = 0.2
        self.ent_coef = 0.005
        self.vf_coef = 0.5
        self.max_grad_norm = 1.0
        self.update_epochs = 6
        self.minibatch_size = 4096
        self.rollout_steps = 256

        self.rollout = RolloutBuffer(num_envs)
        self._step_cache = None
        self._total_updates = 0

        # Kept for compatibility with existing logging path in main.
        self.epsilon = 0.0

        self.wheel_hist = [deque(maxlen=5) for _ in range(num_envs)]
        self.base_hist = [deque(maxlen=5) for _ in range(num_envs)]
        self.dev_hist = [deque(maxlen=5) for _ in range(num_envs)]
        self.vels_hist = [deque(maxlen=5) for _ in range(num_envs)]
        self.ang_hist = [deque(maxlen=5) for _ in range(num_envs)]
        self.hist_reset()

        self.checkpoint_load(checkpoint_path)
        self.printout()

    def printout(self):
        print("PPO Agent initialized with:")
        print(f"\tDevice: {self.device}")
        print(f"\tGamma: {self.gamma}")
        print(f"\tGAE lambda: {self.gae_lambda}")
        print(f"\tClip coefficient: {self.clip_coef}")
        print(f"\tEntropy coefficient: {self.ent_coef}")
        print(f"\tValue loss coefficient: {self.vf_coef}")
        print(f"\tUpdate epochs: {self.update_epochs}")
        print(f"\tRollout steps/env: {self.rollout_steps}")

    def hist_reset(self):
        for env_id in range(self.num_envs):
            self.hist_reset_single_env(env_id)

    def hist_reset_single_env(self, env_id):
        self.wheel_hist[env_id].clear()
        self.base_hist[env_id].clear()
        self.dev_hist[env_id].clear()
        self.vels_hist[env_id].clear()
        self.ang_hist[env_id].clear()
        for _ in range(5):
            self.wheel_hist[env_id].append(np.zeros(6, dtype=np.float32))
            self.base_hist[env_id].append(np.zeros(6, dtype=np.float32))
            self.dev_hist[env_id].append(np.zeros(6, dtype=np.float32))
            self.vels_hist[env_id].append(0.0)
            self.ang_hist[env_id].append(0.0)

    def parse_obs(self, obs, device=None, env_id=0):
        obs_dict = {}
        obs_dict["cross_track_err"] = obs[0]
        obs_dict["heading_err"] = obs[1]
        obs_dict["waypoint_dist"] = obs[2]
        obs_dict["bot_vel"] = obs[3]
        self.vels_hist[env_id].append(obs[3])
        obs_dict["bot_ang"] = obs[4]
        self.ang_hist[env_id].append(obs[4])
        obs_dict["curr_wheel"] = obs[5:11]
        self.wheel_hist[env_id].append(obs[5:11])
        obs_dict["curr_base"] = obs[11:17]
        self.base_hist[env_id].append(obs[11:17])
        obs_dict["curr_dev"] = obs[17:23]
        self.dev_hist[env_id].append(obs[17:23])

        _vels_hist = np.array(self.vels_hist[env_id])
        _ang_hist = np.array(self.ang_hist[env_id])
        _wheel_hist = np.concatenate(list(self.wheel_hist[env_id]))
        _base_hist = np.concatenate(list(self.base_hist[env_id]))
        _dev_hist = np.concatenate(list(self.dev_hist[env_id]))

        obs_arr = np.concatenate((_vels_hist, _ang_hist, _wheel_hist, _base_hist, _dev_hist)).astype(np.float32)
        target_device = self.device if device is None else torch.device(device)
        obs_t = torch.from_numpy(obs_arr).to(target_device)

        return obs_dict, obs_t

    def make_decision(self, obs, explore: bool = True, env_id=0):
        if type(obs) is not torch.Tensor:
            obs_t = torch.stack([self.parse_obs(obs[i], env_id=i)[1] for i in range(self.num_envs)])
        else:
            obs_t = obs.to(self.device)

        with torch.no_grad():
            dist = self.model.dist(obs_t)
            value = self.model.value(obs_t).squeeze(-1)
            if explore:
                action = dist.sample()
            else:
                action = dist.mean
            action = torch.clamp(action, -self.max_action, self.max_action)
            logprob = dist.log_prob(action).sum(dim=-1)

        # Cache per-env policy outputs for the transition that will be saved after env.step.
        self._step_cache = {
            "obs": obs_t.detach().cpu().numpy(),
            "actions": action.detach().cpu().numpy(),
            "logprobs": logprob.detach().cpu().numpy(),
            "values": value.detach().cpu().numpy(),
        }
        return action

    def save_transition(self, obs, action, reward, done, next_obs, env_id=0):
        if self._step_cache is None:
            return

        obs_np = self._step_cache["obs"][env_id]
        action_np = self._step_cache["actions"][env_id]
        logprob = float(self._step_cache["logprobs"][env_id])
        value = float(self._step_cache["values"][env_id])

        next_obs_t = next_obs.to(self.device).unsqueeze(0) if isinstance(next_obs, torch.Tensor) else torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            next_value = float(self.model.value(next_obs_t).squeeze(-1).item())

        self.rollout.add(
            obs=obs_np,
            action=action_np,
            logprob=logprob,
            value=value,
            reward=float(reward),
            done=float(done),
            next_value=next_value,
        )

    def decay_epsilon(self):
        return

    def estimate_value(self, obs_state: torch.Tensor) -> float:
        with torch.no_grad():
            obs = obs_state.to(self.device).unsqueeze(0)
            value = self.model.value(obs)
        return float(value.item())

    def train_step(self):
        needed = self.rollout_steps * self.num_envs
        if len(self.rollout) < needed:
            return None

        obs = torch.as_tensor(np.asarray(self.rollout.obs, dtype=np.float32), device=self.device)
        actions = torch.as_tensor(np.asarray(self.rollout.actions, dtype=np.float32), device=self.device)
        old_logprobs = torch.as_tensor(np.asarray(self.rollout.logprobs, dtype=np.float32), device=self.device)
        values = torch.as_tensor(np.asarray(self.rollout.values, dtype=np.float32), device=self.device)
        rewards = torch.as_tensor(np.asarray(self.rollout.rewards, dtype=np.float32), device=self.device)
        dones = torch.as_tensor(np.asarray(self.rollout.dones, dtype=np.float32), device=self.device)
        next_values = torch.as_tensor(np.asarray(self.rollout.next_values, dtype=np.float32), device=self.device)

        T = len(self.rollout) // self.num_envs
        obs = obs.view(T, self.num_envs, -1)
        actions = actions.view(T, self.num_envs, -1)
        old_logprobs = old_logprobs.view(T, self.num_envs)
        values = values.view(T, self.num_envs)
        rewards = rewards.view(T, self.num_envs)
        dones = dones.view(T, self.num_envs)
        next_values = next_values.view(T, self.num_envs)

        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(self.num_envs, device=self.device)
        for t in range(T - 1, -1, -1):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_values[t] * nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            advantages[t] = gae
        returns = advantages + values

        obs_b = obs.reshape(T * self.num_envs, -1)
        actions_b = actions.reshape(T * self.num_envs, -1)
        old_logprobs_b = old_logprobs.reshape(T * self.num_envs)
        returns_b = returns.reshape(T * self.num_envs)
        advantages_b = advantages.reshape(T * self.num_envs)
        advantages_b = (advantages_b - advantages_b.mean()) / (advantages_b.std(unbiased=False) + 1e-8)

        n = obs_b.shape[0]
        batch_size = min(self.minibatch_size, n)

        policy_loss_val = 0.0
        value_loss_val = 0.0
        entropy_val = 0.0
        update_steps = 0

        for _ in range(self.update_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                mb_obs = obs_b[idx]
                mb_actions = actions_b[idx]
                mb_old_logprobs = old_logprobs_b[idx]
                mb_returns = returns_b[idx]
                mb_advantages = advantages_b[idx]

                dist = self.model.dist(mb_obs)
                new_logprobs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                new_values = self.model.value(mb_obs).squeeze(-1)

                ratio = torch.exp(new_logprobs - mb_old_logprobs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (mb_returns - new_values).pow(2).mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_loss_val += float(policy_loss.item())
                value_loss_val += float(value_loss.item())
                entropy_val += float(entropy.item())
                update_steps += 1

        self._total_updates += 1
        self.rollout.reset()

        denom = max(1, update_steps)
        return {
            "policy_loss": policy_loss_val / denom,
            "value_loss": value_loss_val / denom,
            "entropy": entropy_val / denom,
            "updates": self._total_updates,
        }

    def checkpoint_save(self, path):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "updates": self._total_updates,
            },
            path + ".pth",
        )

    def checkpoint_load(self, path):
        ckpt_path = path + ".pth"
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self._total_updates = int(checkpoint.get("updates", 0))
            self._optimizer_to_device(self.optimizer)
        except FileNotFoundError:
            print(f"No checkpoint found at {ckpt_path}. Starting with uninitialized model.")

    def _optimizer_to_device(self, optimizer):
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
