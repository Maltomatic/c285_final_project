import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import copy

from controllers.models.actor import Actor
from controllers.models.critic import Critic

OBS_DIM = 23
ACT_DIM = 6


class Agent(nn.Module):
    def __init__(self, num_envs: int = 1):
        super().__init__()
        self.num_envs = num_envs
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state_dim  = 6 * 5 * 3 + 1 * 5 * 2   # matches original
        action_dim = ACT_DIM

        self.actor          = Actor(state_dim, action_dim).to(self.device)
        self.actor_target   = copy.deepcopy(self.actor)
        self.critic1        = Critic(state_dim, action_dim).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2        = Critic(state_dim, action_dim).to(self.device)
        self.critic2_target = copy.deepcopy(self.critic2)

        self.actor_optimizer   = torch.optim.Adam(self.actor.parameters(),   lr=1e-4)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=1e-3)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=1e-3)

        self.replay       = deque(maxlen=1_000_000)
        self.batch_size   = 512
        self.gamma        = 0.97
        self.tau          = 0.005
        self.policy_noise = 0.2
        self.noise_clip   = 0.5
        self.policy_freq  = 2
        self.max_action   = 5.0
        self.max_grad_norm = 2.0
        self.total_it     = 0
        self._last_actor_loss = None

        self.epsilon       = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min   = 0.05
        self._decay_countdown = 10

        # Per-environment observation history (list of deques, one per env)
        self._wheel_hist: list[deque] = []
        self._base_hist:  list[deque] = []
        self._dev_hist:   list[deque] = []
        self._vels_hist:  list[deque] = []
        self._ang_hist:   list[deque] = []
        self.hist_reset()   # initialises all envs

        self.checkpoint_load("td3_checkpoint")

    # ── History management ────────────────────────────────────────────────────

    def hist_reset(self):
        """Initialise / reinitialise history buffers for all environments."""
        self._wheel_hist = [deque(maxlen=5) for _ in range(self.num_envs)]
        self._base_hist  = [deque(maxlen=5) for _ in range(self.num_envs)]
        self._dev_hist   = [deque(maxlen=5) for _ in range(self.num_envs)]
        self._vels_hist  = [deque(maxlen=5) for _ in range(self.num_envs)]
        self._ang_hist   = [deque(maxlen=5) for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            self.reset_env_hist(i)

    def reset_env_hist(self, env_idx: int):
        """Reset history for a single environment (called on episode done)."""
        self._wheel_hist[env_idx].clear()
        self._base_hist[env_idx].clear()
        self._dev_hist[env_idx].clear()
        self._vels_hist[env_idx].clear()
        self._ang_hist[env_idx].clear()
        for _ in range(5):
            self._wheel_hist[env_idx].append(np.zeros(6,  dtype=np.float32))
            self._base_hist[env_idx].append(np.zeros(6,   dtype=np.float32))
            self._dev_hist[env_idx].append(np.zeros(6,    dtype=np.float32))
            self._vels_hist[env_idx].append(0.0)
            self._ang_hist[env_idx].append(0.0)

    # ── Observation parsing ───────────────────────────────────────────────────

    def parse_obs(self, obs: np.ndarray, env_idx: int = 0,
                  device: str | None = None) -> torch.Tensor:
        """
        Convert a raw 23-dim observation into a stacked state tensor
        for the given environment.
        """
        self._vels_hist[env_idx].append(float(obs[3]))
        self._ang_hist[env_idx].append(float(obs[4]))
        self._wheel_hist[env_idx].append(obs[5:11].astype(np.float32))
        self._base_hist[env_idx].append(obs[11:17].astype(np.float32))
        self._dev_hist[env_idx].append(obs[17:23].astype(np.float32))

        obs_arr = np.concatenate((
            np.array(self._vels_hist[env_idx], dtype=np.float32),
            np.array(self._ang_hist[env_idx],  dtype=np.float32),
            np.concatenate(list(self._wheel_hist[env_idx])),
            np.concatenate(list(self._base_hist[env_idx])),
            np.concatenate(list(self._dev_hist[env_idx])),
        ))

        target_device = self.device if device is None else torch.device(device)
        return torch.from_numpy(obs_arr).to(target_device)

    # ── Policy ────────────────────────────────────────────────────────────────

    def make_decision(self, obs_batch: torch.Tensor,
                      explore: bool = True) -> torch.Tensor:
        """
        obs_batch : (N, state_dim) tensor
        returns   : (N, ACT_DIM)  tensor
        """
        obs_batch = obs_batch.to(self.device)

        if explore and np.random.rand() < self.epsilon:
            action = torch.empty(
                (obs_batch.shape[0], ACT_DIM),
                dtype=torch.float32, device=self.device,
            ).uniform_(-self.max_action, self.max_action)
        else:
            with torch.no_grad():
                action = self.actor(obs_batch) * self.max_action
                action = torch.clamp(action, -self.max_action, self.max_action)

        if explore:
            self._decay_countdown -= 1
            if self._decay_countdown <= 0:
                self.epsilon = max(self.epsilon_min,
                                   self.epsilon * self.epsilon_decay)
                self._decay_countdown = 10

        return action

    # ── Replay buffer ─────────────────────────────────────────────────────────

    def store_transition(self, obs, action, reward, done, next_obs):
        """Push a single transition into the shared replay buffer."""
        def _to_np(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy().astype(np.float32)
            return np.asarray(x, dtype=np.float32)

        self.replay.append((
            _to_np(obs),
            _to_np(action),
            float(reward),
            float(done),
            _to_np(next_obs),
        ))

    # ── Training ──────────────────────────────────────────────────────────────

    def update(self) -> dict | None:
        """
        One gradient update step drawn from the pooled replay buffer.
        Call once per env step (after storing all N transitions).
        """
        if len(self.replay) < self.batch_size:
            return None

        self.total_it += 1

        # Sample
        idx   = np.random.choice(len(self.replay), size=self.batch_size, replace=False)
        batch = [self.replay[i] for i in idx]
        states      = torch.tensor(np.stack([b[0] for b in batch]),
                                   dtype=torch.float32, device=self.device)
        actions     = torch.tensor(np.stack([b[1] for b in batch]),
                                   dtype=torch.float32, device=self.device)
        rewards     = torch.tensor(np.array([b[2] for b in batch], dtype=np.float32),
                                   device=self.device).unsqueeze(1)
        dones       = torch.tensor(np.array([b[3] for b in batch], dtype=np.float32),
                                   device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.stack([b[4] for b in batch]),
                                   dtype=torch.float32, device=self.device)

        # TD3 critic targets
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_actions = (
                self.actor_target(next_states) * self.max_action + noise
            ).clamp(-self.max_action, self.max_action)

            next_sa   = torch.cat([next_states, next_actions], dim=1)
            target_q  = rewards + (1.0 - dones) * self.gamma * torch.min(
                self.critic1_target(next_sa),
                self.critic2_target(next_sa),
            )

        sa         = torch.cat([states, actions], dim=1)
        current_q1 = self.critic1(sa)
        current_q2 = self.critic2(sa)

        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), self.max_grad_norm)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        nn.utils.clip_grad_norm_(self.critic2.parameters(), self.max_grad_norm)
        self.critic2_optimizer.step()

        actor_loss = self._last_actor_loss
        if self.total_it % self.policy_freq == 0:
            pi        = self.actor(states) * self.max_action
            actor_loss = -self.critic1(torch.cat([states, pi], dim=1)).mean()
            self._last_actor_loss = actor_loss.detach().cpu()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            self._soft_update(self.actor,   self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

        return {
            "critic1_loss": float(critic1_loss.item()),
            "critic2_loss": float(critic2_loss.item()),
            "actor_loss":   float(actor_loss.item()) if actor_loss is not None else None,
        }

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def checkpoint_save(self, path: str):
        torch.save({
            "actor":              self.actor.state_dict(),
            "critic1":            self.critic1.state_dict(),
            "critic2":            self.critic2.state_dict(),
            "actor_target":       self.actor_target.state_dict(),
            "critic1_target":     self.critic1_target.state_dict(),
            "critic2_target":     self.critic2_target.state_dict(),
            "actor_optimizer":    self.actor_optimizer.state_dict(),
            "critic1_optimizer":  self.critic1_optimizer.state_dict(),
            "critic2_optimizer":  self.critic2_optimizer.state_dict(),
        }, path + ".pth")

        if input("Save replay buffer? (y/n): ").strip().lower() == "y":
            torch.save(list(self.replay), path + "_replay.pth")

    def checkpoint_load(self, path: str):
        try:
            ckpt = torch.load(path + ".pth", map_location=self.device)
            self.actor.load_state_dict(ckpt["actor"])
            self.critic1.load_state_dict(ckpt["critic1"])
            self.critic2.load_state_dict(ckpt["critic2"])
            self.actor_target.load_state_dict(ckpt["actor_target"])
            self.critic1_target.load_state_dict(ckpt["critic1_target"])
            self.critic2_target.load_state_dict(ckpt["critic2_target"])
            self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
            self.critic1_optimizer.load_state_dict(ckpt["critic1_optimizer"])
            self.critic2_optimizer.load_state_dict(ckpt["critic2_optimizer"])
            for opt in (self.actor_optimizer,
                        self.critic1_optimizer,
                        self.critic2_optimizer):
                self._optimizer_to_device(opt)
            print(f"Checkpoint loaded from {path}.pth")
        except FileNotFoundError:
            print(f"No checkpoint at {path}.pth — starting fresh.")

        try:
            replay_data = torch.load(path + "_replay.pth",
                                     map_location="cpu", weights_only=False)
            self.replay = deque(replay_data, maxlen=1_000_000)
            print(f"Replay buffer loaded ({len(self.replay):,} transitions).")
        except FileNotFoundError:
            print("No replay buffer found — starting empty.")

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _soft_update(self, source: nn.Module, target: nn.Module):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    def _optimizer_to_device(self, optimizer):
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
