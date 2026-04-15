"""
SAC Agent for Infinite Valley Navigation Environment
=====================================================
Wraps InfiniteValleyEnv in a Gymnasium-compatible adapter and trains
a Soft Actor-Critic (SAC) agent to navigate from (0,0) to (60,60)
across infinite periodic terrain with gravity, friction, and wind.
"""

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

import os
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from infinite_valley_env import InfiniteValleyEnv


# ─────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────

class SaveVideoEveryNEpisodesCallback(BaseCallback):
    """Save an MP4 rollout video every N episodes."""

    def __init__(
        self,
        make_env_fn,
        save_dir: str = "animations",
        every_episodes: int = 200,
        rollout_max_steps: int = 3000,
        fps: int = 30,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.make_env_fn = make_env_fn
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.every_episodes = int(every_episodes)
        self.rollout_max_steps = int(rollout_max_steps)
        self.fps = int(fps)
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if isinstance(infos, (list, tuple)) and len(infos) > 0:
            info0 = infos[0]
            if isinstance(info0, dict) and "episode" in info0:
                self.episode_count += 1
                if self.every_episodes > 0 and (self.episode_count % self.every_episodes) == 0:
                    self._save_video(self.episode_count)
        return True

    def _save_video(self, episode_idx: int) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.save_dir / f"valley_ep_{episode_idx:07d}.mp4"

        eval_env = self.make_env_fn()
        try:
            obs, _ = eval_env.reset()
            eval_env.render()

            fig = plt.figure(1) if plt.fignum_exists(1) else plt.gcf()

            try:
                writer = FFMpegWriter(fps=self.fps, codec="libx264")
            except Exception as e:
                if self.verbose:
                    print(f"[SaveVideo] ffmpeg not available ({e}); skipping.")
                return

            try:
                with writer.saving(fig, str(out_path), dpi=100):
                    for _ in range(self.rollout_max_steps):
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = eval_env.step(action)
                        eval_env.render()
                        fig.canvas.draw()
                        writer.grab_frame()
                        if terminated or truncated:
                            break
                if self.verbose:
                    print(f"[SaveVideo] Saved {out_path}")
            except Exception as e:
                if self.verbose:
                    print(f"[SaveVideo] Failed ({e})")
        finally:
            try:
                eval_env.close()
            except Exception:
                pass


class TensorboardEpisodeStatsCallback(BaseCallback):
    """Logs per-episode reward / length / success to TensorBoard."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not isinstance(infos, (list, tuple)):
            return True
        for info in infos:
            if not isinstance(info, dict):
                continue
            ep_info = info.get("episode")
            if not isinstance(ep_info, dict):
                continue
            self.episode_count += 1
            if "r" in ep_info:
                self.logger.record("episode/reward", float(ep_info["r"]), exclude=("stdout",))
            if "l" in ep_info:
                self.logger.record("episode/length", float(ep_info["l"]), exclude=("stdout",))
            self.logger.record("episode/index", float(self.episode_count), exclude=("stdout",))
            self.logger.dump(step=self.num_timesteps)
        return True


# ─────────────────────────────────────────────────────────────────────
# Gymnasium Adapter for InfiniteValleyEnv
# ─────────────────────────────────────────────────────────────────────

class InfiniteValleyGymAdapter(gym.Env):
    """
    Wraps InfiniteValleyEnv to conform to the Gymnasium API.

    Observation (8-dim):
        [x_norm, y_norm, vx_norm, vy_norm, dx_goal, dy_goal, dist_goal, heading_to_goal]
    where dx/dy_goal are relative goal offsets normalised by initial distance,
    dist_goal is euclidean distance (normalised), and heading_to_goal is the angle.

    Action (2-dim): [ax, ay] in [-1, 1]  (native range of InfiniteValleyEnv)
    """

    metadata = {"render_modes": []}

    def __init__(self, env: InfiniteValleyEnv | None = None):
        super().__init__()
        self.env = env or InfiniteValleyEnv()

        # Normalise positions by initial goal distance (~84.85)
        self._norm = np.linalg.norm(self.env.goal - self.env.start) + 1e-8

        # Observation bounds (generous)
        obs_low = np.array(
            [-3.0, -3.0, -1.0, -1.0, -3.0, -3.0, 0.0, -np.pi],
            dtype=np.float32,
        )
        obs_high = np.array(
            [3.0, 3.0, 1.0, 1.0, 3.0, 3.0, 3.0, np.pi],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Actions are already [-1, 1] in InfiniteValleyEnv
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Internal bookkeeping
        self._prev_dist = None
        self._episode_steps = 0
        self._total_steps = 0
        self._episode_count = 0
        self._successes = 0
        self._best_steps = None
        self._first_solve_step = None

        # CSV log
        self.episode_log_file = "episode_steps.csv"
        if not os.path.exists(self.episode_log_file):
            with open(self.episode_log_file, "w") as f:
                f.write("episode,steps,success,distance\n")

    # ── helpers ──────────────────────────────────────────────────────

    def _build_obs(self, raw_state: np.ndarray) -> np.ndarray:
        """Build normalised observation from raw [x, y, vx, vy]."""
        x, y, vx, vy = raw_state

        gx, gy = self.env.goal
        dx = (gx - x) / self._norm
        dy = (gy - y) / self._norm
        dist = np.sqrt(dx ** 2 + dy ** 2)
        heading = np.arctan2(dy, dx)

        obs = np.array(
            [
                x / self._norm,
                y / self._norm,
                vx / self.env.v_max,
                vy / self.env.v_max,
                dx,
                dy,
                dist,
                heading,
            ],
            dtype=np.float32,
        )
        obs = np.nan_to_num(obs, nan=0.0, posinf=3.0, neginf=-3.0)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs

    # ── Gymnasium API ────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Log previous episode
        if self._episode_count > 0 and self._prev_dist is not None:
            with open(self.episode_log_file, "a") as f:
                f.write(
                    f"{self._episode_count},{self._episode_steps},"
                    f"{int(self._prev_dist < self.env.goal_radius)},"
                    f"{self._prev_dist:.2f}\n"
                )

        self._episode_count += 1
        self._episode_steps = 0

        raw = self.env.reset()
        self._prev_dist = np.linalg.norm(raw[:2] - self.env.goal)
        return self._build_obs(raw), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(2)
        action = np.clip(action, -1.0, 1.0)

        raw, base_reward, done, info = self.env.step(action)
        self._episode_steps += 1
        self._total_steps += 1

        dist = np.linalg.norm(raw[:2] - self.env.goal)

        # ── Reward shaping ──────────────────────────────────────────
        # 1) Dense approach reward (progress toward goal)
        progress = self._prev_dist - dist
        shaped_reward = 5.0 * progress  # strong dense signal

        # 2) Small step penalty to encourage speed
        shaped_reward -= 0.005

        # 3) Large bonus on success (on top of env's +100)
        success = info.get("success", False)
        if success:
            shaped_reward += 200.0  # extra bonus
            # Time bonus: fewer steps → bigger bonus
            shaped_reward += max(0, 3000 - self._episode_steps) * 0.05

        self._prev_dist = dist

        terminated = bool(done)
        truncated = False

        total_reward = base_reward + shaped_reward

        # Track statistics
        if success:
            self._successes += 1
            if self._best_steps is None or self._episode_steps < self._best_steps:
                self._best_steps = self._episode_steps
            if self._first_solve_step is None:
                self._first_solve_step = self._total_steps

        gym_info = {
            "success": success,
            "distance": dist,
            "episode_steps": self._episode_steps,
        }

        return self._build_obs(raw), float(total_reward), terminated, truncated, gym_info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


# ─────────────────────────────────────────────────────────────────────
# SAC Agent
# ─────────────────────────────────────────────────────────────────────

class SoftActorCriticAgent:
    def __init__(self):
        self.env = self._make_env()

    @staticmethod
    def _make_env():
        raw_env = InfiniteValleyEnv()
        gym_env = InfiniteValleyGymAdapter(raw_env)
        gym_env = TimeLimit(gym_env, max_episode_steps=3000)
        return Monitor(gym_env)

    def train(self, total_timesteps: int = 300_000):
        device = os.environ.get("SB3_DEVICE", "auto").strip().lower()
        if device in {"auto", ""}:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[SAC] Using device: {device}")

        video_cb = SaveVideoEveryNEpisodesCallback(
            make_env_fn=self._make_env,
            save_dir="animations",
            every_episodes=200,
            rollout_max_steps=3000,
            fps=30,
            verbose=1,
        )
        tb_cb = TensorboardEpisodeStatsCallback(verbose=0)

        model = SAC(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=3e-4,
            buffer_size=500_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto",
            train_freq=1,
            gradient_steps=1,
            learning_starts=1000,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
            device=device,
        )

        model.learn(
            total_timesteps=total_timesteps,
            tb_log_name="SAC_InfiniteValley",
            log_interval=10,
            callback=[video_cb, tb_cb],
        )

        # Save model with timestamp
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sac_infinite_valley_{ts}"
        model.save(filename)
        print(f"Model saved as: {filename}.zip")

        return model


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    agent = SoftActorCriticAgent()
    model = agent.train(total_timesteps=300_000)

    print("\n" + "=" * 60)
    print("Training complete.")
    print("=" * 60)

    # Unwrap to the adapter: Monitor -> TimeLimit -> InfiniteValleyGymAdapter
    adapter: InfiniteValleyGymAdapter = agent.env.env.env  # type: ignore[assignment]

    print(f"Total environment steps:       {adapter._total_steps}")
    print(f"Total episodes:                {adapter._episode_count}")
    print(f"Successful episodes:           {adapter._successes}")
    print(f"First solve at global step:    {adapter._first_solve_step}")
    print(f"Best (min) steps to solve:     {adapter._best_steps}")

    # Log final episode
    if adapter._episode_count > 0:
        with open(adapter.episode_log_file, "a") as f:
            f.write(
                f"{adapter._episode_count},{adapter._episode_steps},"
                f"{0},{adapter._prev_dist:.2f}\n"
            )

    # Find episode with minimum steps among solved episodes
    try:
        with open(adapter.episode_log_file, "r") as f:
            lines = f.readlines()[1:]  # skip header
            if lines:
                solved = [
                    (int(l.split(",")[0]), int(l.split(",")[1]))
                    for l in lines
                    if int(l.split(",")[2]) == 1
                ]
                if solved:
                    best = min(solved, key=lambda t: t[1])
                    print(f"Best solved episode:  Episode {best[0]} in {best[1]} steps")
    except Exception as e:
        print(f"Could not read episode log: {e}")

    # Save summary
    with open("training_results.txt", "w") as f:
        f.write(f"Total environment steps:       {adapter._total_steps}\n")
        f.write(f"Total episodes:                {adapter._episode_count}\n")
        f.write(f"Successful episodes:           {adapter._successes}\n")
        f.write(f"First solve at global step:    {adapter._first_solve_step}\n")
        f.write(f"Best (min) steps to solve:     {adapter._best_steps}\n")
    print("Results saved to training_results.txt")
    print(f"Episode details saved to {adapter.episode_log_file}")
