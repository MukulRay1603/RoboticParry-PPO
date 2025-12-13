
"""
Training script for SamuraiRL PPO agent.

Run with (inside your samurai_rl venv):

    samurai_rl\Scripts\python.exe train_samurai.py

Make sure you have installed:
    gymnasium==1.2.2
    stable-baselines3[extra]==2.2.1
    pybullet
    torch==2.5.1+cu121 (and matching torchvision/torchaudio)
"""

import os
import argparse
import time

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from samurai_env import SamuraiParryEnv


class ProgressCallback(BaseCallback):
    """Simple textual progress bar with elapsed time and ETA."""

    def __init__(self, total_timesteps: int, log_interval: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.log_interval = log_interval
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.start_time is None:
            return True

        if self.num_timesteps % self.log_interval == 0 or self.num_timesteps == self.total_timesteps:
            elapsed = time.time() - self.start_time
            frac = min(1.0, float(self.num_timesteps) / float(self.total_timesteps))
            eta = (elapsed / frac) - elapsed if frac > 0 else float("inf")
            if self.verbose > 0:
                print(
                    f"[PROGRESS] {self.num_timesteps:,}/{self.total_timesteps:,} "
                    f"({frac*100:5.1f}%) | elapsed {elapsed/60:5.1f} min | "
                    f"ETA {eta/60:5.1f} min"
                )
        return True



def make_env_fn():
    return SamuraiParryEnv(render_mode=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--save-path", type=str, default="samurai_ppo")
    args = parser.parse_args()

    # Wrap in VecEnv for SB3
    env = DummyVecEnv([make_env_fn])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cuda",  # your RTX 3080
        tensorboard_log="./tb_samurai/",
    )

    callback = ProgressCallback(args.timesteps)
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(args.save_path)

    env.close()
    print(f"Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
