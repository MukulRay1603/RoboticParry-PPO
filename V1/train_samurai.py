
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

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from samurai_env import SamuraiParryEnv


def make_env_fn():
    return SamuraiParryEnv(render_mode=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
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

    model.learn(total_timesteps=args.timesteps)
    model.save(args.save_path)

    env.close()
    print(f"Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
