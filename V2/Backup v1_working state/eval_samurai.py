"""
Simple evaluation / visualization script using PyBullet GUI.

Run with (after training):

    samurai_rl\Scripts\python.exe eval_samurai.py --model-path samurai_ppo.zip

You can also run it BEFORE training; if the model file is missing,
the agent will just use random actions so you can sanity-check
that the PyBullet scene and swords work without crashing.

By default this script runs episodes in an endless loop so that the
PyBullet window stays open until you close it or press Ctrl+C.
"""

import argparse
import os
import time

import numpy as np
from stable_baselines3 import PPO

from samurai_env import SamuraiParryEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="samurai_ppo.zip")
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Number of episodes to run (0 = run forever until Ctrl+C).",
    )
    args = parser.parse_args()

    env = SamuraiParryEnv(render_mode="human")

    use_model = False
    model = None

    if os.path.exists(args.model_path):
        print(f"[INFO] Loading model from: {args.model_path}")
        model = PPO.load(args.model_path, env=env, device="cuda")
        use_model = True
    else:
        print(
            f"[WARN] Model file '{args.model_path}' not found.\n"
            f"       Running with RANDOM actions for sanity-check."
        )

    ep = 0
    try:
        while True:
            if args.episodes > 0 and ep >= args.episodes:
                break

            obs, info = env.reset()
            terminated = False
            truncated = False
            ep_reward = 0.0

            prev_parry = False
            prev_hit = False

            while not (terminated or truncated):
                if use_model:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward

                # Edge-triggered logging so we don't spam when contact persists
                parry = bool(info.get("contact_parry"))
                hit = bool(info.get("contact_body_hit"))
                if parry and not prev_parry:
                    print("[EVENT] Parry registered.")
                if hit and not prev_hit:
                    print("[EVENT] Agent was hit!")

                prev_parry = parry
                prev_hit = hit

                # Small sleep so GUI doesn't run too fast
                time.sleep(1.0 / 60.0)

            ep += 1
            print(f"Episode {ep} reward: {ep_reward:.2f}")
            time.sleep(0.25)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user, closing environment.")

    env.close()


if __name__ == "__main__":
    main()
