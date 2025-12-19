"""
Simple evaluation / visualization script using PyBullet GUI.

Now supports interactive mode selection:
1 - Dry Run (no PPO model, local reward_engine.py)
2 - Steady Defence (loads reward_engine.py + PPO from STEADY GUARD/)
3 - Evasion (loads reward_engine.py + PPO from EVASION DEFENCE/)

No other logic is changed.
"""

import argparse
import os
import sys
import time

import numpy as np
from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Number of episodes to run (0 = run forever until Ctrl+C).",
    )
    args = parser.parse_args()

    # ---------------- MODE SELECTION ----------------
    print("\nSelect Mode:")
    print("1 - Dry Run (no model)")
    print("2 - Steady Defence")
    print("3 - Evasion")

    choice = input("Enter 1/2/3: ").strip()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    mode_dir = None
    model_path = None

    if choice == "1":
        print("[INFO] Dry Run selected.")
    elif choice == "2":
        mode_dir = os.path.join(base_dir, "STEADY GUARD")
        model_path = os.path.join(mode_dir, "samurai_ppo.zip")
        print("[INFO] Steady Defence selected.")
    elif choice == "3":
        mode_dir = os.path.join(base_dir, "EVASION DEFENCE")
        model_path = os.path.join(mode_dir, "samurai_ppo.zip")
        print("[INFO] Evasion selected.")
    else:
        print("[ERROR] Invalid selection.")
        return

    # ---------------- ROUTING ----------------
    if mode_dir:
        if not os.path.isdir(mode_dir):
            print(f"[ERROR] Mode folder not found: {mode_dir}")
            return
        # Ensure the selected mode's reward_engine.py is imported
        sys.path.insert(0, mode_dir)

    # Import AFTER sys.path is configured
    from samurai_env import SamuraiParryEnv

    env = SamuraiParryEnv(render_mode="human")

    # ---------------- MODEL LOADING ----------------
    use_model = False
    model = None

    if model_path and os.path.exists(model_path):
        print(f"[INFO] Loading model from: {model_path}")
        model = PPO.load(model_path, env=env, device="cuda")
        use_model = True
    else:
        print("[INFO] No PPO model loaded. Using random actions.")

    # ---------------- MAIN LOOP ----------------
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

                parry = bool(info.get("contact_parry"))
                hit = bool(info.get("contact_body_hit"))

                if parry and not prev_parry:
                    print("[EVENT] Parry registered.")
                if hit and not prev_hit:
                    print("[EVENT] Agent was hit!")

                prev_parry = parry
                prev_hit = hit

                time.sleep(1.0 / 60.0)

            ep += 1
            print(f"Episode {ep} reward: {ep_reward:.2f}")
            time.sleep(0.25)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user, closing environment.")

    env.close()


if __name__ == "__main__":
    main()
