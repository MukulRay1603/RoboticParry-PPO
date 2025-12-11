
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from samurai_env import SamuraiParryEnv


def run_eval_episodes(model_path: str, num_episodes: int = 50, cooldown_steps: int = 15):
    #Evaluate the trained model and count approximate parry events.

    #We treat a parry as:
    # - the first sword–sword contact after at least `cooldown_steps`
     #   timesteps without contact.

    #This avoids double-counting sliding contact, but does not require
    #an explicit attack-cycle flag from the environment.
    
    env = SamuraiParryEnv(render_mode=None)
    model = PPO.load(model_path, env=env)

    ep_rewards = []
    ep_lengths = []
    ep_parries = []
    ep_hits = []

    total_parries = 0
    total_hits = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False

        ep_reward = 0.0
        ep_len = 0
        parries = 0
        hits = 0

        contact_cooldown = 0  # counts down from cooldown_steps

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            ep_len += 1

            parry_contact = info.get("contact_parry", False)

            # Cooldown-based parry counting:
            if contact_cooldown > 0:
                contact_cooldown -= 1

            if parry_contact and contact_cooldown == 0:
                parries += 1
                total_parries += 1
                contact_cooldown = cooldown_steps

            if info.get("contact_body_hit", False):
                hits += 1
                total_hits += 1

        ep_rewards.append(ep_reward)
        ep_lengths.append(ep_len)
        ep_parries.append(parries)
        ep_hits.append(hits)

        print(
            f"Episode {ep+1}/{num_episodes}: reward={ep_reward:.2f}, "
            f"len={ep_len}, parries={parries}, hits={hits}"
        )

    env.close()

    return {
        "ep_rewards": np.array(ep_rewards),
        "ep_lengths": np.array(ep_lengths),
        "ep_parries": np.array(ep_parries),
        "ep_hits": np.array(ep_hits),
        "total_parries": total_parries,
        "total_hits": total_hits,
    }


def plot_metrics(metrics, save_prefix: str = "samurai_metrics_cooldown"):
    ep_rewards = metrics["ep_rewards"]
    ep_lengths = metrics["ep_lengths"]
    ep_parries = metrics["ep_parries"]
    ep_hits = metrics["ep_hits"]
    total_parries = metrics["total_parries"]
    total_hits = metrics["total_hits"]

    episodes = np.arange(1, len(ep_rewards) + 1)

    total_attacks = total_parries + total_hits
    parry_rate = (total_parries / total_attacks) if total_attacks > 0 else 0.0

    print("\\n=== Summary (cooldown-based parry events) ===")
    print(f"Mean reward: {np.mean(ep_rewards):.2f}")
    print(f"Mean episode length: {np.mean(ep_lengths):.1f}")
    print(f"Mean parries/episode: {np.mean(ep_parries):.2f}")
    print(f"Mean hits/episode: {np.mean(ep_hits):.2f}")
    print(f"Total parries: {total_parries}")
    print(f"Total hits: {total_hits}")
    print(f"Parry rate: {parry_rate*100:.1f}%")

    # 1. Parries vs hits
    plt.figure()
    plt.plot(episodes, ep_parries, label="Parries/episode")
    plt.plot(episodes, ep_hits, label="Hits/episode")
    plt.xlabel("Episode")
    plt.ylabel("Count")
    plt.title("Parries vs Hits per Episode (Cooldown-based)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_parries_hits.png", dpi=300)

    # 2. Rewards
    plt.figure()
    plt.plot(episodes, ep_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Rewards")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_rewards.png", dpi=300)

    # 3. Episode length
    plt.figure()
    plt.plot(episodes, ep_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Episode Lengths")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_lengths.png", dpi=300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="samurai_ppo.zip")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--cooldown-steps",
        type=int,
        default=15,
        help="Minimum steps between parry events to avoid double counting.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    metrics = run_eval_episodes(
        args.model_path,
        num_episodes=args.episodes,
        cooldown_steps=args.cooldown_steps,
    )
    plot_metrics(metrics)


if __name__ == "__main__":
    main()
