"""
Plot PPO learning curves from Stable-Baselines3 TensorBoard logs.

Usage (from repo root):
  python plot_tb_learning_curve.py --logdir ./tb_samurai/ --out learning_curve.png

Notes:
- train_samurai.py logs to ./tb_samurai/ via `tensorboard_log` in PPO.  (see train_samurai.py)
- This script reads TensorBoard event files and plots common SB3 scalars if present.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt

def _load_scalars(logdir: Path):
    # TensorBoard's event accumulator is bundled with tensorboard package.
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    # Find all event files under logdir
    event_files = list(logdir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under: {logdir}")

    # SB3 usually creates subfolders; pick the newest event file as primary.
    event_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    ea = EventAccumulator(str(event_files[0]))
    ea.Reload()

    tags = ea.Tags().get("scalars", [])

    # Candidate SB3 tags (varies by SB3 version/policy)
    candidates = [
        "rollout/ep_rew_mean",
        "rollout/ep_len_mean",
        "train/value_loss",
        "train/policy_gradient_loss",
        "train/entropy_loss",
        "time/fps",
    ]

    found = [t for t in candidates if t in tags]
    if not found:
        # fallback: just grab anything rollout/*
        found = [t for t in tags if t.startswith("rollout/")]
        if not found:
            found = tags[:6]  # last resort: plot a few

    series = {}
    for tag in found:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        vals = [e.value for e in events]
        series[tag] = (steps, vals)

    return series, event_files[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=str, default="./tb_samurai/")
    ap.add_argument("--out", type=str, default="learning_curve.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    logdir = Path(args.logdir)
    series, used_file = _load_scalars(logdir)

    # Plot each scalar as its own figure (keeps it clean)
    out_base = Path(args.out)
    out_dir = out_base.parent if out_base.parent.as_posix() != "." else Path(".")

    print(f"[INFO] Using event file: {used_file}")
    print(f"[INFO] Found series: {', '.join(series.keys())}")

    for tag, (steps, vals) in series.items():
        plt.figure()
        plt.plot(steps, vals)
        plt.xlabel("Timesteps")
        plt.ylabel(tag)
        plt.title(tag)
        # Save per-tag
        safe_tag = tag.replace("/", "_")
        out_path = out_dir / f"{out_base.stem}_{safe_tag}{out_base.suffix}"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        print(f"[OK] Wrote: {out_path}")

        if args.show:
            plt.show()
        plt.close()

if __name__ == "__main__":
    main()
