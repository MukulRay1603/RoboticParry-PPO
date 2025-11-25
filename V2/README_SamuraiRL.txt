
SamuraiRL - Franka Panda Parry Environment (PyBullet + Gymnasium + SB3 PPO)
===========================================================================

Files in this mini-project
--------------------------
- samurai_env.py       : Gymnasium environment with two Franka Panda arms + swords.
- train_samurai.py     : PPO training script (Stable-Baselines3).
- eval_samurai.py      : GUI evaluation script for trained policy.

How to use (with your UV venv)
------------------------------
1. Activate your venv (as you already do):
       samurai_rl\Scripts\activate

2. Make sure the key packages are installed in this venv:

       samurai_rl\Scripts\python.exe -m pip install gymnasium==1.2.2
       samurai_rl\Scripts\python.exe -m pip install "stable-baselines3[extra]==2.2.1"
       samurai_rl\Scripts\python.exe -m pip install pybullet
       # torch / torchvision / torchaudio already installed with CUDA 12.1

3. Put these files in some folder, for example:
       D:\SamuraiRL\
       ├── samurai_env.py
       ├── train_samurai.py
       └── eval_samurai.py

4. Train PPO:

       cd D:\SamuraiRL\
       samurai_rl\Scripts\python.exe train_samurai.py --timesteps 200000

   This will create:
       samurai_ppo.zip

5. Watch the trained agent in PyBullet GUI:

       samurai_rl\Scripts\python.exe eval_samurai.py --model-path samurai_ppo.zip

Notes
-----
- The environment is intentionally "research-grade simple": two 7 DOF Franka Panda arms,
  swords are rigidly attached boxes, and the opponent runs a small library of scripted
  attack motions (overhead slash, side slashes).
- Reward:
  * Encourages keeping the agent sword close to the opponent sword, especially when the
    opponent sword is near the agent base (parry zone).
  * Penalizes when the opponent sword hits the agent body.
  * Rewards successful sword-on-sword contact in the parry zone.
- Termination:
  * Episode ends on a successful parry or when the agent is hit.
  * Episode also truncates after a fixed number of steps.
- Everything is kept compatible with:
  * Python 3.10
  * Gymnasium 1.2.2
  * Stable-Baselines3 2.2.1
  * PyTorch 2.5.1 + CUDA 12.1
  * Windows 11 + PyBullet (GUI / DIRECT only, no ROS/MuJoCo).

If something explodes (numerically, not your GPU), we can tweak the reward shaping,
action scaling, or attack scripts.
