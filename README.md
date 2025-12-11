# 🥷 Samurai-Reflex-RL

### Reinforcement Learning for Robotic Sword Parrying in PyBullet



Samurai-Reflex-RL is a robotics project exploring  **defensive reflex learning ** using reinforcement learning (PPO) inside a custom PyBullet simulation.  

A 7-DOF robotic arm learns to  **detect, react and parry ** incoming sword attacks from an opponent robot using continuous control.



This repository contains the full environment, training code, evaluation pipeline and reproducible setup scripts.



---



 ## ✨ Features


- 🗡️  **Scripted opponent attack model ** with curved Bézier sword arcs  

- 🛡️  **PPO-trained defensive policy **  that learns parry reflexes  

- 🔁  **Cooldown-based parry detection metric ** (fixes false positives)  

- 📊 Automatic  **evaluation graphs **: rewards, parries, distributions  

- 🧪  **Deterministic evaluation ** over 50 episodes  

- ⚙️ Fully reproducible  **UV-powered Python 3.10 environment **  

- 🪶 Stable-Baselines3 + PyBullet + Gymnasium integration  


---



 # ⚙️ Installation  & Environment Setup (Windows + UV)



This project uses  **Python 3.10 ** because PyBullet wheels do not support 3.11+.  

We use   **UV  ** for a clean and stable virtual environment.



 ### 1️⃣ Install UV



```bash
pip install uv
```





 ### 2️⃣ Create environment (Python 3.10 required)

```bash
uv venv samurai _rl --python 3.10
```



If you have a global path for python version above 3.10, (ie. 3.11 and above use) after installing 3.10.



```bash
py -3.11 -m uv venv samurai _rl --python 3.10
```



 ### 3️⃣ Activate environment



```bash
.  samurai _rl  Scripts  Activate.ps1
```



 ### 4️⃣ Install dependencies



```bash

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

pip install "gymnasium [all]==1.2.2"

pip install "stable-baselines3 [extra]==2.2.1"

pip install https://github.com/bulletphysics/bullet3/releases/download/3.25/pybullet-3.25-cp310-cp310-win _amd64.whl

pip install matplotlib numpy

```



---



 # 🧠 Technical Overview



 ##🗡️ Opponent Attack Model



 *Attacks are not random — they follow a realistic sword swing using: *



* Quadratic Bézier curve interpolation
* Wind-up → Strike → Follow-through phases
* Random lateral offsets for realism
* Continuous updates per simulation step
* This produces lifelike attack trajectories that the agent must defend against.



---



 # 🔁 Cooldown-Based Parry Detection



Originally the system counted every contact frame as a “parry”, inflating numbers.



 *fixed this using: *



* Sliding cooldown window (≥ 15 steps)
* Spatial position check
* Blade orientation check
* Contact force validation



Such as:



Mean parries per episode: 3.02

Total parries: 151

Total hits: 0

Parry rate: 100%





---



 # 🥋 Training the Agent



```bash

python train.py

```



This will:



* CLI Training mode
* Train for N timesteps
* Save model to: models/samurai _ppo.zip



---



 # 🔬 Evaluation



 *Use event-based parry metric: *



```bash

python evaluate.py

```


 *NOTE:  *

* Running this without training works as a dry run
* It uses base reward system idea
* Running after training will show trained results



Output summary:



Mean reward: 8.66

Mean episode length: 200

Mean parries/episode: 3.02

Parry rate: 100%

Total hits: 0



---



 # 🚧 Known Limitations



* Opponent is scripted (not a learning agent)
* No real sensor noise or actuation latency
* Parry angle thresholds still coarse
* No domain randomization yet
* Designed for Windows; Linux requires PyBullet wheel rebuild



---
