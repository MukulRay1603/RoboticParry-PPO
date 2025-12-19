# 🥷 Samurai-Reflex-RL

### Reinforcement Learning for Robotic Sword Parrying in PyBullet



Samurai-Reflex-RL is a robotics project exploring **defensive reflex learning** using reinforcement learning (PPO) inside a custom PyBullet simulation.  

A 7-DOF robotic arm learns to **react and parry** incoming sword attacks from an opponent robot using continuous control.


This repository contains the full environment, training code, evaluation pipeline and reproducible setup scripts.



---



 ## ✨ Features


- 🗡️ **Scripted opponent attack model** with curved Bézier sword arcs  

- 🛡️ **PPO-trained defensive policy**  that learns parry reflexes  

- 🔁 **Cooldown-based parry detection metric** (fixes false positives)  

- 📊 Automatic **evaluation graphs**: rewards, parries, distributions  

- 🧪 **Deterministic evaluation** over 50 episodes  

- ⚙️ Fully reproducible **UV-powered Python 3.10 environment**  

- 🪶 Stable-Baselines3 + PyBullet + Gymnasium integration

- Interactive feature to select the **PPO Style**

---
# Overview

A rendered display of 2 Frank Panda arms holding swords in a pybullet Environment
![Pybullet Env](./outputs/Env_setup.png)


---

## NOTE

- The code contains two folders which have trained models for steady guard & defensive stance
- You can copy paste all files and replace the files within the respective directory where all the py files reside
- Then run
  ```bash
  python eval_samurai.py
  ```

  This shows the right run for each type of defensive training style

---

## Docker Running Setup

## Install Docker

```bash
sudo apt update
sudo apt install -y docker.io
sudo usermod -aG docker $USER
newgrp docker
```
*To Verify*
```bash
docker run hello-world
```

## X11 Forwarding 

```bash
xhost +local:docker
```

## Clone Repo
```bash
git clone https://github.com/MukulRay1603/RoboticParry-PPO.git
cd RoboticParry-PPO
```

## Build Docker Image
- No compose
- No http+docker
- Uses Docker CLI only
- Ubuntu 22.04 friendly

```bash
docker build -t samurai-ppo .
```

## Run with X11 forwarding 
```bash
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  samurai-ppo
```

---

### EXPECTED OUTPUT

Select Mode:
1 - Dry Run
2 - Steady Defence
3 - Evasion

---

 # ⚙️ Installation  & Environment Setup (Windows + UV) without docker



This project uses **Python 3.10** because PyBullet wheels do not support 3.11+.  

We use  **UV** for a clean and stable virtual environment.

##$ PROJECT RUNS BASED ON NVIDIA CUDA

- Please ensure you have a working gpu or tensor on your device
- Identify and get the right device drivers (ensure to upgrade them to the latest)
- Have the correct CUDA or CuDnn driver as well

Note: Project was tested using RTX 3080 10gb OC - CUDA 12.1


 ### 1️⃣ Install UV



```bash
pip install uv
```

1. Clone this repository
2. setup uv outside the folder
   

 ### 2️⃣ Create environment (Python 3.10 required)

```bash
uv venv samurai_rl --python 3.10
```


If you have a global path for python version above 3.10, (ie. 3.11 and above use) after installing 3.10.


```bash
py -3.11 -m uv venv samurai_rl --python 3.10
```



 ### 3️⃣ Activate environment



```bash
.\samurai_rl\Scripts\activate
```



 ### 4️⃣ Install dependencies



```bash

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

pip install "gymnasium [all]==1.2.2"

pip install "stable-baselines3 [extra]==2.2.1"

pip install https://github.com/bulletphysics/bullet3/releases/download/3.25/pybullet-3.25-cp310-cp310-win _amd64.whl

pip install matplotlib numpy

```

### 5️⃣ Change directory while in uv

```bash
cd SamuraiProject
```

### 6️⃣ For final sanity check
Install all requirements with correct versions to ensure smooth running

```bash
pip install -r requirements.txt
```
---



 # 🧠 Technical Overview

 # 🗡️ Opponent Attack Model


 *Attacks are not random — they follow a realistic sword swing using: *


* Quadratic Bézier curve interpolation
* Wind-up → Strike → Follow-through phases
* Random lateral offsets for realism
* Continuous updates per simulation step
* This produces lifelike attack trajectories that the agent must defend against.


# Baseline Test (Dry-Run)


 * Ensure no model or cache is saved in the model dir as the evaluate.py searches for the model within the same folder *

```bash

python evaluate.py

```

- This render stage is where we test the agent without a trained ppo model

<table>
  <tr>
    <td width="50%" align="center">
      <img src="./outputs/Baseline1.png" width="90%" alt="Falling bot image">
      <br>
      ** Falling Bot **
      <br>
      <small>The agent failed to maintain balance and fell immediately.</small>
    </td>
    <td width="50%" align="center">
      <img src="./outputs/Baseline2.png" width="90%" alt="Flailing bot image">
      <br>
      ** Flailing Bot **
      <br>
      <small>The agent flailed its limbs wildly attempting to recover.</small>
    </td>
  </tr>
</table>


---


 # 🥋 Training the Agent

```bash

python train.py

```

This will:

* CLI Training mode
* Train for N timesteps
* Save model to: models/samurai _ppo.zip

* Saved models are available for 2 styles with unique reward engine
- evasion
- steady guard

Trained ppo model for 200k iterations for steady guard

<img src= "./outputs/training_200k.png" width="500"/>
---

 # 🔬 Evaluation

 *Use event-based parry metric:*

```bash

python evaluate.py

```

 *NOTE:*

* Running this without training works as a dry run
* It uses base reward system idea
* Running after training will show trained results

### Test with Steady Guard

<table>
  <tr>
    <td width="50%" align="center">
      <img src="./outputs/steadydefence1.png" width="90%" alt="Steadydefence">
      <br>
      ** Steady Defence 1 **
      <br>
      <small>The agent holds it's body in a strong pose by only moving the wrist ie. sword arc deflect </small>
    </td>
    <td width="50%" align="center">
      <img src="./outputs/steadydefence2.png" width="90%" alt="Steadydefence">
      <br>
      ** Steady Defence 2 **
      <br>
      <small> The agent's reward engine delivers higher rewards for sword contact than motion </small>
    </td>
  </tr>
</table>

### Test with Evasion

<table>
  <tr>
    <td width="50%" align="center">
      <img src="./outputs/evasion1.png" width="90%" alt="Evasion">
      <br>
      ** Evasion 1 **
      <br>
      <small>The agent feels the need to move out of the way by leaning back/ forward </small>
    </td>
    <td width="50%" align="center">
      <img src="./outputs/evasion2.png" width="90%" alt="Evasion">
      <br>
      ** Evasion 2 **
      <br>
      <small> The agent's reward engine delivers higher rewards for motion than sword contact </small>
    </td>
  </tr>
</table>
 
Output summary:

<img src="./outputs/summary.png" width="500"/>

We observe the metrics post test
- Mean reward: 8.66
- Mean episode length: 200
- Mean parries/episode: 3.02
- Parry rate: 100%
- Total hits: 0

---

 # 🔁 Cooldown-Based Parry Detection


Originally the system counted every contact frame as a “parry”, inflating numbers.


 *fixed this using:*

* Sliding cooldown window (≥ 15 steps)
* Spatial position check
* Blade orientation check
* Contact force validation

Such as:

- Mean parries per episode: 3.02
- Total parries: 151
- Total hits: 0
- Parry rate: 100%

---

# Metrics for Training

<table>
  <tr>
    <td width="50%" align="center">
      ** Learning Curve Based on FPS **
      <img src="outputs/learning_curve_time_fps.png" width="100%" alt="Learning curve based on FPS">
    </td>
    <td width="50%" align="center">
      ** Learning Curve Based on Training Entropy Loss **
      <img src="outputs/learning_curve_train_entropy_loss.png" width="100%" alt="Learning curve based on train entropy loss">
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      ** Learning Curve Based on Policy Gradient Loss **
      <img src="outputs/learning_curve_train_policy_gradient_loss.png" width="100%" alt="Learning curve based on train policy gradient loss">
    </td>
    <td width="50%" align="center">
      ** Learning Curve Based on Value Loss **
      <img src="outputs/learning_curve_train_value_loss.png" width="100%" alt="Learning curve based on train value loss">
    </td>
  </tr>
</table>

---

 # 🚧 Known Limitations


* Opponent is scripted (not a learning agent)
* No real sensor noise or actuation latency
* Parry angle thresholds still coarse
* No domain randomization yet
* Designed for Windows; Linux requires PyBullet wheel rebuild



---
