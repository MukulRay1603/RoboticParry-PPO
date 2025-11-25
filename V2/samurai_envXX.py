import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
from typing import Optional, Tuple, Dict, Any

from sword_utils import create_sword, attach_sword_to_hand, get_sword_tip_position
from enemy_ai import OpponentController
from reward_engine import compute_reward_and_done
from obs_builder import build_observation
from camera_utils import get_default_camera
from reset_controller import smooth_reset_to_neutral


class SamuraiParryEnv(gym.Env):
    """
    Two-arm Franka Panda PyBullet environment for a simple samurai parry task.
    This version uses small helper modules to keep the environment logic clean.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", None],
        "render_fps": 60,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        time_step: float = 1.0 / 240.0,
        frame_skip: int = 1,
        max_steps: int = 200,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        assert render_mode in [None, "human", "rgb_array"], "Unsupported render_mode"
        self.render_mode = render_mode
        self.time_step = time_step
        self.frame_skip = frame_skip
        self.max_steps = max_steps

        self.n_joints = 7
        self.max_joint_vel = 4.0
        self.sword_length = 0.7

        # Connect to PyBullet
        if self.render_mode == "human":
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)

        p.setTimeStep(self.time_step, physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        if self.render_mode == "human":
            # Hide side debug windows
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=self.client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=self.client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=self.client_id)

        # Base positions
        self.agent_base_pos = np.array([-0.55, 0.0, 0.0], dtype=np.float32)
        self.opponent_base_pos = np.array([0.55, 0.0, 0.0], dtype=np.float32)

        self.agent_id: Optional[int] = None
        self.opponent_id: Optional[int] = None
        self.agent_sword_id: Optional[int] = None
        self.opponent_sword_id: Optional[int] = None
        self.agent_ee_link: Optional[int] = None
        self.opponent_ee_link: Optional[int] = None

        self.step_count = 0
        self.rng = np.random.RandomState(seed if seed is not None else 0)

        self._prev_agent_tip = np.zeros(3, dtype=np.float32)

        # Spaces
        obs_dim = 7 + 7 + 3 + 3 + 1 + 1
        high = np.ones(obs_dim, dtype=np.float32) * 5.0
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )

        # Build scene
        self._build_scene()

        # Opponent controller
        self.opponent_ctrl = OpponentController(
            client_id=self.client_id,
            robot_id=self.opponent_id,
            ee_link=self.opponent_ee_link,
            agent_base_pos=self.agent_base_pos,
            time_step=self.time_step,
            max_steps=self.max_steps,
            rng=self.rng,
        )

    # ------------- Gymnasium API -------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)

        self.step_count = 0

        # Neutral duel stance
        neutral = np.array([0.0, -0.4, 0.0, -1.5, 0.0, 1.8, 0.8], dtype=np.float32)
        smooth_reset_to_neutral(
            client_id=self.client_id,
            agent_id=self.agent_id,
            opponent_id=self.opponent_id,
            n_joints=self.n_joints,
            neutral=neutral,
            time_step=self.time_step,
            render_mode=self.render_mode,
        )

        self.opponent_ctrl.reset_attack()

        agent_tip = get_sword_tip_position(
            self.client_id, self.agent_sword_id, self.sword_length
        )
        self._prev_agent_tip = agent_tip.copy()

        obs = self._get_observation()
        info: Dict[str, Any] = {}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Clip and scale
        if isinstance(action, np.ndarray):
            a = np.clip(action, self.action_space.low, self.action_space.high)
        else:
            a = np.array(action, dtype=np.float32)
            a = np.clip(a, self.action_space.low, self.action_space.high)

        joint_vel = a * self.max_joint_vel

        self._apply_agent_action(joint_vel)
        self.opponent_ctrl.step()

        for _ in range(self.frame_skip):
            p.stepSimulation(physicsClientId=self.client_id)
            if self.render_mode == "human":
                time.sleep(self.time_step)

        self.step_count += 1

        obs = self._get_observation()

        agent_tip = get_sword_tip_position(
            self.client_id, self.agent_sword_id, self.sword_length
        )
        opp_tip = get_sword_tip_position(
            self.client_id, self.opponent_sword_id, self.sword_length
        )

        reward, done_hit, done_parry, info = compute_reward_and_done(
            client_id=self.client_id,
            agent_sword_id=self.agent_sword_id,
            opponent_sword_id=self.opponent_sword_id,
            agent_body_id=self.agent_id,
            agent_base_pos=self.agent_base_pos,
            agent_sword_tip=agent_tip,
            opponent_sword_tip=opp_tip,
            prev_agent_tip=self._prev_agent_tip,
        )

        self._prev_agent_tip = agent_tip.copy()

        terminated = done_hit
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            width, height, view_matrix, proj_matrix = get_default_camera(self.client_id)
            img = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                physicsClientId=self.client_id,
            )
            rgb_array = np.reshape(img[2], (height, width, 4))[:, :, :3]
            return rgb_array
        return None

    def close(self) -> None:
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)

    # ------------- Internal helpers -------------

    def _build_scene(self) -> None:
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(self.time_step, physicsClientId=self.client_id)
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)

        flags = p.URDF_USE_SELF_COLLISION

        # Agent
        self.agent_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=self.agent_base_pos.tolist(),
            baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
            useFixedBase=True,
            flags=flags,
            physicsClientId=self.client_id,
        )

        # Opponent rotated 180 degrees around Z
        self.opponent_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=self.opponent_base_pos.tolist(),
            baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, np.pi]),
            useFixedBase=True,
            flags=flags,
            physicsClientId=self.client_id,
        )

        self.agent_ee_link = self._find_link(self.agent_id, "panda_hand")
        self.opponent_ee_link = self._find_link(self.opponent_id, "panda_hand")

        # Create swords
        self.agent_sword_id = create_sword(self.client_id, self.sword_length)
        self.opponent_sword_id = create_sword(self.client_id, self.sword_length)

        attach_sword_to_hand(
            self.client_id, self.agent_id, self.agent_ee_link, self.agent_sword_id, self.sword_length
        )
        attach_sword_to_hand(
            self.client_id, self.opponent_id, self.opponent_ee_link, self.opponent_sword_id, self.sword_length
        )

        # Disable default motors on first 7 joints
        for robot_id in [self.agent_id, self.opponent_id]:
            for j in range(self.n_joints):
                p.setJointMotorControl2(
                    robot_id,
                    j,
                    controlMode=p.VELOCITY_CONTROL,
                    force=0,
                    physicsClientId=self.client_id,
                )

    def _find_link(self, body_id: int, name_substring: str) -> int:
        num_joints = p.getNumJoints(body_id, physicsClientId=self.client_id)
        for i in range(num_joints):
            info = p.getJointInfo(body_id, i, physicsClientId=self.client_id)
            joint_name = info[12].decode("utf-8")
            if name_substring in joint_name:
                return i
        return num_joints - 1

    def _apply_agent_action(self, joint_vel: np.ndarray) -> None:
        p.setJointMotorControlArray(
            self.agent_id,
            jointIndices=list(range(self.n_joints)),
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=joint_vel.tolist(),
            forces=[87.0] * self.n_joints,
            physicsClientId=self.client_id,
        )

    def _get_observation(self) -> np.ndarray:
        agent_q, agent_dq = self._get_joint_state(self.agent_id)
        agent_tip = get_sword_tip_position(
            self.client_id, self.agent_sword_id, self.sword_length
        )
        opp_tip = get_sword_tip_position(
            self.client_id, self.opponent_sword_id, self.sword_length
        )
        dist_to_base = float(np.linalg.norm(opp_tip - self.agent_base_pos))

        obs = build_observation(
            agent_q,
            agent_dq,
            agent_tip,
            opp_tip,
            dist_to_base,
            self.step_count,
            self.max_steps,
        )
        return obs

    def _get_joint_state(self, body_id: int):
        js = p.getJointStates(
            body_id, list(range(self.n_joints)), physicsClientId=self.client_id
        )
        q = np.array([s[0] for s in js], dtype=np.float32)
        dq = np.array([s[1] for s in js], dtype=np.float32)
        return q, dq


def make_env(render_mode: Optional[str] = None, **kwargs) -> SamuraiParryEnv:
    return SamuraiParryEnv(render_mode=render_mode, **kwargs)
