import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
from typing import Optional, Tuple, Dict, Any


class SamuraiParryEnv(gym.Env):
    """
    Two-arm Franka Panda PyBullet environment for a simple samurai parry task.

    - Agent: left Panda arm with a sword attached to its end-effector.
    - Opponent: right Panda arm with a sword attached, performs scripted/random attacks.
    - Goal: agent moves its sword to parry incoming attacks and avoid being hit.
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

        # Basic robot / sword parameters
        self.n_joints = 7  # 7 DOF Panda arm
        self.max_joint_vel = 4.0  # rad/s
        self.sword_length = 0.7  # meters, tip-to-hilt length for box "blade"

        # PyBullet setup
        if self.render_mode == "human":
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)

        p.setTimeStep(self.time_step, physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Hide PyBullet debug GUI panels for a clean view when using the GUI
        if self.render_mode == "human":
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=self.client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=self.client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=self.client_id)

        # Arm base positions (agent on the left, opponent on the right)
        self.agent_base_pos = np.array([-0.6, 0.0, 0.0], dtype=np.float32)
        self.opponent_base_pos = np.array([0.6, 0.0, 0.0], dtype=np.float32)

        self.agent_id: Optional[int] = None
        self.opponent_id: Optional[int] = None
        self.agent_sword_id: Optional[int] = None
        self.opponent_sword_id: Optional[int] = None
        self.agent_ee_link: Optional[int] = None
        self.opponent_ee_link: Optional[int] = None

        # Attack / episode state
        self.step_count = 0
        self.rng = np.random.RandomState(seed if seed is not None else 0)

        # For motion bonus
        self._prev_agent_tip: Optional[np.ndarray] = None

        # Observations: [agent_q(7), agent_dq(7), agent_sword_tip_xyz(3),
        #                opp_sword_tip_xyz(3), dist_to_base(1), time_fraction(1)]
        obs_dim = 7 + 7 + 3 + 3 + 1 + 1
        high = np.ones(obs_dim, dtype=np.float32) * 5.0
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Actions: joint velocity commands for 7 arm joints in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )

        # Build the world once
        self._build_scene()
        self._set_random_seed(seed)

    # ----------------- Gymnasium API -----------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._set_random_seed(seed)

        self.step_count = 0
        self._reset_simulation()
        obs = self._get_observation()
        info: Dict[str, Any] = {}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Clip and scale action
        if isinstance(action, np.ndarray):
            a = np.clip(action, self.action_space.low, self.action_space.high)
        else:
            a = np.array(action, dtype=np.float32)
            a = np.clip(a, self.action_space.low, self.action_space.high)

        joint_velocities = a * self.max_joint_vel

        # Apply actions to agent arm
        self._apply_agent_action(joint_velocities)

        # Opponent scripted attack
        self._opponent_attack_step()

        # Step physics
        for _ in range(self.frame_skip):
            p.stepSimulation(physicsClientId=self.client_id)
            if self.render_mode == "human":
                time.sleep(self.time_step)

        self.step_count += 1

        obs = self._get_observation()
        reward, done_hit, done_parry, info = self._compute_reward_and_done()

        # Only a body hit ends the episode; successful parries keep the duel going.
        terminated = done_hit
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            width, height, view_matrix, proj_matrix = self._get_camera_matrices()
            img = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                physicsClientId=self.client_id,
            )
            rgb_array = np.reshape(img[2], (height, width, 4))[:, :, :3]
            return rgb_array
        # For "human", PyBullet GUI is already running
        return None

    def close(self) -> None:
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)

    # ----------------- Internal helpers -----------------

    def _set_random_seed(self, seed: Optional[int]) -> None:
        if seed is not None:
            self.rng.seed(seed)

    def _build_scene(self) -> None:
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(self.time_step, physicsClientId=self.client_id)

        p.loadURDF("plane.urdf", physicsClientId=self.client_id)

        flags = p.URDF_USE_SELF_COLLISION

        # Agent: facing +X
        self.agent_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=self.agent_base_pos.tolist(),
            baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
            useFixedBase=True,
            flags=flags,
            physicsClientId=self.client_id,
        )

        # Opponent: rotated 180deg around Z to face the agent
        self.opponent_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=self.opponent_base_pos.tolist(),
            baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, np.pi]),
            useFixedBase=True,
            flags=flags,
            physicsClientId=self.client_id,
        )

        # Find end-effector link index by name (panda_hand)
        self.agent_ee_link = self._find_link_index(self.agent_id, "panda_hand")
        self.opponent_ee_link = self._find_link_index(self.opponent_id, "panda_hand")

        # Create swords and attach to end-effectors
        self.agent_sword_id = self._create_and_attach_sword(
            self.agent_id, self.agent_ee_link
        )
        self.opponent_sword_id = self._create_and_attach_sword(
            self.opponent_id, self.opponent_ee_link
        )

        # Disable default motors for first 7 joints to allow direct control
        for robot_id in [self.agent_id, self.opponent_id]:
            for j in range(self.n_joints):
                p.setJointMotorControl2(
                    robot_id,
                    j,
                    controlMode=p.VELOCITY_CONTROL,
                    force=0,
                    physicsClientId=self.client_id,
                )

    def _reset_simulation(self) -> None:
        """
        Smoothly move both arms back to a neutral duel stance instead of teleporting.
        """
        neutral = np.array([0.0, -0.3, 0.0, -1.2, 0.0, 1.7, 0.5], dtype=np.float32)
        settle_steps = 60

        # Get current joint angles
        agent_q, _ = self._get_joint_state(self.agent_id)
        opp_q, _ = self._get_joint_state(self.opponent_id)

        for step in range(settle_steps):
            alpha = float(step + 1) / float(settle_steps)
            q_agent = (1.0 - alpha) * agent_q + alpha * neutral
            q_opp = (1.0 - alpha) * opp_q + alpha * neutral

            p.setJointMotorControlArray(
                self.agent_id,
                jointIndices=list(range(self.n_joints)),
                controlMode=p.POSITION_CONTROL,
                targetPositions=q_agent.tolist(),
                forces=[87.0] * self.n_joints,
                physicsClientId=self.client_id,
            )
            p.setJointMotorControlArray(
                self.opponent_id,
                jointIndices=list(range(self.n_joints)),
                controlMode=p.POSITION_CONTROL,
                targetPositions=q_opp.tolist(),
                forces=[87.0] * self.n_joints,
                physicsClientId=self.client_id,
            )

            p.stepSimulation(physicsClientId=self.client_id)
            if self.render_mode == "human":
                time.sleep(self.time_step)

        # Snap exactly to neutral to avoid drift
        for j in range(self.n_joints):
            p.resetJointState(
                self.agent_id,
                j,
                float(neutral[j]),
                targetVelocity=0.0,
                physicsClientId=self.client_id,
            )
            p.resetJointState(
                self.opponent_id,
                j,
                float(neutral[j]),
                targetVelocity=0.0,
                physicsClientId=self.client_id,
            )

        # Reset motion bonus state
        self._prev_agent_tip = None

        # Initialize a new opponent attack
        self._init_opponent_attack()

        # Let things settle a few steps at neutral
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.client_id)

    def _find_link_index(self, body_id: int, name_substring: str) -> int:
        num_joints = p.getNumJoints(body_id, physicsClientId=self.client_id)
        for i in range(num_joints):
            info = p.getJointInfo(body_id, i, physicsClientId=self.client_id)
            joint_name = info[12].decode("utf-8")
            if name_substring in joint_name:
                return i
        return num_joints - 1

    def _create_and_attach_sword(self, robot_id: int, ee_link: int) -> int:
        """
        Create a simple thin box as a "sword" and rigidly attach it to the
        given end-effector link.

        Convention:
          - Sword local +X is along the blade, from hilt to tip.
          - Hilt is at local X = -length/2.
          - Tip is at  local X = +length/2.
        """
        length = self.sword_length
        half_len = length * 0.5

        # Box aligned with its X-axis as the blade direction
        half_extents = [half_len, 0.015, 0.015]

        col_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self.client_id,
        )
        vis_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self.client_id,
        )

        sword_id = p.createMultiBody(
            baseMass=0.5,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=[0, 0, 0],
            physicsClientId=self.client_id,
        )

        # Attach at the hand so the hilt (local X = -half_len) is at the hand origin.
        parent_frame_pos = [0.0, 0.0, 0.0]
        child_frame_pos = [-half_len, 0.0, 0.0]

        # No rotation: sword +X follows hand +X.
        child_frame_orn = p.getQuaternionFromEuler([0.0, 0.0, 0.0])

        p.createConstraint(
            parentBodyUniqueId=robot_id,
            parentLinkIndex=ee_link,
            childBodyUniqueId=sword_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=parent_frame_pos,
            childFramePosition=child_frame_pos,
            childFrameOrientation=child_frame_orn,
            physicsClientId=self.client_id,
        )

        # Disable collisions between this sword and its own robot only
        num_links = p.getNumJoints(robot_id, physicsClientId=self.client_id)
        for link_idx in range(-1, num_links):
            p.setCollisionFilterPair(
                robot_id,
                sword_id,
                link_idx,
                -1,
                enableCollision=0,
                physicsClientId=self.client_id,
            )

        return sword_id

    def _apply_agent_action(self, joint_velocities: np.ndarray) -> None:
        p.setJointMotorControlArray(
            self.agent_id,
            jointIndices=list(range(self.n_joints)),
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=joint_velocities.tolist(),
            forces=[87.0] * self.n_joints,
            physicsClientId=self.client_id,
        )

    # ------------- Opponent attack scripting -------------

    def _init_opponent_attack(self):
        """Initialize clean diagonal left/right kesa slashes.
        Sword orientation is FIXED. Only joints move to produce arcs.
        """

        base_pose = np.array([0.0, -0.3, 0.0, -1.2, 0.0, 1.7, 0.5])

        # --- B) LEFT DIAGONAL KESA CUT (from high right → low left) ---
        left_windup = base_pose + np.array([
            0.65,   # shoulder pan to right
            -0.55,  # raise arm
            0.55,   # elbow outward
            -0.65,  # wrist back
            -0.45,  # wrist roll
            0.25,   # wrist lift
            0.0,
        ])
        left_strike = base_pose + np.array([
            0.05,
            -0.95,
            0.1,
            -0.25,
            -1.05,
            0.05,
            0.0,
        ])
        left_recover = base_pose

        # --- C) RIGHT DIAGONAL KESA CUT (mirror) ---
        right_windup = base_pose + np.array([
            -0.65,
            -0.55,
            -0.55,
            -0.65,
            0.45,
            0.25,
            0.0,
        ])
        right_strike = base_pose + np.array([
            -0.05,
            -0.95,
            -0.1,
            -0.25,
            1.05,
            0.05,
            0.0,
        ])
        right_recover = base_pose

        self.attack_library = [
            [left_windup, left_strike, left_recover],
            [right_windup, right_strike, right_recover],
        ]
        idx = int(self.rng.randint(len(self.attack_library)))
        self.attack_keyframes = self.attack_library[idx]

        self.attack_duration_steps = max(45, self.max_steps // 4)
        self.attack_progress = 0

    def _opponent_attack_step(self) -> None:
        # Keyframed opponent attack: interpolate through windup -> strike -> recover
        if not hasattr(self, "attack_keyframes") or self.attack_keyframes is None:
            self._init_opponent_attack()

        kfs = self.attack_keyframes
        n_seg = len(kfs) - 1
        if n_seg <= 0:
            return

        # Progress in [0, 1]
        t_global = float(self.attack_progress) / float(max(1, self.attack_duration_steps - 1))
        t_global = max(0.0, min(1.0, t_global))

        # Determine which segment of the attack we are in
        seg = int(t_global * n_seg)
        if seg >= n_seg:
            seg = n_seg - 1
            local_t = 1.0
        else:
            seg_start = float(seg) / float(n_seg)
            seg_end = float(seg + 1) / float(n_seg)
            local_t = (t_global - seg_start) / max(1e-6, (seg_end - seg_start))

        q_start = np.array(kfs[seg], dtype=np.float32)
        q_end = np.array(kfs[seg + 1], dtype=np.float32)
        q_des = (1.0 - local_t) * q_start + local_t * q_end

        p.setJointMotorControlArray(
            self.opponent_id,
            jointIndices=list(range(self.n_joints)),
            controlMode=p.POSITION_CONTROL,
            targetPositions=q_des.tolist(),
            forces=[87.0] * self.n_joints,
            physicsClientId=self.client_id,
        )

        self.attack_progress += 1
        if self.attack_progress >= self.attack_duration_steps:
            # Start a new swing once the current one finishes
            self._init_opponent_attack()

    # ------------- Observation & Reward -------------

    def _get_joint_state(self, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
        js = p.getJointStates(
            body_id, list(range(self.n_joints)), physicsClientId=self.client_id
            )
        q = np.array([s[0] for s in js], dtype=np.float32)
        dq = np.array([s[1] for s in js], dtype=np.float32)
        return q, dq

    def _get_sword_tip_pos(self, sword_id: int) -> np.ndarray:
        """
        Get the world position of the sword tip, assuming tip at local +X.
        """
        pos, orn = p.getBasePositionAndOrientation(
            sword_id, physicsClientId=self.client_id
        )
        pos = np.array(pos, dtype=np.float32)
        half_len = self.sword_length * 0.5
        rot_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        local_tip = np.array([half_len, 0.0, 0.0], dtype=np.float32)
        tip_world = pos + rot_mat @ local_tip
        return tip_world.astype(np.float32)

    def _get_observation(self) -> np.ndarray:
        agent_q, agent_dq = self._get_joint_state(self.agent_id)
        agent_sword_tip = self._get_sword_tip_pos(self.agent_sword_id)
        opp_sword_tip = self._get_sword_tip_pos(self.opponent_sword_id)

        dist_to_base = np.linalg.norm(opp_sword_tip - self.agent_base_pos).astype(
            np.float32
        )

        time_fraction = float(self.step_count) / float(max(1, self.max_steps))

        obs = np.concatenate(
            [
                agent_q,
                agent_dq,
                agent_sword_tip,
                opp_sword_tip,
                np.array([dist_to_base], dtype=np.float32),
                np.array([time_fraction], dtype=np.float32),
            ]
        ).astype(np.float32)

        return obs

    def _compute_reward_and_done(self) -> Tuple[float, bool, bool, Dict[str, Any]]:
        agent_sword_tip = self._get_sword_tip_pos(self.agent_sword_id)
        opp_sword_tip = self._get_sword_tip_pos(self.opponent_sword_id)

        dist_swords = float(np.linalg.norm(agent_sword_tip - opp_sword_tip))
        dist_to_base = float(np.linalg.norm(opp_sword_tip - self.agent_base_pos))

        parry_zone_radius = 0.7
        parry_active = dist_to_base < parry_zone_radius

        reward = 0.0

        # Encourage being close to the opponent sword tip, but not just camping
        if parry_active:
            reward += 1.5 * np.exp(-4.0 * dist_swords)
            # Small penalty for "just blocking" in the parry zone with no actual contact
            # to discourage stationary camping solutions.
            contact_parry_tmp = p.getContactPoints(
                bodyA=self.agent_sword_id,
                bodyB=self.opponent_sword_id,
                physicsClientId=self.client_id,
            )
            if len(contact_parry_tmp) == 0:
                reward -= 0.02
        else:
            reward += 0.1 * np.exp(-2.0 * dist_swords)

        # Smooth movement bonus: encourage some motion of the agent sword tip
        if self._prev_agent_tip is None:
            self._prev_agent_tip = agent_sword_tip.copy()
        movement = float(np.linalg.norm(agent_sword_tip - self._prev_agent_tip))
        self._prev_agent_tip = agent_sword_tip.copy()
        reward += 0.01 * movement

        # Small control penalty
        _, agent_dq = self._get_joint_state(self.agent_id)
        control_cost = 0.01 * float(np.sum(agent_dq ** 2))
        reward -= control_cost

        # Contacts for parry / hit detection
        contact_parry = p.getContactPoints(
            bodyA=self.agent_sword_id,
            bodyB=self.opponent_sword_id,
            physicsClientId=self.client_id,
        )
        contact_body_hit = p.getContactPoints(
            bodyA=self.opponent_sword_id,
            bodyB=self.agent_id,
            physicsClientId=self.client_id,
        )

        done_hit = False
        done_parry = False

        if len(contact_body_hit) > 0:
            # Opponent sword hit agent body: big penalty and terminate
            reward -= 5.0
            done_hit = True

        if parry_active and len(contact_parry) > 0:
            # Successful parry: big reward but do NOT terminate the episode
            reward += 5.0
            done_parry = False

        info: Dict[str, Any] = {
            "dist_swords": dist_swords,
            "dist_to_base": dist_to_base,
            "parry_active": parry_active,
            "contact_parry": len(contact_parry) > 0,
            "contact_body_hit": len(contact_body_hit) > 0,
        }

        return float(reward), done_hit, done_parry, info

    # ------------- Camera helpers -------------

    def _get_camera_matrices(self):
        cam_target = [0.0, 0.0, 0.6]
        cam_distance = 2.5
        yaw = 90
        pitch = -35
        roll = 0
        up_axis_index = 2

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=cam_target,
            distance=cam_distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            upAxisIndex=up_axis_index,
            physicsClientId=self.client_id,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.0,
            nearVal=0.1,
            farVal=5.0,
            physicsClientId=self.client_id,
        )
        width = 640
        height = 480
        return width, height, view_matrix, proj_matrix


def make_env(render_mode: Optional[str] = None, **kwargs) -> SamuraiParryEnv:
    return SamuraiParryEnv(render_mode=render_mode, **kwargs)
