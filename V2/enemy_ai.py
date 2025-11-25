import numpy as np
import pybullet as p
from typing import Optional


class OpponentController:
    """
    Scripted opponent that performs repeated sword attacks towards the agent.

    Implementation:
    - Each attack is parameterised as a sequence of target EE positions in
      world coordinates (a simple arc).
    - For each control step, we interpolate along that arc and use
      inverse kinematics to compute joint targets.
    """

    def __init__(
        self,
        client_id: int,
        robot_id: int,
        ee_link: int,
        agent_base_pos: np.ndarray,
        time_step: float,
        max_steps: int,
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        self.client_id = client_id
        self.robot_id = robot_id
        self.ee_link = ee_link
        self.agent_base_pos = np.array(agent_base_pos, dtype=np.float32)
        self.time_step = float(time_step)
        self.max_steps = int(max_steps)
        self.rng = rng if rng is not None else np.random.RandomState(0)

        self.attack_duration_steps = max(60, self.max_steps // 3)
        self.attack_progress = 0
        self.ctrl_points = None  # type: ignore

    # ------------- Attack definition -------------

    def reset_attack(self) -> None:
        """
        Sample a new attack trajectory that passes near the agent.
        """
        # Get current EE pose to anchor the attack
        ee_state = p.getLinkState(
            self.robot_id,
            self.ee_link,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        ee_pos = np.array(ee_state[4], dtype=np.float32)

        # Target region: a point slightly above the agent base
        target_body = self.agent_base_pos + np.array([0.0, 0.0, 0.9], dtype=np.float32)

        # Lateral offset to vary angle of attack
        lateral = float(self.rng.uniform(-0.25, 0.25))
        target_mid = target_body + np.array([0.0, lateral, 0.0], dtype=np.float32)

        # Define a simple 3-point arc: windup -> strike -> follow-through
        windup = ee_pos + np.array([0.0, 0.0, 0.25], dtype=np.float32)
        strike = target_mid
        follow_through = strike + np.array([-0.15, 0.0, -0.25], dtype=np.float32)

        self.ctrl_points = np.stack([windup, strike, follow_through], axis=0)
        self.attack_progress = 0

    # ------------- Step attack -------------

    def step(self) -> None:
        if self.ctrl_points is None:
            self.reset_attack()

        # Normalised progress in [0, 1]
        t_global = float(self.attack_progress) / float(
            max(1, self.attack_duration_steps - 1)
        )
        t_global = np.clip(t_global, 0.0, 1.0)

        # Quadratic Bezier interpolation with 3 control points
        p0, p1, p2 = self.ctrl_points
        pos = (
            (1.0 - t_global) ** 2 * p0
            + 2.0 * (1.0 - t_global) * t_global * p1
            + t_global**2 * p2
        )

        # Use IK to compute joint targets (orientation is left free)
        q_des = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link,
            pos.tolist(),
            maxNumIterations=50,
            residualThreshold=1e-3,
            physicsClientId=self.client_id,
        )

        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client_id)
        joint_indices = list(range(min(7, num_joints)))  # first 7 arm joints
        q_des = q_des[: len(joint_indices)]

        p.setJointMotorControlArray(
            self.robot_id,
            jointIndices=joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=q_des,
            forces=[87.0] * len(joint_indices),
            physicsClientId=self.client_id,
        )

        self.attack_progress += 1
        if self.attack_progress >= self.attack_duration_steps:
            # Start a new attack
            self.reset_attack()
