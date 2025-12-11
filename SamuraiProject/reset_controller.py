import time
from typing import Optional

import numpy as np
import pybullet as p


def smooth_reset_to_neutral(
    client_id: int,
    agent_id: int,
    opponent_id: int,
    n_joints: int,
    neutral: np.ndarray,
    time_step: float,
    render_mode: Optional[str] = None,
    settle_steps: int = 60,
) -> None:
    """
    Smoothly move both arms to a neutral pose using position control.
    """
    # Get current joint angles
    js_agent = p.getJointStates(
        agent_id, list(range(n_joints)), physicsClientId=client_id
    )
    js_opp = p.getJointStates(
        opponent_id, list(range(n_joints)), physicsClientId=client_id
    )
    agent_q = np.array([s[0] for s in js_agent], dtype=np.float32)
    opp_q = np.array([s[0] for s in js_opp], dtype=np.float32)

    for step in range(settle_steps):
        alpha = float(step + 1) / float(settle_steps)
        q_agent = (1.0 - alpha) * agent_q + alpha * neutral
        q_opp = (1.0 - alpha) * opp_q + alpha * neutral

        p.setJointMotorControlArray(
            agent_id,
            jointIndices=list(range(n_joints)),
            controlMode=p.POSITION_CONTROL,
            targetPositions=q_agent.tolist(),
            forces=[87.0] * n_joints,
            physicsClientId=client_id,
        )
        p.setJointMotorControlArray(
            opponent_id,
            jointIndices=list(range(n_joints)),
            controlMode=p.POSITION_CONTROL,
            targetPositions=q_opp.tolist(),
            forces=[87.0] * n_joints,
            physicsClientId=client_id,
        )

        p.stepSimulation(physicsClientId=client_id)
        if render_mode == "human":
            time.sleep(time_step)

    # Snap exactly to neutral
    for j in range(n_joints):
        p.resetJointState(
            agent_id,
            j,
            float(neutral[j]),
            targetVelocity=0.0,
            physicsClientId=client_id,
        )
        p.resetJointState(
            opponent_id,
            j,
            float(neutral[j]),
            targetVelocity=0.0,
            physicsClientId=client_id,
        )

    # Let things settle a bit
    for _ in range(10):
        p.stepSimulation(physicsClientId=client_id)
