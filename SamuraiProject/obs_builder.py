from typing import Tuple

import numpy as np


def build_observation(
    agent_q: np.ndarray,
    agent_dq: np.ndarray,
    agent_sword_tip: np.ndarray,
    opponent_sword_tip: np.ndarray,
    dist_to_base: float,
    step_count: int,
    max_steps: int,
) -> np.ndarray:
    time_fraction = float(step_count) / float(max(1, max_steps))

    obs = np.concatenate(
        [
            agent_q.astype(np.float32),
            agent_dq.astype(np.float32),
            agent_sword_tip.astype(np.float32),
            opponent_sword_tip.astype(np.float32),
            np.array([dist_to_base], dtype=np.float32),
            np.array([time_fraction], dtype=np.float32),
        ]
    ).astype(np.float32)

    return obs
