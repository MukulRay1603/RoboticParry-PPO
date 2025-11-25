from typing import Dict, Any, Tuple

import numpy as np
import pybullet as p


def compute_reward_and_done(
    client_id: int,
    agent_sword_id: int,
    opponent_sword_id: int,
    agent_body_id: int,
    agent_base_pos: np.ndarray,
    agent_sword_tip: np.ndarray,
    opponent_sword_tip: np.ndarray,
    prev_agent_tip: np.ndarray,
) -> Tuple[float, bool, bool, Dict[str, Any]]:
    """
    Shared reward logic for the SamuraiRL environment.

    Returns:
        reward, done_hit, done_parry, info
    """
    dist_swords = float(np.linalg.norm(agent_sword_tip - opponent_sword_tip))
    dist_to_base = float(np.linalg.norm(opponent_sword_tip - agent_base_pos))

    parry_zone_radius = 0.7
    parry_active = dist_to_base < parry_zone_radius

    reward = 0.0

    # Encourage being close to the opponent sword tip, but not just camping
    if parry_active:
        reward += 1.5 * np.exp(-4.0 * dist_swords)
    else:
        reward += 0.2 * np.exp(-2.0 * dist_swords)

    # Motion bonus for the agent sword tip
    movement = float(np.linalg.norm(agent_sword_tip - prev_agent_tip))
    reward += 0.02 * movement

    # Contact checks
    contact_parry = p.getContactPoints(
        bodyA=agent_sword_id,
        bodyB=opponent_sword_id,
        physicsClientId=client_id,
    )
    contact_body_hit = p.getContactPoints(
        bodyA=opponent_sword_id,
        bodyB=agent_body_id,
        physicsClientId=client_id,
    )

    done_hit = False
    done_parry = False

    if len(contact_body_hit) > 0:
        reward -= 5.0
        done_hit = True

    if parry_active and len(contact_parry) > 0:
        reward += 5.0
        done_parry = False  # successful parry but continue the episode

    info: Dict[str, Any] = {
        "dist_swords": dist_swords,
        "dist_to_base": dist_to_base,
        "parry_active": parry_active,
        "contact_parry": len(contact_parry) > 0,
        "contact_body_hit": len(contact_body_hit) > 0,
    }

    return float(reward), done_hit, done_parry, info
