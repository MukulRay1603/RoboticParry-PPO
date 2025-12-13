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


def compute_reward_and_done_advanced(
    client_id: int,
    agent_sword_id: int,
    opponent_sword_id: int,
    agent_body_id: int,
    agent_base_pos: np.ndarray,
    agent_sword_tip: np.ndarray,
    opponent_sword_tip: np.ndarray,
    prev_agent_tip: np.ndarray,
    agent_q: np.ndarray,
    agent_dq: np.ndarray,
    neutral_q: np.ndarray,
    attack_speed: float,
    attack_speed_along: float,
    threat_scalar: float,
    parry_angle_deg: float,
    parry_zone_radius: float = 0.7,
) -> Tuple[float, bool, bool, Dict[str, Any], np.ndarray]:
    """
    Advanced reward for SamuraiRL.

    Extends the simple distance/contact reward with:
      - parry angle quality,
      - timing vs. threatening swings,
      - stance maintenance,
      - gentle anti-camping shaping,
      - control cost.
    """
    dist_swords = float(np.linalg.norm(agent_sword_tip - opponent_sword_tip))
    dist_to_base = float(np.linalg.norm(opponent_sword_tip - agent_base_pos))
    parry_active = dist_to_base < parry_zone_radius

    reward = 0.0

    # Encourage being near the opponent sword tip, especially in the parry zone.
    if parry_active:
        reward += 1.5 * np.exp(-4.0 * dist_swords)
    else:
        reward += 0.15 * np.exp(-2.0 * dist_swords)

    # --- Motion bonus & anti-camping ---
    if prev_agent_tip is None:
        movement = 0.0
        new_prev_agent_tip = agent_sword_tip.copy()
    else:
        movement = float(np.linalg.norm(agent_sword_tip - prev_agent_tip))
        new_prev_agent_tip = agent_sword_tip.copy()

    # Reward some motion of the sword tip (small coefficient to avoid flailing)
    reward += 0.01 * movement

    # If the agent is in the parry zone, threat is low, and it barely moves,
    # apply a very small "camping" penalty.
    if parry_active and threat_scalar < 0.05 and movement < 0.002:
        reward -= 0.02

    # --- Stance maintenance (posture / "kamae") ---
    # Penalize large deviations from a neutral ready stance; stronger when
    # there is no strong threat so the agent doesn't crouch or sprawl.
    stance_error = agent_q - neutral_q
    stance_cost = float(np.sum(stance_error ** 2))
    if threat_scalar < 0.2:
        reward -= 0.003 * stance_cost
    else:
        reward -= 0.0015 * stance_cost

    # --- Control cost ---
    control_cost = 0.01 * float(np.sum(agent_dq ** 2))
    reward -= control_cost

    # Contact checks for parry / hit
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

    # --- Hit / failed defense ---
    if len(contact_body_hit) > 0:
        reward -= 5.0
        done_hit = True
    else:
        near_body = dist_to_base < 0.35
        approaching = attack_speed_along > 0.15
        if parry_active and approaching and near_body and len(contact_parry) == 0:
            reward -= 0.5

    # --- Parry geometry / timing ---
    parry_registered = False
    good_parry_geom = False

    # Prefer roughly 30–120 degrees between blades (crossed, not parallel)
    good_parry_angle = 30.0 <= parry_angle_deg <= 120.0
    approaching = attack_speed_along > 0.15

    if parry_active and len(contact_parry) > 0:
        parry_registered = True
        # Base reward for any sword–sword contact in the active zone
        reward += 3.0

        if threat_scalar > 0.1 and approaching:
            reward += 1.0 * float(threat_scalar)

        if good_parry_angle:
            reward += 2.0
            good_parry_geom = True
        else:
            reward += 0.5

        done_parry = False

    info: Dict[str, Any] = {
        "dist_swords": dist_swords,
        "dist_to_base": dist_to_base,
        "parry_active": parry_active,
        "contact_parry": len(contact_parry) > 0,
        "contact_body_hit": len(contact_body_hit) > 0,
        "parry_angle_deg": parry_angle_deg,
        "good_parry_angle": good_parry_angle,
        "attack_speed": attack_speed,
        "attack_speed_along": attack_speed_along,
        "threat_scalar": threat_scalar,
        "parry_registered": parry_registered,
        "good_parry_geom": good_parry_geom,
        "stance_cost": stance_cost,
        "movement": movement,
    }

    return float(reward), done_hit, done_parry, info, new_prev_agent_tip
