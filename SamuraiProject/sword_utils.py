import numpy as np
import pybullet as p


def create_sword(client_id: int, length: float = 0.7, mass: float = 0.5) -> int:
    """
    Create a simple box "sword" aligned so that its local +X axis points along
    the blade from hilt to tip.

    Convention:
      - COM at (0, 0, 0)
      - Hilt  at X = -length/2
      - Tip   at X = +length/2
    """
    half_len = length * 0.5
    half_extents = [half_len, 0.015, 0.015]

    col_id = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=half_extents,
        physicsClientId=client_id,
    )
    vis_id = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=half_extents,
        physicsClientId=client_id,
    )

    sword_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=[0, 0, 0],
        physicsClientId=client_id,
    )

    # Make blade contact stable (less sliding / bouncing)
    p.changeDynamics(
        sword_id,
        -1,
        lateralFriction=1.0,
        spinningFriction=1.0,
        restitution=0.0,
        physicsClientId=client_id,
    )
    return sword_id


def attach_sword_to_hand(
    client_id: int,
    robot_id: int,
    ee_link: int,
    sword_id: int,
    length: float,
) -> None:
    """
    Attach the sword so that its hilt appears inside the Panda fingers and the
    blade points along the hand's +X axis.

    Convention matches create_sword():
      - local +X is blade from hilt -> tip
      - hilt at X = -length/2 (this is attached to the hand)
    """
    half_len = length * 0.5

    # Position of the hilt relative to the hand frame
    parent_frame_pos = [0.0, 0.0, 0.0]

    # Hilt at local X = -half_len
    child_frame_pos = [-half_len, 0.0, 0.0]

    # No rotation: sword +X follows hand +X
    child_frame_orn = p.getQuaternionFromEuler([0.0, 0.0, 0.0])

    p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=ee_link,
        childBodyUniqueId=sword_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=parent_frame_pos,
        childFramePosition=childFrame_pos,
        childFrameOrientation=child_frame_orn,
        physicsClientId=client_id,
    )

    # Disable collisions between the sword and its own robot
    num_links = p.getNumJoints(robot_id, physicsClientId=client_id)
    for link_idx in range(-1, num_links):
        p.setCollisionFilterPair(
            robot_id,
            sword_id,
            link_idx,
            -1,
            enableCollision=0,
            physicsClientId=client_id,
        )


def get_sword_tip_position(client_id: int, sword_id: int, length: float) -> np.ndarray:
    """
    Compute the world position of the sword tip assuming the blade is along
    the local +X axis of the sword body and the tip is at +length/2.
    """
    pos, orn = p.getBasePositionAndOrientation(sword_id, physicsClientId=client_id)
    pos = np.array(pos, dtype=np.float32)
    rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    local_tip = np.array([length * 0.5, 0.0, 0.0], dtype=np.float32)
    tip_world = pos + rot @ local_tip
    return tip_world.astype(np.float32)


def get_sword_tip_and_direction(
    client_id: int, sword_id: int, length: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (tip_position, direction_vector) for a sword.

    - Tip position is world coordinates of the blade tip.
    - Direction vector is the world-space +X axis of the sword (unit length),
      pointing from hilt toward tip.
    """
    pos, orn = p.getBasePositionAndOrientation(sword_id, physicsClientId=client_id)
    pos = np.array(pos, dtype=np.float32)
    rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    direction = rot[:, 0].astype(np.float32)
    direction /= max(1e-8, float(np.linalg.norm(direction)))
    local_tip = np.array([length * 0.5, 0.0, 0.0], dtype=np.float32)
    tip_world = pos + rot @ local_tip
    return tip_world.astype(np.float32), direction


def compute_sword_angle(dir_a: np.ndarray, dir_b: np.ndarray) -> float:
    """
    Compute the unsigned angle (in radians) between two direction vectors.
    """
    a = np.asarray(dir_a, dtype=np.float32)
    b = np.asarray(dir_b, dtype=np.float32)
    if a.shape != (3,) or b.shape != (3,):
        a = a.reshape(3)
        b = b.reshape(3)
    a /= max(1e-8, float(np.linalg.norm(a)))
    b /= max(1e-8, float(np.linalg.norm(b)))
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.arccos(dot))
