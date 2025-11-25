import numpy as np
import pybullet as p


def create_sword(client_id: int, length: float = 0.7, mass: float = 0.5) -> int:
    """
    Create a simple box "sword" aligned so that its local +Y axis points along
    the blade from hilt to tip.
    """
    half_extents = [0.015, length * 0.5, 0.015]

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
    blade points roughly forward from the hand.

    We assume the Panda hand frame has its +Y axis roughly "forward" in the
    direction the gripper opens.
    """
    # Position of the hilt relative to the hand frame
    parent_frame_pos = [0.0, 0.04, 0.0]

    # Position of the sword COM relative to its own base so that
    # the hilt sits at the hand and the blade extends forward.
    child_frame_pos = [0.0, -length * 0.5, 0.0]

    # Rotate so that sword +Y points forward from the hand
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
    the local +Y axis of the sword body.
    """
    pos, orn = p.getBasePositionAndOrientation(sword_id, physicsClientId=client_id)
    pos = np.array(pos, dtype=np.float32)
    rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    local_tip = np.array([0.0, length * 0.5, 0.0], dtype=np.float32)
    tip_world = pos + rot @ local_tip
    return tip_world.astype(np.float32)
