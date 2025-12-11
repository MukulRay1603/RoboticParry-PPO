import numpy as np
import pybullet as p
from typing import Tuple


def get_default_camera(client_id: int) -> Tuple[int, int, list, list]:
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
        physicsClientId=client_id,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=1.0,
        nearVal=0.1,
        farVal=5.0,
        physicsClientId=client_id,
    )
    width = 640
    height = 480
    return width, height, view_matrix, proj_matrix
