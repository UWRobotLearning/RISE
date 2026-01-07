from dataclasses import dataclass
from typing import Tuple, Optional
from pathlib import Path
from sim_pipeline.configs.constants import CALIBRATION_PATH

@dataclass
class RobotConfigFranka:
    control_hz: int = 10
    DoF: int = 2
    gripper: bool = False

    # Franka model: 'panda' 'fr3'
    robot_type: str = "panda"

    # randomize arm position on reset
    randomize_ee_on_reset: bool = True
    # allows user to pause to reset reset of the environment
    pause_after_reset: bool = False

    # observation space configuration
    qpos: bool = True
    ee_pos: bool = True
    imgs: bool = False
    normalize: bool = False

    # pass IP if not running on NUC "localhost" if running on NUC None if running sim
    ip_address: Optional[str] = None
    # specify path length if resetting after a fixed length
    max_path_length: int = 40
    # camera type to use: 'realsense' 'zed'
    camera_model: Optional[str] = None
    camera_resolution: Tuple[int, int] = (480, 640)  # (128 128) -> HxW
    calibration_file: str = CALIBRATION_PATH
    # Mujoco: model name
    model_name: str = "base_franka"
    on_screen_rendering: bool = False