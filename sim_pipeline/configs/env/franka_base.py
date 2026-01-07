from dataclasses import dataclass, field
from typing import Dict, Tuple
from sim_pipeline.configs.constants import EnvType

from sim_pipeline.configs.env.base import BaseEnvConfig
from sim_pipeline.configs.robot.base_franka import RobotConfigFranka

@dataclass
class EnvConfigFranka(BaseEnvConfig):
    env_type: EnvType = EnvType.FRANKA
    obj_id: str = "cube"
    obj_pos_noise: bool = True
    goal_obj_delta_pose: Dict[str, float] = field(default_factory=lambda:
    { 
        "x": 0.25,
        # "y": 0.0,
    })
    obj_pose_noise_dict: Dict[str, Dict[str, float]] = field(default_factory=lambda:
    {
        "x": { "min": 0.00, "max": 0.00 },
        "y": { "min": 0.0, "max": 0.0 },
        "yaw": { "min": 0.0, "max": 0.0 },
    })
    safety_penalty: float = 0.0
    obs_keys: Tuple[str] = ("lowdim_ee", "lowdim_qpos", "obj_pose")
    flatten: bool = True
    
    robot_config: RobotConfigFranka = RobotConfigFranka()

