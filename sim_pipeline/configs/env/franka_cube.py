from dataclasses import dataclass, field
from typing import Dict
from sim_pipeline.configs.env.franka_base import EnvConfigFranka

@dataclass
class EnvConfigFrankaCube(EnvConfigFranka):
    obj_id: str = "cube"
    obj_pose_noise_dict: Dict[str, Dict[str, float]] = field(default_factory=lambda:
    {
        "x": { "min": -0.02, "max": 0.05 },
        "y": { "min": -0.25, "max": 0.25 },
        "yaw": { "min": 0.0, "max": 1.5 },
    })