import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List
from sim_pipeline.configs.constants import RobomimicEnvType
from sim_pipeline.configs.env.robomimic_base import RobomimicEnvConfig

@dataclass
class RobomimicThreadingEnvConfig(RobomimicEnvConfig):
    env_name: RobomimicEnvType = RobomimicEnvType.THREADING
    render_offscreen: bool = True
    custom_obj_init: bool = True
    init_obj_range: dict[str, list[list[float]]] = field(default_factory=lambda: 
    {
        'Needle':
        [
            [-0.2, 0.12], 
            [0.05, 0.25],
            [-2. * np.pi / 3., -np.pi / 3.],
        ],
        'Tripod':
        [
            [0.0, 0.0], 
            [-0.15, -0.15],
            [np.pi / 2., np.pi / 2.],
        ],
    })