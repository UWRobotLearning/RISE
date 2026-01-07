import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List
from sim_pipeline.configs.constants import RobomimicEnvType
from sim_pipeline.configs.env.robomimic_tool import RobomimicToolHangEnvConfig

@dataclass
class RobomimicToolHangEnvConfig(RobomimicToolHangEnvConfig):
    env_name: RobomimicEnvType = RobomimicEnvType.TOOL_HANG
    render_offscreen: bool = True
    custom_obj_init: bool = True
    init_obj_range: dict[str, list[list[float]]] = field(default_factory=lambda: 
    {
    'standObject':
    [
        [-0.08, -0.08], 
        [0.0, 0.0],
        [0.0, 0.0],
    ],
    'frameObject':
    [
        [-0.15, -0.02], 
        [-0.26, -0.22],
        [(-np.pi / 2) + (np.pi / 6) - np.pi / 18, (-np.pi / 2) + (np.pi / 6) + np.pi / 18],
    ],
    'toolObject':
    [
        [0.02, 0.15], 
        [-0.22, -0.18],
        [(-np.pi / 2) - (np.pi / 9.0) - np.pi / 18, (-np.pi / 2) - (np.pi / 9.0) + np.pi / 18],
    ],
    })
