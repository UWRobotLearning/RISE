from dataclasses import dataclass, field
from typing import Dict, List
from sim_pipeline.configs.constants import RobomimicEnvType
from sim_pipeline.configs.env.robomimic_base import RobomimicEnvConfig

@dataclass
class RobomimicThreePieceAssemblyEnvConfig(RobomimicEnvConfig):
    env_name: RobomimicEnvType = RobomimicEnvType.THREE_PIECE_ASSEMBLY
    render_offscreen: bool = True
    
    custom_obj_init: bool = True
    init_obj_range: dict[str, list[list[float]]] = field(default_factory=lambda: 
    {
        'Base':
        [
            [0.0, 0.0], 
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        'Piece1':
        [
            [-0.22, -0.0], 
            [0.0, 0.22],
            [1.3708, 1.7708],
        ],
        # 'Piece2':
        # [
        #     [-0.22, 0.22], 
        #     [-0.22, 0.22],
        #     [1.5708, 1.5708],
        # ]
    })