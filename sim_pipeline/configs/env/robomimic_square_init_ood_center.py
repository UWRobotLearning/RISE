from dataclasses import dataclass, field
from sim_pipeline.configs.env.robomimic_square import RobomimicSquareEnvConfig

@dataclass
class LLOODRobomimicSquareEnvConfig(RobomimicSquareEnvConfig):
    custom_ee_init: bool = False
    # x, y, z, [min, max]
    init_ee_range: list[list[float]] = field(default_factory=lambda: [
        [-0.12, -0.08], 
        [-0.1, 0.05],
        [0.9, 1.012]
    ])
    custom_obj_init: bool = True
    init_obj_range: dict[str, list[list[float]]] = field(default_factory=lambda: 
    {
        'SquareNut':
        [
            [-0.05, 0.0],
            [0.05, 0.15],
            
        ],
    })