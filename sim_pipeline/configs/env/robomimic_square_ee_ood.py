from dataclasses import dataclass, field
from sim_pipeline.configs.env.robomimic_square import RobomimicSquareEnvConfig

@dataclass
class RobommimicSquareEEOODEnvConfig(RobomimicSquareEnvConfig):
    custom_ee_init: bool = True
    # x, y, z, [min, max]
    init_ee_range: list[list[float]] = field(default_factory=lambda: [
        [-0.32, -0.34], 
        [0.34, 0.35],
        [0.65, 0.65]
    ])
