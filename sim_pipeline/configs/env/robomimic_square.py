from dataclasses import dataclass, field
from typing import Dict, List
from sim_pipeline.configs.constants import RobomimicEnvType
from sim_pipeline.configs.env.robomimic_base import RobomimicEnvConfig

@dataclass
class RobomimicSquareEnvConfig(RobomimicEnvConfig):
    env_name: RobomimicEnvType = RobomimicEnvType.NUT_ASSEMBLY_SQUARE
    render_offscreen: bool = True