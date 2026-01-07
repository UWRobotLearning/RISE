import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List
from sim_pipeline.configs.constants import RobomimicEnvType
from sim_pipeline.configs.env.robomimic_base import RobomimicEnvConfig

@dataclass
class RobomimicCoffeeEnvConfig(RobomimicEnvConfig):
    env_name: RobomimicEnvType = RobomimicEnvType.COFFEE
    render_offscreen: bool = True
