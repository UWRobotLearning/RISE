from dataclasses import dataclass, field
from typing import Dict, List
from sim_pipeline.configs.constants import RobomimicEnvType
from sim_pipeline.data_manager.dataset_metadata_enums import ObsType
from sim_pipeline.configs.env.robomimic_base import RobomimicEnvConfig

@dataclass
class RobomimicSquareImageEnvConfig(RobomimicEnvConfig):
    env_name: RobomimicEnvType = RobomimicEnvType.NUT_ASSEMBLY_SQUARE
    render_offscreen: bool = True
    
    obs_type: ObsType = ObsType.IMAGE