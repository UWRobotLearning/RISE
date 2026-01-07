from dataclasses import dataclass
import sim_pipeline.configs.default as default
from sim_pipeline.configs.env.base import BaseEnvConfig
from sim_pipeline.configs.env.robomimic_square import RobomimicSquareEnvConfig
from sim_pipeline.configs.training._diffusion.base_diffusion import DiffusionTrainingConfig
from sim_pipeline.configs.exp.diffusion_train import DiffusionExpConfig

@dataclass
class DiffusionBaseConfig(default.TopLevelConfig):
    name: str = 'diffusion_square'
    env: BaseEnvConfig = RobomimicSquareEnvConfig()
    exp: DiffusionExpConfig = DiffusionExpConfig()
    training: DiffusionTrainingConfig = DiffusionTrainingConfig()
    
