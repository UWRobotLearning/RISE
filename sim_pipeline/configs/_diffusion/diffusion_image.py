from dataclasses import dataclass
import sim_pipeline.configs.default as default
from sim_pipeline.configs.env.base import BaseEnvConfig
from sim_pipeline.configs.env.robomimic_square_image import RobomimicSquareImageEnvConfig
from sim_pipeline.configs.training._diffusion.diffusion_image import DiffusionImageTrainingConfig
from sim_pipeline.configs.exp.diffusion_train import DiffusionExpConfig

@dataclass
class DiffusionImageConfig(default.TopLevelConfig):
    name: str = 'diffusion_image_square'
    env: BaseEnvConfig = RobomimicSquareImageEnvConfig()
    exp: DiffusionExpConfig = DiffusionExpConfig()
    training: DiffusionImageTrainingConfig = DiffusionImageTrainingConfig()
    
