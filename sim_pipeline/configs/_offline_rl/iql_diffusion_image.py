from dataclasses import dataclass
import sim_pipeline.configs.default as default
from sim_pipeline.configs.env.base import BaseEnvConfig
from sim_pipeline.configs.env.robomimic_square import RobomimicSquareEnvConfig
from sim_pipeline.configs.training._imitation_learning._offline_rl.iql_diffusion_image import IQLDiffusionImageTrainingConfig
from sim_pipeline.configs.exp.robomimic_train import RobomimicTrainExpConfig

@dataclass
class RobomimicIQLBaseConfig(default.TopLevelConfig):
    name: str = 'iql_diffusion_image_square'
    env: BaseEnvConfig = RobomimicSquareEnvConfig()
    exp: RobomimicTrainExpConfig = RobomimicTrainExpConfig()
    training: IQLDiffusionImageTrainingConfig = IQLDiffusionImageTrainingConfig()