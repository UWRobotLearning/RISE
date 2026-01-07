from dataclasses import dataclass
import sim_pipeline.configs.default as default
from sim_pipeline.configs.env.base import BaseEnvConfig
from sim_pipeline.configs.env.robomimic_square import RobomimicSquareEnvConfig
from sim_pipeline.configs.training._imitation_learning.base_imitation import ImitationTrainingConfig
from sim_pipeline.configs.exp.robomimic_train import RobomimicTrainExpConfig

@dataclass
class RobomimicImitationBaseConfig(default.TopLevelConfig):
    name: str = 'imitation_square'
    env: BaseEnvConfig = RobomimicSquareEnvConfig()
    exp: RobomimicTrainExpConfig = RobomimicTrainExpConfig()
    training: ImitationTrainingConfig = ImitationTrainingConfig()