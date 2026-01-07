from dataclasses import dataclass
import sim_pipeline.configs.default as default
from sim_pipeline.configs.env.base import BaseEnvConfig
from sim_pipeline.configs.env.robomimic_square import RobomimicSquareEnvConfig
from sim_pipeline.configs.training._imitation_learning.base_imitation import ImitationTrainingConfig
from sim_pipeline.configs.training._imitation_learning.discriminator import DiscriminatorTrainingConfig
from sim_pipeline.configs.exp.robomimic_train import RobomimicTrainExpConfig
from sim_pipeline.configs.data.base_data import BaseDataConfig

@dataclass
class RobomimicDiscriminatorConfig(default.TopLevelConfig):
    name: str = 'discriminator_square'
    env: BaseEnvConfig = RobomimicSquareEnvConfig()
    exp: RobomimicTrainExpConfig = RobomimicTrainExpConfig()
    training: ImitationTrainingConfig = DiscriminatorTrainingConfig()