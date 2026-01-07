from dataclasses import dataclass
import sim_pipeline.configs.default as default

from sim_pipeline.configs.env.robomimic_square import RobomimicSquareEnvConfig
from sim_pipeline.configs.training._reinforcement_learning.robomimic_sac import SACTrainingConfig

@dataclass
class Config(default.TopLevelConfig):
    name: str = 'sac_square'
    env: RobomimicSquareEnvConfig = RobomimicSquareEnvConfig()
    training: SACTrainingConfig = SACTrainingConfig()
