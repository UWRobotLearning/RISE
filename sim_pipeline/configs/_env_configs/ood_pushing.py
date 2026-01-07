from dataclasses import dataclass
import sim_pipeline.configs.default as default

from sim_pipeline.configs.env.robomimic_square_ood_pushing import RobommimicSquareEnvOODPushingConfig
from sim_pipeline.configs.training._reinforcement_learning.robomimic_sac import SACTrainingConfig

@dataclass
class Config(default.TopLevelConfig):
    name: str = ''
    env: RobommimicSquareEnvOODPushingConfig = RobommimicSquareEnvOODPushingConfig()
