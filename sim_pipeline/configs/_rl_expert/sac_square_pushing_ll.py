from dataclasses import dataclass
import sim_pipeline.configs.default as default

from sim_pipeline.configs.env.robomimic_square_pushing_ll import PushingLLRobomimicSquareEnvConfig
from sim_pipeline.configs.training._reinforcement_learning.robomimic_sac import SACTrainingConfig

@dataclass
class Config(default.TopLevelConfig):
    name: str = 'sac_square_pushing_ll'
    env: PushingLLRobomimicSquareEnvConfig = PushingLLRobomimicSquareEnvConfig()
    training: SACTrainingConfig = SACTrainingConfig()
