from dataclasses import dataclass
import sim_pipeline.configs.default as default

from sim_pipeline.configs.env.robomimic_square_pushing import PushingRobomimicSquareEnvConfig
from sim_pipeline.configs.training._reinforcement_learning.robomimic_sac import SACTrainingConfig

@dataclass
class Config(default.TopLevelConfig):
    name: str = 'sac_square_pushing'
    env: PushingRobomimicSquareEnvConfig = PushingRobomimicSquareEnvConfig()
    training: SACTrainingConfig = SACTrainingConfig()
