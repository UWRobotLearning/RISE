from dataclasses import dataclass

from sim_pipeline.configs.env.robomimic_square import RobomimicSquareEnvConfig
from sim_pipeline.configs.training._reinforcement_learning.robomimic_ppo import PPOTrainingConfig
import sim_pipeline.configs.default as default

@dataclass
class Config(default.TopLevelConfig):
    name: str = 'square_ppo'
    env: RobomimicSquareEnvConfig = RobomimicSquareEnvConfig()
    training: PPOTrainingConfig = PPOTrainingConfig()
