from dataclasses import dataclass, field
from enum import Enum
from sim_pipeline.configs.constants import RLAlgorithm

@dataclass
class RLTrainingConfig:
    algo: RLAlgorithm = RLAlgorithm.PPO
    total_timesteps: int = 1_000_000
    policy_kwargs: dict = field(default_factory=lambda: {

    })
    policy: str = 'MlpPolicy'

    resume: bool = True