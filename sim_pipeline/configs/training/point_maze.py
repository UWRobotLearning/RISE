from dataclasses import dataclass, field
from enum import Enum
from sim_pipeline.configs.constants import RLAlgorithm

@dataclass
class TrainingConfig:
    algo: RLAlgorithm = RLAlgorithm.PPO
    policy_kwargs: dict = field(default_factory=lambda: {

    })
    policy: str = 'MultiInputPolicy'
