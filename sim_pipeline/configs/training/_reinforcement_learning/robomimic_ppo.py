from dataclasses import dataclass, field
from enum import Enum
from sim_pipeline.configs.constants import RLAlgorithm

from sim_pipeline.configs.training._reinforcement_learning.base_rl import RLTrainingConfig

@dataclass
class PPOTrainingConfig(RLTrainingConfig):
    algo: RLAlgorithm = RLAlgorithm.PPO
    policy_kwargs: dict = field(default_factory=lambda: {
        'net_arch': [128, 128],
    })
    policy: str = 'MlpPolicy'
