from dataclasses import dataclass
from typing import Any
from sim_pipeline.configs.env.base import BaseEnvConfig
from sim_pipeline.configs.data.base_data import BaseDataConfig
from sim_pipeline.configs.exp.base import BaseExpConfig
from sim_pipeline.configs.training._reinforcement_learning.base_rl import RLTrainingConfig

@dataclass
class TopLevelConfig:
    name: str = 'default_exp'
    debug: bool = False
    env: Any = BaseEnvConfig()
    exp: Any = BaseExpConfig()
    training: Any = RLTrainingConfig()
    data: BaseDataConfig = BaseDataConfig()