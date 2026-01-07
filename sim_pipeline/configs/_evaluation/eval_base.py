from dataclasses import dataclass
from typing import Any
import sim_pipeline.configs.default as default

from sim_pipeline.configs.env.base import BaseEnvConfig
from sim_pipeline.configs.exp._eval.evaluation import EvalExpConfig

@dataclass
class EvaluationConfig(default.TopLevelConfig):
    name: str = 'eval_default'
    env: BaseEnvConfig = BaseEnvConfig()
    exp: EvalExpConfig = EvalExpConfig()