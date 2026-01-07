from dataclasses import dataclass
from typing import Any
from sim_pipeline.configs._evaluation.eval_base import EvaluationConfig
from sim_pipeline.configs.env.base import BaseEnvConfig
from sim_pipeline.configs.env.robomimic_square import RobomimicSquareEnvConfig
from sim_pipeline.configs.exp._eval.robomimic_eval import RobomimicEvalExpConfig
from sim_pipeline.configs.exp._eval.evaluation import EvalExpConfig

@dataclass
class RobomimicEvalConfig(EvaluationConfig):
    env: BaseEnvConfig = RobomimicSquareEnvConfig()
    exp: EvalExpConfig = RobomimicEvalExpConfig()