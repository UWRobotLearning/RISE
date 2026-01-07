from dataclasses import dataclass
from typing import Any
from sim_pipeline.configs._evaluation.eval_base import EvaluationConfig
from sim_pipeline.configs.env.base import BaseEnvConfig
from sim_pipeline.configs.env.robomimic_square import RobomimicSquareEnvConfig
from sim_pipeline.configs.exp._eval.sb_eval import RLEvalExpConfig
from sim_pipeline.configs.exp._eval.evaluation import EvalExpConfig
from sim_pipeline.configs.training._reinforcement_learning.robomimic_sac import SACTrainingConfig

@dataclass
class RLEvalConfig(EvaluationConfig):
    env: BaseEnvConfig = RobomimicSquareEnvConfig()
    exp: EvalExpConfig = RLEvalExpConfig()
    training: Any = SACTrainingConfig()