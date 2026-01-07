from dataclasses import dataclass
from sim_pipeline.configs._evaluation.eval_base import EvaluationConfig
from sim_pipeline.configs.env.base import BaseEnvConfig
from sim_pipeline.configs.env.robomimic_square_ee_ood import RobommimicSquareEEOODEnvConfig
from sim_pipeline.configs.exp._eval.pd_eval import PDEvalExpConfig  

@dataclass
class RobomimicEvalConfig(EvaluationConfig):
    env: BaseEnvConfig = RobommimicSquareEEOODEnvConfig()
    exp: PDEvalExpConfig = PDEvalExpConfig()