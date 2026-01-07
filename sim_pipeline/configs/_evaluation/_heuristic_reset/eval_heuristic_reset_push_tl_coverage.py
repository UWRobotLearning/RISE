from dataclasses import dataclass
from sim_pipeline.configs._evaluation.eval_base import EvaluationConfig
from sim_pipeline.configs.env.base import BaseEnvConfig
from sim_pipeline.configs.env.robomimic_square_coverage import RobomimicSquareEnvOODConfig
from sim_pipeline.configs.exp._eval.evaluation import EvalExpConfig
from sim_pipeline.configs.exp._eval.heuristic_reset_coverage_tl_eval import HeuristicResetSplitEvalExpConfig
from sim_pipeline.configs.training._diffusion.base_diffusion import DiffusionTrainingConfig

@dataclass
class RobomimicEvalConfig(EvaluationConfig):
    env: BaseEnvConfig = RobomimicSquareEnvOODConfig()
    exp: EvalExpConfig = HeuristicResetSplitEvalExpConfig()
    training: DiffusionTrainingConfig = DiffusionTrainingConfig()