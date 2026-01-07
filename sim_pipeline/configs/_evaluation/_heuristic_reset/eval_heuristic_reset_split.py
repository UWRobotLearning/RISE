from dataclasses import dataclass
from sim_pipeline.configs._evaluation.eval_base import EvaluationConfig
from sim_pipeline.configs.env.base import BaseEnvConfig
from sim_pipeline.configs.env.robomimic_square_ood import RobommimicSquareEnvOODConfig
from sim_pipeline.configs.exp._eval.evaluation import EvalExpConfig
from sim_pipeline.configs.exp._eval.heuristic_reset_split_eval import HeuristicResetSplitEvalExpConfig
from sim_pipeline.configs.training._diffusion.base_diffusion import DiffusionTrainingConfig

@dataclass
class RobomimicEvalConfig(EvaluationConfig):
    env: BaseEnvConfig = RobommimicSquareEnvOODConfig()
    exp: EvalExpConfig = HeuristicResetSplitEvalExpConfig()
    training: DiffusionTrainingConfig = DiffusionTrainingConfig()