from dataclasses import dataclass
from sim_pipeline.configs._evaluation.eval_base import EvaluationConfig
from sim_pipeline.configs.env.base import BaseEnvConfig
from sim_pipeline.configs.env.robomimic_square import RobomimicSquareEnvConfig
from sim_pipeline.configs.exp._eval._combined_individual.square_pushing_diffusion_lowdim import DiffusionEvalExpConfig
from sim_pipeline.configs.exp._eval.evaluation import EvalExpConfig
from sim_pipeline.configs.training._diffusion.base_diffusion import DiffusionTrainingConfig

@dataclass
class RobomimicEvalConfig(EvaluationConfig):
    name: str ='diffusion_square_pushing_lowdim'
    env: BaseEnvConfig = RobomimicSquareEnvConfig()
    exp: EvalExpConfig = DiffusionEvalExpConfig()
    training: DiffusionTrainingConfig = DiffusionTrainingConfig()