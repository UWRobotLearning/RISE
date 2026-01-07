from dataclasses import dataclass
from sim_pipeline.configs._evaluation.eval_base import EvaluationConfig
from sim_pipeline.configs.env.base import BaseEnvConfig
from sim_pipeline.configs.env.robomimic_square import RobomimicSquareEnvConfig
from sim_pipeline.configs.exp._eval._combined_individual.square_expert_diffusion_lowdim_rohan import DiffusionEvalExpConfig
from sim_pipeline.configs.exp._eval.evaluation import EvalExpConfig
from sim_pipeline.configs.training._diffusion.base_diffusion_rohan import DiffusionTrainingRohanConfig

@dataclass
class RobomimicEvalConfig(EvaluationConfig):
    name: str ='diffusion_square_expert_lowdim_rohan'
    env: BaseEnvConfig = RobomimicSquareEnvConfig()
    exp: EvalExpConfig = DiffusionEvalExpConfig()
    training: DiffusionTrainingRohanConfig = DiffusionTrainingRohanConfig()