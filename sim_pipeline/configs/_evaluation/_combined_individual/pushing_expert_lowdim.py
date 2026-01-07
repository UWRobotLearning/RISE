from dataclasses import dataclass
from sim_pipeline.configs._evaluation.eval_base import EvaluationConfig
from sim_pipeline.configs.env.base import BaseEnvConfig
from sim_pipeline.configs.env.robomimic_square_ood import RobommimicSquareEnvOODConfig
from sim_pipeline.configs.exp._eval._combined_individual.pushing_expert_diffusion_lowdim import DiffusionEvalExpConfig
from sim_pipeline.configs.exp._eval.evaluation import EvalExpConfig
from sim_pipeline.configs.training._diffusion.base_diffusion_rohan import DiffusionTrainingRohanConfig

@dataclass
class RobomimicEvalConfig(EvaluationConfig):
    name: str ='diffusion_combined_lowdim_rohan'
    env: BaseEnvConfig = RobommimicSquareEnvOODConfig()
    exp: EvalExpConfig = DiffusionEvalExpConfig()
    training: DiffusionTrainingRohanConfig = DiffusionTrainingRohanConfig()