from dataclasses import dataclass
from sim_pipeline.configs._evaluation.eval_base import EvaluationConfig
from sim_pipeline.configs.env.base import BaseEnvConfig
from sim_pipeline.configs.env.robomimic_square import RobomimicSquareEnvConfig
from sim_pipeline.configs.exp._eval._combined_individual.square_combined_diffusion_images import DiffusionEvalExpConfig
from sim_pipeline.configs.exp._eval.evaluation import EvalExpConfig
from sim_pipeline.configs.training._diffusion.diffusion_image import DiffusionImageTrainingConfig

@dataclass
class RobomimicEvalConfig(EvaluationConfig):
    name: str ='diffusion_square_combined_image'
    env: BaseEnvConfig = RobomimicSquareEnvConfig()
    exp: EvalExpConfig = DiffusionEvalExpConfig()
    training: DiffusionImageTrainingConfig = DiffusionImageTrainingConfig()