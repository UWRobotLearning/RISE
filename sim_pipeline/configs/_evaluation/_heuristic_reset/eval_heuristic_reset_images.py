from dataclasses import dataclass
from sim_pipeline.configs._evaluation.eval_base import EvaluationConfig
from sim_pipeline.configs.env.base import BaseEnvConfig
from sim_pipeline.configs.env.robomimic_square_image_ood import RobommimicSquareImageEnvOODConfig
from sim_pipeline.configs.exp._eval.evaluation import EvalExpConfig
from sim_pipeline.configs.exp._eval.heuristic_reset_images_eval import HeuristicResetImagesEvalExpConfig
from sim_pipeline.configs.training._diffusion.diffusion_image import DiffusionImageTrainingConfig

@dataclass
class RobomimicEvalConfig(EvaluationConfig):
    env: BaseEnvConfig = RobommimicSquareImageEnvOODConfig()
    exp: EvalExpConfig = HeuristicResetImagesEvalExpConfig()
    training: DiffusionImageTrainingConfig = DiffusionImageTrainingConfig()