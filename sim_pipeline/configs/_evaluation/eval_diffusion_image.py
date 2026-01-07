from dataclasses import dataclass
from sim_pipeline.configs._evaluation.eval_base import EvaluationConfig
from sim_pipeline.configs.env.base import BaseEnvConfig
from sim_pipeline.configs.env.robomimic_base import RobomimicEnvConfig
from sim_pipeline.configs.exp._eval.diffusion_eval import DiffusionEvalExpConfig
from sim_pipeline.configs.exp._eval.evaluation import EvalExpConfig
from sim_pipeline.configs.training._diffusion.diffusion_image import DiffusionImageTrainingConfig

@dataclass
class RobomimicEvalConfig(EvaluationConfig):
    env: BaseEnvConfig = RobomimicEnvConfig()
    exp: EvalExpConfig = DiffusionEvalExpConfig()
    training: DiffusionImageTrainingConfig = DiffusionImageTrainingConfig()