from dataclasses import dataclass
from typing import Any
from sim_pipeline.configs._evaluation.eval_base import EvaluationConfig
from sim_pipeline.configs.env.base import BaseEnvConfig
from sim_pipeline.configs.env.robomimic_base import RobomimicEnvConfig
from sim_pipeline.configs.exp._eval.robomimic_eval import RobomimicEvalExpConfig
from sim_pipeline.configs.exp._eval.evaluation import EvalExpConfig
from sim_pipeline.configs.training._imitation_learning._offline_rl.iql_diffusion_image import IQLDiffusionImageTrainingConfig

@dataclass
class RobomimicEvalConfig(EvaluationConfig):
    env: BaseEnvConfig = RobomimicEnvConfig()
    exp: EvalExpConfig = RobomimicEvalExpConfig()
    training: IQLDiffusionImageTrainingConfig = IQLDiffusionImageTrainingConfig()