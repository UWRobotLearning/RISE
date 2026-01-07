from dataclasses import dataclass
from sim_pipeline.configs._evaluation.eval_base import EvaluationConfig
from sim_pipeline.configs.env.base import BaseEnvConfig
from sim_pipeline.configs.env.robomimic_square import RobomimicSquareEnvConfig
from sim_pipeline.configs.exp._eval.robomimic_eval import RobomimicEvalExpConfig
from sim_pipeline.configs.exp._eval.evaluation import EvalExpConfig
from sim_pipeline.configs.training._imitation_learning._offline_rl.iql_diffusion_image import IQLDiffusionTrainingConfig

@dataclass
class RobomimicEvalConfig(EvaluationConfig):
    name: str ='iql_diffusion_push_300r_200e_expert_image_01_rew'
    env: BaseEnvConfig = RobomimicSquareEnvConfig()
    exp: EvalExpConfig = RobomimicEvalExpConfig()
    training: IQLDiffusionTrainingConfig = IQLDiffusionTrainingConfig()