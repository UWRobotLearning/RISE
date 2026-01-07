from dataclasses import dataclass
from sim_pipeline.configs._evaluation.eval_rl import RLEvalConfig
from sim_pipeline.configs.env.base import BaseEnvConfig
from sim_pipeline.configs.env.robomimic_square_pushing import PushingRobomimicSquareEnvConfig
from sim_pipeline.configs.exp._eval.sb_eval import RLEvalExpConfig
from sim_pipeline.configs.exp._eval.evaluation import EvalExpConfig

@dataclass
class PushingRLEvalConfig(RLEvalConfig):
    env: BaseEnvConfig = PushingRobomimicSquareEnvConfig()
    exp: EvalExpConfig = RLEvalExpConfig()