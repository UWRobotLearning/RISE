from dataclasses import dataclass
import sim_pipeline.configs.default as default
from sim_pipeline.configs.env.base import BaseEnvConfig
from sim_pipeline.configs.env.robomimic_square import RobomimicSquareEnvConfig
from sim_pipeline.configs.training._imitation_learning._offline_rl.base_iql import IQLTrainingConfig
from sim_pipeline.configs.exp._eval.robomimic_eval import RobomimicEvalExpConfig

@dataclass
class RobomimicIQLDiagnosticConfig(default.TopLevelConfig):
    name: str = 'iql_diagnostic'
    env: BaseEnvConfig = RobomimicSquareEnvConfig()
    exp: RobomimicEvalExpConfig = RobomimicEvalExpConfig()
    training: IQLTrainingConfig = IQLTrainingConfig()