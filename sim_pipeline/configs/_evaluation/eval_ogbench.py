from dataclasses import dataclass
import sim_pipeline.configs.default as default
from sim_pipeline.configs.env.base import BaseEnvConfig
from sim_pipeline.configs.env.robomimic_square import RobomimicSquareEnvConfig
from sim_pipeline.configs.training._ogbench.crl import CRLTrainingConfig
from sim_pipeline.configs.exp._eval.ogbench_eval import OGBenchEvalExpConfig

@dataclass
class OGBenchBaseConfig(default.TopLevelConfig):
    name: str = 'ogbench_square'
    env: BaseEnvConfig = RobomimicSquareEnvConfig()
    exp: OGBenchEvalExpConfig = OGBenchEvalExpConfig()
    training: CRLTrainingConfig = CRLTrainingConfig()