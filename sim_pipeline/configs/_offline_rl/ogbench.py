from dataclasses import dataclass
import sim_pipeline.configs.default as default
from sim_pipeline.configs.env.base import BaseEnvConfig
from sim_pipeline.configs.env.robomimic_square import RobomimicSquareEnvConfig
from sim_pipeline.configs.training._ogbench.ogbench import OGBenchTrainingConfig
from sim_pipeline.configs.exp.robomimic_train import RobomimicTrainExpConfig

@dataclass
class OGBenchBaseConfig(default.TopLevelConfig):
    name: str = 'ogbench_square'
    env: BaseEnvConfig = RobomimicSquareEnvConfig()
    exp: RobomimicTrainExpConfig = RobomimicTrainExpConfig()
    training: OGBenchTrainingConfig = OGBenchTrainingConfig()