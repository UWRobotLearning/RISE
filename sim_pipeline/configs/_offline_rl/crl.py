from dataclasses import dataclass
import sim_pipeline.configs.default as default
from sim_pipeline.configs.env.base import BaseEnvConfig
from sim_pipeline.configs.env.robomimic_square import RobomimicSquareEnvConfig
from sim_pipeline.configs.training._ogbench.crl import CRLTrainingConfig
from sim_pipeline.configs.exp.robomimic_train import RobomimicTrainExpConfig
from sim_pipeline.configs._offline_rl.ogbench import OGBenchBaseConfig

@dataclass
class CRLBaseConfig(OGBenchBaseConfig):
    name: str = 'crl'
    training: CRLTrainingConfig = CRLTrainingConfig()