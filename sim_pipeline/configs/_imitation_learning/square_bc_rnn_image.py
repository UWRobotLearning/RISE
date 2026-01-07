from dataclasses import dataclass

from sim_pipeline.configs._imitation_learning.robomimic_imitation_base import RobomimicImitationBaseConfig
from sim_pipeline.configs.training._imitation_learning.bcrnn_image import BCRNNImageTrainingConfig

@dataclass
class RobomimicImitationBaseConfig(RobomimicImitationBaseConfig):
    name: str = 'square_bc_rnn_image'
    training: BCRNNImageTrainingConfig = BCRNNImageTrainingConfig()
