from dataclasses import dataclass
from sim_pipeline.configs._imitation_learning.robomimic_imitation_base import RobomimicImitationBaseConfig
from sim_pipeline.configs.training._imitation_learning.base_imitation import ImitationTrainingConfig

@dataclass
class RobomimicImitationBaseConfig(RobomimicImitationBaseConfig):
    name: str = 'square_bc_rnn_lowdim'
