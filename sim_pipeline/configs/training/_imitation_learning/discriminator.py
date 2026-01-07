from dataclasses import dataclass, field
from sim_pipeline.configs.constants import ImitationAlgorithm
from sim_pipeline.configs.training._imitation_learning.base_imitation import ImitationTrainingConfig

@dataclass
class DiscriminatorTrainingConfig(ImitationTrainingConfig):
    algo: ImitationAlgorithm = ImitationAlgorithm.DISCRIMINATOR
    
    batch_size: int = 100
    num_epochs: int = 10
    epoch_every_n_steps: int = 500