from dataclasses import dataclass, field
from sim_pipeline.configs.constants import ImitationAlgorithm
from sim_pipeline.configs.training._imitation_learning.base_imitation import ImitationTrainingConfig

@dataclass
class BCRNNImageTrainingConfig(ImitationTrainingConfig):
    algo: ImitationAlgorithm = ImitationAlgorithm.BC_RNN
    rgb_keys: list[str] = field(default_factory=lambda: [
        "agentview_image",
        "robot0_eye_in_hand_image"        
    ])
    
    batch_size: int = 16
    num_epochs: int = 600
    epoch_every_n_steps: int = 500