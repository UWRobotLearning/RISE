from dataclasses import dataclass, field
from sim_pipeline.configs.constants import ImitationAlgorithm

@dataclass
class QTransformerTrainingConfig:    
    low_dim_keys: list[str] = field(default_factory=lambda: [
    ])
    
    rgb_keys: list[str] = field(default_factory=lambda: [
        "agentview_image",
        "robot0_eye_in_hand_image",
    ])
    
    depth_keys: list[str] = field(default_factory=lambda: [
    ])
    
    batch_size: int = 16
    num_epochs: int = 10000
    learning_rate: float = 3e-4