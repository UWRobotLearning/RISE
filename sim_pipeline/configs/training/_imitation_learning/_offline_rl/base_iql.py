from dataclasses import dataclass, field
from sim_pipeline.configs.constants import ImitationAlgorithm
from sim_pipeline.configs.training._imitation_learning.base_imitation import ImitationTrainingConfig

@dataclass
class IQLTrainingConfig(ImitationTrainingConfig):
    algo: ImitationAlgorithm = ImitationAlgorithm.IQL
    
    layer_dims: list[int] = field(default_factory=lambda: [
        300, 
        400
    ])
    
    vf_quantile: float = 0.9
    target_tau: float = 0.01
    beta: float = 1.0
    learning_rate: float = 0.0001
    discount_rate: float = 0.99
    
    net_arch: str = 'gmm'
    gmm_heads: int = 8