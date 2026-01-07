from dataclasses import dataclass, field

from robomimic.algo.iql_diffusion import MultiStepMethod

from sim_pipeline.configs.constants import ImitationAlgorithm
from sim_pipeline.configs.training._imitation_learning.base_imitation import ImitationTrainingConfig

@dataclass
class IQLDiffusionTrainingConfig(ImitationTrainingConfig):
    algo: ImitationAlgorithm = ImitationAlgorithm.IQL_DIFFUSION
    
    low_dim_keys: list[str] = field(default_factory=lambda: [
        "states"
    ])

    layer_dims: list[int] = field(default_factory=lambda: [
        300, 
        400
    ])
    
    vf_quantile: float = 0.9
    target_tau: float = 0.01
    beta: float = 1.0
    learning_rate: float = 0.0001
    discount_rate: float = 0.99

    down_dims: list[int] = field(default_factory=lambda: [
        256, 
        512, 
        1024
    ])
    diffusion_step_embed_dim: int = 256
    
    multi_step_method: MultiStepMethod = MultiStepMethod.ONE_STEP

    # don't update actor until this epoch
    actor_freeze_until_epoch: int | None = None
    
    observation_horizon: int = 2
    action_horizon: int = 8
    prediction_horizon: int = 16