from dataclasses import dataclass, field

from robomimic.algo.iql_diffusion import MultiStepMethod

from sim_pipeline.configs.constants import ImitationAlgorithm
from sim_pipeline.configs.training._imitation_learning._offline_rl.iql_diffusion import IQLDiffusionTrainingConfig

@dataclass
class IQLDiffusionBottleneckTrainingConfig(IQLDiffusionTrainingConfig):
    layer_dims: list[int] = field(default_factory=lambda: [
        300, 
        400
    ])
    
    down_dims: list[int] = field(default_factory=lambda: [
        128, 
        256, 
        512
    ])
    diffusion_step_embed_dim: int = 128

    bottleneck_value: bool = False
    bottleneck_policy: bool = True

    q_bottleneck_beta: float = 0.2
    policy_bottleneck_beta: float = 0.2