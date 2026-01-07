from dataclasses import dataclass, field

from robomimic.algo.iql_diffusion import MultiStepMethod

from sim_pipeline.configs.constants import ImitationAlgorithm
from sim_pipeline.configs.training._imitation_learning._offline_rl.iql_diffusion import IQLDiffusionTrainingConfig

@dataclass
class IQLDiffusionSpectralTrainingConfig(IQLDiffusionTrainingConfig):
    layer_dims: list[int] = field(default_factory=lambda: [
        300, 
        400,
        300
    ])
    
    down_dims: list[int] = field(default_factory=lambda: [
        256, 
        512, 
        1024
    ])
    diffusion_step_embed_dim: int = 256

    bottleneck_value: bool = False
    bottleneck_policy: bool = False
    
    spectral_norm_value: bool = False
    spectral_norm_policy: bool = True

    q_bottleneck_beta: float = 0.2
    policy_bottleneck_beta: float = 0.2