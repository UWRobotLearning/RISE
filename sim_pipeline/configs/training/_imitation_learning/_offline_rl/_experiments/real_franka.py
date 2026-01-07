from dataclasses import dataclass, field

from robomimic.algo.iql_diffusion import MultiStepMethod

from sim_pipeline.configs.constants import ImitationAlgorithm
from sim_pipeline.configs.training._imitation_learning._offline_rl.iql_diffusion import IQLDiffusionTrainingConfig

@dataclass
class IQLDiffusionImageTrainingConfig(IQLDiffusionTrainingConfig):
    low_dim_keys: list[str] = field(default_factory=lambda: [
        "states",
    ])

    rgb_keys: list[str] = field(default_factory=lambda: [
        "front_image",
        "wrist_image",
    ])
    
    diffusion_step_embed_dim: int = 128
    num_epochs: int = 5000
    batch_size: int = 64

    cache_mode: str | None = 'all'

    spectral_norm_policy: bool = True
    policy_bottleneck_beta: float = 0.1

    action_augmentation: bool = True
    advanced_augmentation: bool = True
    distance_threshold: float = 0.022
    num_neighbors: int = 15