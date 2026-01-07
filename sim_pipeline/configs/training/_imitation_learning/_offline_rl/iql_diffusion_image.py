from dataclasses import dataclass, field

from robomimic.algo.iql_diffusion import MultiStepMethod

from sim_pipeline.configs.constants import ImitationAlgorithm
from sim_pipeline.configs.training._imitation_learning._offline_rl.iql_diffusion import IQLDiffusionTrainingConfig

@dataclass
class IQLDiffusionImageTrainingConfig(IQLDiffusionTrainingConfig):
    low_dim_keys: list[str] = field(default_factory=lambda: [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    ])

    rgb_keys: list[str] = field(default_factory=lambda: [
        "agentview_image",
        "robot0_eye_in_hand_image",
    ])
    
    layer_dims: list[int] = field(default_factory=lambda: [
        300, 
        400,
        300
    ])
    
    down_dims: list[int] = field(default_factory=lambda: [
        512, 
        1024, 
        2048
    ])
    diffusion_step_embed_dim: int = 128
    num_epochs: int = 5000
    batch_size: int = 64

    cache_mode: str | None = 'all'