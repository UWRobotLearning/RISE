from dataclasses import dataclass, field

from robomimic.algo.iql_diffusion import MultiStepMethod

from sim_pipeline.configs.constants import ImitationAlgorithm
from sim_pipeline.configs.training._imitation_learning._offline_rl.iql_diffusion_image import IQLDiffusionImageTrainingConfig

@dataclass
class IQLDiffusionLipschitzTrainingConfig(IQLDiffusionImageTrainingConfig):
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
    
    diffusion_step_embed_dim: int = 256

    bottleneck_value: bool = False
    bottleneck_policy: bool = False
    
    spectral_norm_value: bool = False
    spectral_norm_policy: bool = False
    
    lipschitz: bool = True
    lipschitz_slack: bool = True
    lipschitz_constant: float = 3.0
    lipschitz_weight: float = 0.005
    lipschitz_denoiser: bool = True

    q_bottleneck_beta: float = 0.2
    policy_bottleneck_beta: float = 0.2