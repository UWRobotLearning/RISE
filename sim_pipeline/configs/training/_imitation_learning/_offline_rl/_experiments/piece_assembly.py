from dataclasses import dataclass, field

from robomimic.algo.iql_diffusion import MultiStepMethod

from sim_pipeline.configs.constants import ImitationAlgorithm
from sim_pipeline.configs.training._imitation_learning._offline_rl.iql_diffusion_spectral_image import IQLDiffusionSpectralImageTrainingConfig

@dataclass
class TwoPieceConfig(IQLDiffusionSpectralImageTrainingConfig):
    low_dim_keys: list[str] = field(default_factory=lambda: [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    ])

    rgb_keys: list[str] = field(default_factory=lambda: [
        "agentview_image",
        "robot0_eye_in_hand_image",
    ])
            
    spectral_norm_policy: bool = True
    policy_bottleneck_beta: float = 0.1

    action_augmentation: bool = True
    advanced_augmentation: bool = True
    distance_threshold: float = 0.08
    num_neighbors: int = 15