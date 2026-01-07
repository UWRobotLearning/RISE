from dataclasses import dataclass, field

from robomimic.algo.iql_diffusion import MultiStepMethod

from sim_pipeline.configs.constants import ImitationAlgorithm
from sim_pipeline.configs.training._imitation_learning.base_imitation import ImitationTrainingConfig

@dataclass
class IQLDiffusionTrainingConfig(ImitationTrainingConfig):
    algo: ImitationAlgorithm = ImitationAlgorithm.IDQL
    
    layer_dims: list[int] = field(default_factory=lambda: [
        300, 
        400,
        300
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
    
    use_bc: bool = False

    bottleneck_value: bool = False
    bottleneck_policy: bool = False
    
    lipschitz: bool = False
    lipschitz_slack: bool = False
    lipschitz_constant: float = 3.0
    lipschitz_weight: float = 0.005
    lipschitz_denoiser: bool = False

    q_bottleneck_beta: float = 0.2
    policy_bottleneck_beta: float = 0.2
    
    spectral_norm_value: bool = False
    spectral_norm_policy: bool = False
    
    late_fusion_key: str | None = 'robot0_gripper_qpos'
    late_fusion_layer_index: int = 1
    multiply_late_fusion_key: bool = False
    multiply_constant: int = 1
    
    use_dino_features: bool = False

    l2_policy: float = 0.0
    l2_value: float = 0.0
    
    action_augmentation: bool = False
    advanced_augmentation: bool = False
    augment_init_cutoff_threshold: int = 50
    augment_init_cutoff_thresh_expert: int = 20
    distance_threshold: float = 0.08
    num_neighbors: int = 10
    augment_play: bool = True
    mask_augmentation: bool = False
    proprio_keys: list[str] = field(default_factory=lambda: ['robot0_eef_pos', 'robot0_eef_quat'])
    
    pretrained: bool = False