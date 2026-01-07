from dataclasses import dataclass, field

from robomimic.algo.iql_diffusion import MultiStepMethod

from sim_pipeline.configs.constants import ImitationAlgorithm
from sim_pipeline.configs.training._imitation_learning._offline_rl.iql_diffusion_image import IQLDiffusionImageTrainingConfig

@dataclass
class IQLDiffusionTrainingConfig(IQLDiffusionImageTrainingConfig):
    algo: ImitationAlgorithm = ImitationAlgorithm.IQL_DIFFUSION
    
    low_dim_keys: list[str] = field(default_factory=lambda: [
        "lowdim_ee",
    ])
    
    rgb_keys: list[str] = field(default_factory=lambda: [
        "agentview_image",
        "eye_in_hand_image",
    ])

    # don't update actor until this epoch
    actor_freeze_until_epoch: int | None = None
    
    observation_horizon: int = 2
    action_horizon: int = 8
    prediction_horizon: int = 16