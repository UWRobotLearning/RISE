from dataclasses import dataclass, field

from robomimic.algo.iql_diffusion import MultiStepMethod

from sim_pipeline.configs.constants import ImitationAlgorithm
from sim_pipeline.configs.training._imitation_learning._offline_rl.iql_diffusion_image import IQLDiffusionImageTrainingConfig

@dataclass
class IQLDiffusionTrainingConfig(IQLDiffusionImageTrainingConfig):
    low_dim_keys: list[str] = field(default_factory=lambda: [
        "states",
    ])

    rgb_keys: list[str] = field(default_factory=lambda: [
        "imgs"
    ])