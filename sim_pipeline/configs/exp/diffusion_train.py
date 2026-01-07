from dataclasses import dataclass
from sim_pipeline.configs.exp.base import BaseExpConfig

@dataclass
class DiffusionExpConfig(BaseExpConfig):
    # rollout every n epochs
    eval_interval: int = 500
    # checkpoint every n epochs
    save_interval: int = 500
