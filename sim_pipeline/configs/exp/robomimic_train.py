from dataclasses import dataclass
from sim_pipeline.configs.exp.base import BaseExpConfig

@dataclass
class RobomimicTrainExpConfig(BaseExpConfig):
    # eval every n epochs
    eval_interval: int = 250
