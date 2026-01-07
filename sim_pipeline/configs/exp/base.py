from dataclasses import dataclass
from typing import Tuple
from omegaconf import MISSING
from sim_pipeline.configs.constants import LOG_DIR
@dataclass
class BaseExpConfig:
    num_workers: int = 16
    seed: int = MISSING
    device_id: int = 0
    logdir: str = LOG_DIR
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 500
    format_strings: Tuple = ('stdout','tensorboard', 'wandb')
    entity: str = 'kehuang'
    project: str = 'iql_diffusion'