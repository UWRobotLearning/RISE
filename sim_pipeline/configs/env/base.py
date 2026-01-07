from dataclasses import dataclass
from omegaconf import MISSING
from sim_pipeline.configs.constants import EnvType

@dataclass
class BaseEnvConfig:
    env_type: EnvType = MISSING


