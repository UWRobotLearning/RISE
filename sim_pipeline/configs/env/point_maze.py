from dataclasses import dataclass, field
from typing import Dict, Tuple
from sim_pipeline.configs.constants import EnvType

from sim_pipeline.configs.env.base import BaseEnvConfig

@dataclass
class EnvConfigPointMaze(BaseEnvConfig):
    env_type: EnvType = EnvType.POINT_MAZE

    maze_id: str = 'train_easy'
    render_mode: str = 'rgb_array'
    max_ep_steps: int = 1000
    reward_type: str = 'sparse'
