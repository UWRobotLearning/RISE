from dataclasses import dataclass
from sim_pipeline.configs.env.franka_base import EnvConfigFranka
from sim_pipeline.configs.env.franka_cube import EnvConfigFrankaCube
import sim_pipeline.configs.default as default

@dataclass
class Config(default.TopLevelConfig):
    name: str = 'pick_cube'
    env: EnvConfigFrankaCube = EnvConfigFrankaCube()
