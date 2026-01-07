from dataclasses import dataclass, field
from sim_pipeline.configs.constants import EnvType
from sim_pipeline.configs.constants import RobomimicEnvType
from sim_pipeline.configs.env.base import BaseEnvConfig
from sim_pipeline.data_manager.dataset_metadata_enums import ObsType
from sim_pipeline.reward_functions import RewardFunction

@dataclass
class RobomimicEnvConfig(BaseEnvConfig):
    env_type: EnvType = EnvType.ROBOMIMIC
    env_name: RobomimicEnvType = RobomimicEnvType.NUT_ASSEMBLY_SQUARE

    # whether to render in window
    render: bool = True
    render_offscreen: bool = False

    ignore_done: bool = False
    # len of episode, no effect if ignore_done is True
    horizon: int = 200

    # use robosuite instead of robomimic wrapper when doing data collection
    data_collection: bool = False

    # obs keys that are included in flattened observations (only applicable to RL currently)
    obs_keys: list[str] = field(default_factory=lambda: [
        'eef_pos',
        'eef_quat',
        'gripper_qpos',
        'object-state',
    ])
    # names of cameras to render. only used if obs_type includes images/depth
    camera_names: list[str] = field(default_factory=lambda: [
        'agentview',
        'robot0_eye_in_hand'
    ])
    # determines whether or not env should return images as part of its obs dict
    obs_type: ObsType = ObsType.IMAGE
    # custom reward
    reward_function: RewardFunction | None = None

    ## Initialization Config

    custom_ee_init: bool = False
    # x, y, z, [min, max] (default robomimic values are listed here)
    init_ee_range: list[list[float]] = field(default_factory=lambda: [
        [-0.1031, -0.103], 
        [-0.0035, -0.0034],
        [1.0174, 1.0175]
    ])
    custom_obj_init: bool = False
    init_obj_range: dict[str, list[list[float]]] = field(default_factory=lambda: 
    {
        'SquareNut':
        [
            [0, 0.12],
            [-0.225, 0.0], 
            
        ],
    })