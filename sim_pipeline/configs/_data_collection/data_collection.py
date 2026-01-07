from dataclasses import dataclass, field
from typing import Any
from sim_pipeline.configs.constants import IO_Devices, ROOT_DIR, DATA_DIR
from sim_pipeline.configs.env.robomimic_base import RobomimicEnvConfig
from sim_pipeline.configs.env.robomimic_square_ood import RobommimicSquareEnvOODConfig


@dataclass
class DataCollectionConfig:
    data_name: str = 'test'
    
    env: Any = RobomimicEnvConfig()
    device: IO_Devices = IO_Devices.SPACEMOUSE

    pos_sensitivity: float = 1.0
    rot_sensitivity: float = 1.0

    directory: str = str(DATA_DIR)

    print_rew: bool = False
    
    # if debug, don't save the data to avoid cluttering the
    # data directory with test runs
    debug: bool = False
    
    # for evaluating policies while controlling env manually
    model_path: str | None = None
    discriminator: bool = False
    eval_model: bool = False
    eval_value: bool = False
    
    n_obs_steps: int = 2
    
    lowdim_keys: list[str] | None = field(default_factory=lambda: [
        'robot0_eef_pos',
        'robot0_eef_quat',
        'robot0_gripper_qpos',
        'object-state',
    ])
    rgb_keys: list[str] | None = field(default_factory=lambda: [
        'agentview_image',
        'robot0_eye_in_hand_image'
    ])
    
    record_image_embedding: bool = False
    
    use_dino_features: bool = False