from dataclasses import dataclass, field
from sim_pipeline.configs.constants import RobomimicEnvType
from sim_pipeline.data_manager.dataset_metadata_enums import *
from sim_pipeline.configs.data.base_data import BaseDataConfig

@dataclass
class PushingDataConfig(BaseDataConfig):
    # override dataset_dir if you want to specify the dataset path explicitly
    # instead of through the dataset manager
    dataset_path: str = ''
    
    # override these values with the desired query
    name: str | None = 'push_pushing_reward'
    env_name: str | None = RobomimicEnvType.NUT_ASSEMBLY_SQUARE.value
    env_type: list[str] | None = field(default_factory=lambda: [EnvType.ROBOSUITE.value])
    dataset_type: list[str] | None  = field(default_factory=lambda: [DatasetType.PLAY.value])
    action_type: str = ActionType.RELATIVE.value
    reward_type: list[str] | None = field(default_factory=lambda: [RewardType.DENSE.value])
    obs_type: ObsType = ObsType.STATE