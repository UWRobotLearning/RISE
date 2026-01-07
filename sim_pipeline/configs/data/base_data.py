from dataclasses import dataclass, field
from sim_pipeline.configs.constants import RobomimicEnvType
from sim_pipeline.data_manager.dataset_metadata_enums import *

@dataclass
class BaseDataConfig:
    # override dataset_dir if you want to specify the dataset path explicitly
    # instead of through the dataset manager
    dataset_path: str = ''
    
    # override these values with the desired query
    # empty string is equivalent to None
    name: str | None = ''
    env_name: str | None = None
    env_type: list[str] | None = None
    dataset_type: list[str] | None  = None
    action_type: str | None = None
    reward_type: list[str] | None = None
    obs_type: ObsType | None = None
    
    # if true, will not attempt to combine. if multiple datasets are
    # returned, just error
    do_not_combine: bool = False