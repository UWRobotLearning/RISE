import h5py
import datetime
import getpass
from sim_pipeline.data_manager.dataset_metadata_enums import *

def fill_metadata(
    dataset_file: h5py.File,
    dataset_id: str | None = None,
    dataset_type: DatasetType | None = None,
    action_type: ActionType | None = None,
    is_real: bool | None = None,
    env_type: EnvType | None = None,
    env_name: str | None = None,
    reward_type: RewardType | None = None,
    include_date: bool = False,
    include_creator: bool = False,
):
    grp = dataset_file['data']

    if dataset_id is not None:    
        grp.attrs['dataset_id'] = dataset_id
    if dataset_type is not None:
        grp.attrs['dataset_type'] = dataset_type.value
    if action_type is not None:
        grp.attrs['action_type'] = action_type.value
    if is_real is not None:
        grp.attrs['is_real'] = is_real
    if env_type is not None:
        grp.attrs['env_type'] = env_type.value
    if env_name is not None:
        grp.attrs['env_name'] = env_name
    if reward_type is not None:
        grp.attrs['reward_type'] = reward_type.value
        
    if include_date:
        now = datetime.datetime.now()
        grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
        grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)

    if include_creator:
        grp.attrs["creator"] = getpass.getuser()