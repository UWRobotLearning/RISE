import sys
import json
import uuid
import getpass
import h5py
import datetime
import git 

from omegaconf import OmegaConf, DictConfig

from pathlib import Path
from sim_pipeline.configs.constants import PolicyType
from sim_pipeline.utils.config_utils import convert_hydra_for_serialization

def create_model_metadata(
    name: str,
    policy_type: PolicyType,
    policy_identifier: str,
    dataset_path: str | None,
    hydra_config: DictConfig,
    filepath: str | Path,
    valid: bool = False,
    description: str = '',
):
    metadata = {}
    metadata['name'] = name
    metadata['policy_id'] = str(uuid.uuid4())
    metadata['policy_type'] = policy_type.value
    metadata['identifier'] = policy_identifier
    hydra_config = OmegaConf.to_container(hydra_config)
    convert_hydra_for_serialization(hydra_config)
    metadata['hydra_config'] = hydra_config
    
    if dataset_path is not None:
        with h5py.File(dataset_path, 'r') as f:
            metadata['dataset_name'] = Path(dataset_path).stem
            metadata['dataset_uuid'] = f['data'].attrs['dataset_id']
            if 'description' in f['data'].attrs:
                metadata['dataset_description'] = f['data'].attrs['description']
            else:
                metadata['dataset_description'] = ''
    
    metadata['creator'] = getpass.getuser()
    now = datetime.datetime.now()
    metadata["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    metadata["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    metadata['commit'] = sha

    args = sys.argv
    command = ' '.join(args)
    
    metadata['command'] = command
    
    # by default, a policy is not considered trained until after X number
    # of training steps and update_metadata_trained is called. Otherwise,
    # faulty/test runs would produce valid metadata
    metadata['valid'] = valid
    metadata['description'] = description
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)    
    
def update_metadata_trained(filepath: str | Path):
    with open(filepath, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
        metadata['valid'] = True
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)