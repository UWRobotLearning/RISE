import os
from pathlib import Path
from omegaconf import OmegaConf
from sim_pipeline.data_manager.data_manager import DataManager
from sim_pipeline.configs.constants import DATASET_DB_PATH
from sim_pipeline.configs.data.base_data import BaseDataConfig
from sim_pipeline.data_manager.dataset_metadata_enums import ObsType

def get_default_directories() -> list[Path]:
    if 'DATASET_PATHS' not in os.environ:
        print(
            "Please set the DATASET_PATHS environment variable to a list of directories to search for datasets.\n"
            "Delimit directories with commas.\n"
            "Example: export DATASET_PATHS='/path/to/dataset1,/path/to/dataset2'\n"
        )
        raise ValueError
    else:
        paths = os.environ['DATASET_PATHS'].split(',')
        return [Path(p) for p in paths]

    
def convert_data_cfg_to_query(data_cfg: BaseDataConfig) -> dict:
    data_queries = OmegaConf.to_container(data_cfg)
    dq = {}
    for key, value in data_queries.items():
        if key == 'name' and value != '':
            value = value.split('+')
        if key == 'obs_type' and value is not None:
            assert isinstance(value, ObsType)
            included_keys = value.included_keys()
            excluded_keys = value.excluded_keys()
            if included_keys:
                dq['obs_keys'] = included_keys
            if excluded_keys:
                dq['exclude_obs_keys'] = excluded_keys
        if not (value is None or value == ''):
            dq[key] = value
    return dq


def update_db_if_moved(dm: DataManager) -> None:
    if not dm.check_updated():
        print('Dataset paths moved. Re-parsing directories...')
        dm.create_tables()
        paths = get_default_directories()
        dm.parse_directories(paths)


def get_dataset(data_cfg: BaseDataConfig) -> str:
    if not DATASET_DB_PATH.exists():
        raise ValueError(f'Dataset manager not set up. Please run `python scripts/setup_data_manager.py`')
    with DataManager(DATASET_DB_PATH) as dm:
        update_db_if_moved(dm)
            
        if data_cfg.dataset_path:
            if Path(data_cfg.dataset_path).exists() and Path(data_cfg.dataset_path).is_file():
                return data_cfg.dataset_path
            else:
                raise ValueError(f'Provided explicit dataset path, but it is invalid: {data_cfg.dataset_path}')
            
        data_queries = convert_data_cfg_to_query(data_cfg)
        dataset_path = dm.get_dataset(attempt_combination=not data_cfg.do_not_combine, **data_queries)
        if dataset_path is None:
            print(
                f'No dataset found with the given query in cfg {data_cfg}\n'
                f'Either no datasets match the query or the query returned multiple\n'
                f'datasets that are incompatible with eachother.\n'
                f'If you think the query is correct,\n',
                f'please make sure your data manager is updated if you have\n'
                f'recently collected data. This can be done by running \n'
                f'`python scripts/setup_data_manager.py`.\n'
                f'Also, when doing this, please make sure your DATASET_PATHS environment variable\n'
                f'is set to the correct directories, deliminated by commas.\n'
                f"Example: export DATASET_PATHS='/path/to/dataset1,/path/to/dataset2'\n"
            )
            raise ValueError
        return dataset_path