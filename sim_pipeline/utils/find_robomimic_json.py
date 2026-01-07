import os
import robomimic
import h5py
import json

from pathlib import Path
from sim_pipeline.configs.training._imitation_learning.base_imitation import ImitationTrainingConfig
from sim_pipeline.configs.constants import ImitationAlgorithm, LOG_DIR

def file_if_exists(filepath: Path) -> str | None:
    return str(filepath) if filepath.exists() else None

def find_json(training_config: ImitationTrainingConfig, env: str, use_image: bool) -> str | None:
    algo: ImitationAlgorithm = training_config.algo
            
    robomimic_dir = Path(robomimic.__path__[0])
    template_dir = robomimic_dir / 'exps' / 'templates'
    paper_configs_dir = robomimic_dir / 'exps' / 'paper'
    
    if algo not in [ImitationAlgorithm.BC, ImitationAlgorithm.BC_RNN, ImitationAlgorithm.CQL]:
        return file_if_exists(Path(f'{template_dir}/{algo.value}.json'))

    if not paper_configs_dir.exists():
        generate_script = robomimic_dir / 'scripts' / 'generate_paper_configs.py'
        os.system(f'python {generate_script} --output_dir {LOG_DIR}')
        
    obs = 'image' if use_image else 'low_dim'
    json_dir = paper_configs_dir / 'core' / env / 'ph' / obs / f'{algo.value}.json'
    
    if not json_dir.exists():
        return file_if_exists(Path(f'{template_dir}/{algo.value}.json'))
    return json_dir
    
    # try:
    #     with open(json_dir, 'r') as f:
    #         config = json.load(f)
    #         obs_modalities = config['observation']['modalities']['obs']
    #         for config_obs_key in obs_modalities.values():
    #             for key in config_obs_key:
    #                 if key not in obs_keys:
    #                     violating_key = key
    #                     break
    #             else:
    #                 # config key is in data
    #                 continue
                
    #             # config key is not in data
    #             break
    #         else:
    #             # all config keys are in data
    #             return str(json_dir)
            
    #         # not all config keys are in data, invalid config
    #         print(f'found json config {json_dir}, but provided data is missing key {violating_key}')
    #         return None
        
    # except Exception as e:
    #     print(f'Error loading robomimic json config: {e}')
    #     return None