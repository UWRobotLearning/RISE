import numpy as np
from sim_pipeline.configs.env.robomimic_base import RobomimicEnvConfig

def flatten_obs_dict(obs_dict: dict[str, np.ndarray], robot_prefix: str, env_config: RobomimicEnvConfig | None) -> np.ndarray:
    if env_config is not None and hasattr(env_config, 'obs_keys'):
        obs_keys: list[str] = env_config.obs_keys
        for i in range(len(obs_keys)):
            try:
                obs_dict[obs_keys[i]]
            except KeyError:
                # dataset will have 'object' key, but obs returned from env will have 'object-state'
                if obs_keys[i] == 'object':
                    obs_keys[i] = 'object-state'
                elif obs_keys[i] == 'object-state':
                    obs_keys[i] = 'object'
                else:
                    obs_keys[i] = f'{robot_prefix}{obs_keys[i]}'
                
        obs_list = [obs_dict[key] for key in obs_keys]
        obs = np.concatenate(obs_list, axis=-1)
    else:
        obs = np.concatenate([obs_dict[f'{robot_prefix}eef_pos'], obs_dict[f'{robot_prefix}eef_quat'], obs_dict['object-state']], axis=-1)
    return obs
