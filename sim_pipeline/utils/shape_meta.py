import h5py
from omegaconf import DictConfig

def get_shape_meta(dataset_path: str, all_obs_keys: set[str]) -> tuple[dict[str, tuple[int]], tuple[int]]:
    with h5py.File(dataset_path, 'r') as f:
        data = f['data']
        for ep_key in data.keys():
            ep = data[ep_key]
            break
        
        obs = ep['obs']
        action_shape = tuple(ep['actions'].shape)[1:]
        
        obs_shape_meta = {}
        for key in all_obs_keys:
            obs_shape_meta[key] = tuple(obs[key].shape)[1:]
            
        return obs_shape_meta, action_shape
    

def update_diffusion_cfg_with_shape_meta(
    shape_meta_cfg: DictConfig, 
    obs_shape_meta: dict[str, tuple[int]],
    action_shape_meta: tuple[int],
    rgb_keys: list[str],
    lowdim_keys: list[str],
    depth_keys: list[str]
):
    shape_meta_cfg.action.shape = action_shape_meta
    
    def get_type(key: str) -> str:
        if key in rgb_keys:
            return 'rgb'
        if key in lowdim_keys:
            return 'low_dim'
        if key in depth_keys:
            return 'depth'
        raise ValueError(f"Unknown key {key}")
    
    del shape_meta_cfg.obs
    shape_meta_cfg.obs = {}
    
    shape_dict = {}
    for key, shape in obs_shape_meta.items():
        # format must be c, h, w
        if get_type(key) == 'rgb':
            shape = (shape[2], shape[0], shape[1])
        shape_dict[key] = {
            'shape': shape,
            'type': get_type(key)
        }
        
    shape_meta_cfg.obs = shape_dict
    