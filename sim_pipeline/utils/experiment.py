import wandb
import omegaconf
import random
import torch
import numpy as np

from omegaconf import MissingMandatoryValue
from sim_pipeline.configs.exp.base import BaseExpConfig

def setup_wandb(cfg, name, entity, project):

    run = wandb.init(
        name=name,
        entity=entity,
        project=project,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        settings=wandb.Settings(start_method="thread"),
    )

    return run

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
def set_gpu_mode(mode, gpu_id=0):
    global _GPU_ID
    global _USE_GPU
    global _DEVICE
    _GPU_ID = gpu_id
    _USE_GPU = mode
    _DEVICE = torch.device(("cuda:" + str(_GPU_ID)) if _USE_GPU else "cpu")
    torch.set_default_tensor_type(
        torch.cuda.FloatTensor if _USE_GPU else torch.FloatTensor
    )
    
def get_device() -> torch.device:
    global _DEVICE
    return _DEVICE


def setup_experiment(exp_config: BaseExpConfig, eval: bool = False) -> torch.device:
    try:
        seed = exp_config.seed
    except MissingMandatoryValue:
        exp_config['seed'] = np.random.randint(0, 1000000)
        seed = exp_config.seed
        
    if eval:
        try:
            seed = exp_config.eval_seed
        except MissingMandatoryValue:
            exp_config['eval_seed'] = np.random.randint(0, 1000000)
            seed = exp_config.eval_seed
        
    set_random_seed(seed)
    set_gpu_mode(torch.cuda.is_available(), exp_config.device_id)
    return get_device()