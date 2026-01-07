import hydra
import time
import json
import os
import torch

import robomimic.utils.file_utils as FileUtils
from robosuite import load_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from sim_pipeline.configs.constants import IO_Devices
from sim_pipeline.configs._data_collection.data_collection import DataCollectionConfig
from sim_pipeline.data_collection.teleop_utils import collect_human_trajectory, gather_demonstrations_as_hdf5
from sim_pipeline.envs.make_env import make_env
from sim_pipeline.envs.initialization_wrapper import InitializationWrapper
from sim_pipeline.data_manager.edit_dataset_description import edit_dataset_description

@hydra.main(version_base=None, config_path='../configs', config_name='data_collection')
def collect(cfg: DataCollectionConfig):
    env_config = cfg.env
    env_config.data_collection = True
    env_config.ignore_done = True
    env_config.render = True
    if cfg.discriminator and cfg.model_path is not None:
        env_config.render_offscreen = True
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=cfg.model_path, device=device, verbose=True, discriminator=True)
    else:
        env_config.render_offscreen = False
        policy = None
        
    if cfg.model_path is not None and not cfg.discriminator:
        env_config.render_offscreen = True
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=cfg.model_path, device=device, verbose=True, discriminator=False)
        if cfg.eval_value:
            value_function = policy.policy.nets['vf']
            q_function = policy.policy.nets['critic'][0]
        else:
            value_function = None
            q_function = None
    else:
        value_function = None
        q_function = None
        
    if not cfg.discriminator and not cfg.eval_model:
        eval_model = None
        
    if cfg.record_image_embedding:
        env_config.render_offscreen = True

    controller_config = load_controller_config(default_controller='OSC_POSE')
    # Create argument configuration
    config = {
        "env_name": env_config.env_name.value,
        "robots": ['Panda'],
        "controller_configs": controller_config,
    }

    # Create environment
    env = make_env(env_config)
    if isinstance(env, InitializationWrapper):
        env_name = env.env.__class__.__name__
    else:
        try:
            env_name = env.serialize()['env_name']
        except AttributeError:
            env_name = env.__class__.__name__

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    device = cfg.device.get_device()

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    data_name = cfg.data_name
    new_dir = os.path.join(cfg.directory, f"{data_name}_{t1}_{t2}")
    if not cfg.debug:
        os.makedirs(new_dir)
        
    if cfg.device == IO_Devices.KEYBOARD or cfg.device == IO_Devices.SPACEMOUSE:
        # set keyboard repeat delay to be faster for smoother keyboard teleop
        # os.system('xset r rate 50 33')
        pass
    try:
        # collect demonstrations
        ep = 1
        while True:
            continue_collection, success = collect_human_trajectory(
                env, 
                device, 
                'right', 
                'single-arm-opposed', 
                print_rew=cfg.print_rew, 
                eval_model=policy, 
                lowdim_keys=cfg.lowdim_keys, 
                rgb_keys=cfg.rgb_keys,
                value_function=value_function,
                q_function=q_function,
                n_obs_steps=cfg.n_obs_steps,
                record_image_embedding=cfg.record_image_embedding,
                use_dino_features=cfg.use_dino_features,
            )
            if success:
                print(f'Collected {ep} successful demonstrations\r', end='', flush=True)
                ep += 1
            if not cfg.debug:
                hdf5_path = gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info, env_name, data_name)  
            if not continue_collection:
                break
    finally:
        os.system('xset r rate 500 33')
        
    if not cfg.debug:
        edit_dataset_description(hdf5_path)