import os
if 'DISPLAY' not in os.environ:
    import glfw
    glfw.ERROR_REPORTING = False

import hydra
import torch
import numpy as np

from pathlib import Path
from omegaconf import OmegaConf
from omegaconf import MissingMandatoryValue

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from sim_pipeline.utils.logger import configure_logger
from sim_pipeline.utils.logger import Logger

from sim_pipeline.configs.default import TopLevelConfig
from sim_pipeline.configs.constants import PolicyType
from sim_pipeline.training.custom_callback import LoggerCallback
from sim_pipeline.training.init_buffer import init_buffer
from sim_pipeline.envs.make_env import make_vec_env
from sim_pipeline.utils.experiment import setup_wandb, setup_experiment
from sim_pipeline.utils.train_utils import get_latest_sb_policy
from sim_pipeline.data_manager.get_dataset import get_dataset
from sim_pipeline.model_manager.create_model_metadata import create_model_metadata, update_metadata_trained

@hydra.main(version_base=None, config_path='../configs', config_name="default")
def train(config: TopLevelConfig):
    import traceback
    import sys
    try:
        run(config)
    except:
        traceback.print_exc(file=sys.stderr)
        raise

def run(config: TopLevelConfig):
    device = setup_experiment(config.exp)
            
    log_path = config.exp.logdir
    logdir = Path(log_path) / f'{config.name}' / f'{config.exp.seed}'
    ckptdir = logdir / 'policy'
        
    if config.debug:
        # no wandb
        fs = list(config.exp.format_strings)
        config.exp.format_strings = [i for i in fs if i != "wandb"]
    
    if "wandb" in config.exp.format_strings:
        run = setup_wandb(
            config,
            name=f"{config.name}[sim][{config.exp.seed}]",
            entity=config.exp.entity,
            project=config.exp.project,
        )

    logger: Logger = configure_logger(str(logdir), config.exp.format_strings)

    envs: VecEnv = make_vec_env(
        env_config=config.env,
        num_workers=config.exp.num_workers,
        seed=config.exp.seed,
        device_id=config.exp.device_id,
        debug=config.debug
    )

    policy_kwargs = OmegaConf.to_container(config.training.policy_kwargs)
    algo = config.training.algo.get_algo()
    
    resuming = False
    if config.training.resume and os.path.exists(ckptdir):
        last_checkpoint = get_latest_sb_policy(ckptdir / '_step_')
        if last_checkpoint is not None:
            resuming = True
    
    if resuming:
        model: BaseAlgorithm = algo.load(last_checkpoint, env=envs, device=device)
    else:
        model: BaseAlgorithm = algo(
            policy=config.training.policy,
            env=envs,
            tensorboard_log=logdir,
            policy_kwargs=policy_kwargs,
            device=device
        )

    if config.training.init_buffer:
        dataset_path = get_dataset(config.data)
        init_buffer(model.replay_buffer, dataset_path, envs, config.env)
    
    eval_envs: VecEnv = make_vec_env(
        env_config=config.env,
        num_workers=config.exp.num_workers,
        seed=config.exp.seed * 777,
        device_id=config.exp.device_id,
        debug=config.debug
    )

    model.set_logger(logger)

    callback = LoggerCallback(
        eval_envs=eval_envs,
        eval_interval=config.exp.eval_interval,
        save_dir=ckptdir,
        save_interval=config.exp.save_interval,
    )
    
    create_model_metadata(
        name=config.name,
        policy_type=PolicyType.DIFFUSION,
        policy_identifier=config.exp.seed,
        dataset_path=dataset_path,
        hydra_config=config,
        filepath=logdir / 'metadata.json',
    )
    try:
        model.learn(
            total_timesteps=config.training.total_timesteps,
            callback=callback,
            progress_bar=not config.debug,
        )
    finally:
        max_epoch_saved: int = get_latest_sb_policy(ckptdir / '_step_', find_epoch=True)
        if max_epoch_saved >= 2000:
            update_metadata_trained(logdir / 'metadata.json')  

        envs.close()

    return model
