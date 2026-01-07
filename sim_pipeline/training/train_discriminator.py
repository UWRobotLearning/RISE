import os

if 'DISPLAY' not in os.environ:
    import glfw
    glfw.ERROR_REPORTING = False

import hydra
import json
import torch
import numpy as np
import robomimic.macros as Macros
import time
import psutil
import h5py

import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils

from pathlib import Path

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from robomimic.config import config_factory
from robomimic.scripts.split_train_val import split_train_val_from_hdf5
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings

from sim_pipeline.configs.constants import PolicyType
from sim_pipeline.configs._imitation_learning.robomimic_imitation_base import RobomimicImitationBaseConfig
from sim_pipeline.configs.training._imitation_learning.base_imitation import ImitationTrainingConfig
from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import RobomimicReplayLowdimDataset

# from sim_pipeline.configs.constants import PolicyType
from sim_pipeline.data_manager.get_dataset import get_dataset
from sim_pipeline.utils.experiment import setup_experiment
from sim_pipeline.utils.train_utils import get_latest_robomimic_policy
from sim_pipeline.model_manager.create_model_metadata import create_model_metadata, update_metadata_trained

from sim_pipeline.ilid.discriminator import Discriminator


def robomimic_train(config, device, log_dir, ckpt_dir):
    """
    Train a model using the algorithm.
    """

    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    torch.set_num_threads(2)

    print("\n============= New Training Run with Config =============")
    print("")

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # make sure the dataset exists
    dataset_path = os.path.expanduser(config.train.data)
    if not os.path.exists(dataset_path):
        raise Exception("Dataset at provided path {} not found!".format(dataset_path))

    # load basic metadata from training file
    print("\n============= Loaded Environment Metadata =============")
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data,
        all_obs_keys=config.all_obs_keys,
        verbose=True
    )
    # setup for a new training run
    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )
    
    model = Discriminator(
        config=config,
        obs_shapes=shape_meta["all_shapes"],
        device=device,
    )

    # save the config as a json file
    with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    # load training data
    # train_sampler = trainset.get_dataset_sampler()
    # print("\n============= Training Dataset =============")
    # print(trainset)
    # print("")
    # if validset is not None:
    #     print("\n============= Validation Dataset =============")
    #     print(validset)
    #     print("")
    

    if config.observation.modalities.obs.rgb:
        with h5py.File(dataset_path, 'r') as f:
            data = f['data']
            ep_keys = list(data.keys())
            ep = data[ep_keys[0]]
            action_shape = ep['actions'].shape[1:]
            obs_shapes = {}
            for key in config.observation.modalities.obs.low_dim:
                if key not in ep['obs'].keys():
                    raise ValueError(f"Key {key} not found in dataset")
                obs_shapes[key] = {'shape': ep['obs'][key].shape[1:]}
            for key in config.observation.modalities.obs.rgb:
                if key not in ep['obs'].keys():
                    raise ValueError(f"Key {key} not found in dataset")
                # switch to channels first
                obs_shapes[key] = ep['obs'][key].shape[1:]
                obs_shapes[key] = {
                    'shape': (obs_shapes[key][2], obs_shapes[key][0], obs_shapes[key][1]),
                    'type': 'rgb',
                }
                
            shape_meta_dataloader = {
                'obs': obs_shapes,
                'action': {
                    'shape': action_shape,
                }
            }
        dataset = RobomimicReplayImageDataset(
            shape_meta=shape_meta_dataloader,
            dataset_path=dataset_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=1,
            abs_action=False,
            use_cache=True,
            seed=config.train.seed,
            val_ratio=0.0,
            extra_info=True,
            sample_goals=False,
            squeeze=True,
        )
    else:
        dataset, validset = TrainUtils.load_data_for_training(
            config, obs_keys=shape_meta["all_obs_keys"])


    obs_normalization_stats = None

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_data_workers,
        generator=torch.Generator(device=device),
        drop_last=True
    )
    # train_loader = DataLoader(
    #     dataset=trainset,
    #     sampler=train_sampler,
    #     batch_size=config.train.batch_size,
    #     shuffle=(train_sampler is None),
    #     num_workers=config.train.num_data_workers,
    #     generator=torch.Generator(device=device),
    #     drop_last=True
    # )

    config.experiment.validate = False
    # if config.experiment.validate:
    #     # cap num workers for validation dataset at 1
    #     num_workers = min(config.train.num_data_workers, 1)
    #     valid_sampler = validset.get_dataset_sampler()
    #     valid_loader = DataLoader(
    #         dataset=validset,
    #         sampler=valid_sampler,
    #         batch_size=config.train.batch_size,
    #         shuffle=(valid_sampler is None),
    #         num_workers=num_workers,
    #         generator=torch.Generator(device=device),
    #         drop_last=True
    #     )
    # else:
    valid_loader = None

    # print all warnings before training begins
    print("*" * 50)
    print("Warnings generated by robomimic have been duplicated here (from above) for convenience. Please check them carefully.")
    flush_warnings()
    print("*" * 50)
    print("")

    # main training loop
    best_valid_loss = None
    last_ckpt_time = time.time()

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    for epoch in range(1, config.train.num_epochs + 1): # epoch numbers start at 1
        step_log = TrainUtils.run_epoch(
            model=model,
            data_loader=train_loader,
            epoch=epoch,
            num_steps=train_num_steps,
            obs_normalization_stats=obs_normalization_stats,
        )
        model.on_epoch_end(epoch)

        # setup checkpoint path
        epoch_ckpt_name = "model_epoch_{}".format(epoch)

        # check for recurring checkpoint saving conditions
        should_save_ckpt = False
        if config.experiment.save.enabled:
            time_check = (config.experiment.save.every_n_seconds is not None) and \
                (time.time() - last_ckpt_time > config.experiment.save.every_n_seconds)
            epoch_check = (config.experiment.save.every_n_epochs is not None) and \
                (epoch > 0) and (epoch % config.experiment.save.every_n_epochs == 0)
            epoch_list_check = (epoch in config.experiment.save.epochs)
            should_save_ckpt = (time_check or epoch_check or epoch_list_check)
        ckpt_reason = None
        if should_save_ckpt:
            last_ckpt_time = time.time()
            ckpt_reason = "time"
            
        print("Train Epoch {}".format(epoch))
        should_save_ckpt = epoch % 2 == 0
        print(f"Should save ckpt: {should_save_ckpt}, enabled: {config.experiment.save.enabled}, freq: {config.experiment.save.every_n_epochs}")
        print(json.dumps(step_log, sort_keys=True, indent=4))
        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record("Timing_Stats/Train_{}".format(k[5:]), v, epoch)
            else:
                data_logger.record("Train/{}".format(k), v, epoch)

        # Evaluate the model on validation set
        if config.experiment.validate:
            with torch.no_grad():
                step_log = TrainUtils.run_epoch(model=model, data_loader=valid_loader, epoch=epoch, validate=True, num_steps=valid_num_steps)
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record("Timing_Stats/Valid_{}".format(k[5:]), v, epoch)
                else:
                    data_logger.record("Valid/{}".format(k), v, epoch)

            print("Validation Epoch {}".format(epoch))
            print(json.dumps(step_log, sort_keys=True, indent=4))

            # save checkpoint if achieve new best validation loss
            valid_check = "Loss" in step_log
            if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                best_valid_loss = step_log["Loss"]
                if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                    epoch_ckpt_name += "_best_validation_{}".format(best_valid_loss)
                    should_save_ckpt = True
                    ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

        # Save model checkpoints based on conditions (success rate, validation loss, etc)
        if should_save_ckpt:
            print('saving!!!!')
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
            )


        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print("\nEpoch {} Memory Usage: {} MB\n".format(epoch, mem_usage))    

    # terminate logging
    data_logger.close()
    

@hydra.main(version_base=None, config_path='../configs', config_name='robomimic_iql_base')
def train(config: RobomimicImitationBaseConfig):
    import traceback
    import sys
    try:
        run(config)
    except:
        traceback.print_exc(file=sys.stderr)
        raise
    
    
def get_robomimic_config(config: RobomimicImitationBaseConfig, dataset_path: str):
    train_cfg: ImitationTrainingConfig = config.training
    
    with h5py.File(dataset_path, 'r') as f:
        add_mask = 'mask' not in f
    
    if add_mask:        
        split_train_val_from_hdf5(dataset_path, val_ratio=0.1)
            
    robomimic_config = config_factory('bc')
    robomimic_config.observation.modalities.obs.rgb = OmegaConf.to_container(train_cfg.rgb_keys)
    robomimic_config.observation.modalities.obs.low_dim = OmegaConf.to_container(train_cfg.low_dim_keys)

    robomimic_config.train.hdf5_cache_mode = 'all'
    
    robomimic_config.train.data = dataset_path
    robomimic_config.train.output_dir = config.exp.logdir
    robomimic_config.train.batch_size = train_cfg.batch_size
    robomimic_config.train.num_epochs = train_cfg.num_epochs
    robomimic_config.experiment.name = config.name
    
    robomimic_config.experiment.logging.log_wandb = 'wandb' in config.exp.format_strings and not config.debug 
    robomimic_config.experiment.logging.wandb_proj_name = config.exp.project
    robomimic_config.experiment.logging.log_tb = 'tensorboard' in config.exp.format_strings
    
    Macros.WANDB_ENTITY = config.exp.entity
    
    return robomimic_config


def run(config: RobomimicImitationBaseConfig):
    device = setup_experiment(config.exp)
    
    dataset_path = get_dataset(config.data)

    robomimic_config = get_robomimic_config(config, dataset_path=dataset_path)
                
    log_dir = Path(config.exp.logdir) / config.name
    log_dir_rm, ckpt_dir_rm, video_dir_rm = TrainUtils.get_exp_dir(robomimic_config)
    dt_string = Path(log_dir_rm).parent.name
    create_model_metadata(
        name=config.name,
        policy_type=PolicyType.ROBOMIMIC,
        policy_identifier=dt_string,
        dataset_path=dataset_path,
        hydra_config=config,
        filepath=log_dir / dt_string / 'metadata.json',
    )
    try:
        robomimic_train(robomimic_config, device, log_dir_rm, ckpt_dir_rm)
    finally:
        max_epochs_saved: int = get_latest_robomimic_policy(log_dir / dt_string / 'models', find_epoch=True)
        if max_epochs_saved >= 1:
            update_metadata_trained(log_dir / dt_string / 'metadata.json')  