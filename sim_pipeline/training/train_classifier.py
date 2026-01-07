import hydra
import h5py
import json
from tqdm import tqdm
import numpy as np
import os

from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf
from sim_pipeline.configs._discriminator.robomimic_discriminator import RobomimicDiscriminatorConfig
from sim_pipeline.configs.training._imitation_learning.base_imitation import ImitationTrainingConfig
from sim_pipeline.configs.constants import PolicyType, ImitationAlgorithm
from sim_pipeline.utils.experiment import setup_experiment
from sim_pipeline.data_manager.get_dataset import get_dataset
from sim_pipeline.utils.shape_meta import get_shape_meta, update_diffusion_cfg_with_shape_meta
from sim_pipeline.utils.find_robomimic_json import find_json
from sim_pipeline.utils.diffusion_utils import get_policy_class, get_workspace
from sim_pipeline.utils.train_utils import get_latest_diffusion_policy, get_latest_robomimic_policy
from sim_pipeline.model_manager.create_model_metadata import create_model_metadata, update_metadata_trained

import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.log_utils import DataLogger
from robomimic.scripts.split_train_val import split_train_val_from_hdf5
from robomimic.config import config_factory
import robomimic.macros as Macros
from robomimic.algo.discriminator import Discriminator

import torch
from torch.utils.data import DataLoader


@hydra.main(version_base=None, config_path='../configs', config_name='robomimic_discriminator')
def train(config: RobomimicDiscriminatorConfig):
    import traceback
    import sys
    try:
        run(config)
    except:
        traceback.print_exc(file=sys.stderr)
        raise


def run(config: RobomimicDiscriminatorConfig):
    device = setup_experiment(config.exp)
    
    expert_dataset_path = get_dataset(config.data)
    suboptimal_dataset_path = get_dataset(config.suboptimal_data)

    # Generate robomimc config for the purpose of loading dataset
    expert_robomimic_config = get_robomimic_config(config, expert_dataset_path)
    suboptimal_robomimic_config = get_robomimic_config(config, suboptimal_dataset_path)

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=expert_dataset_path)

    # print(expert_robomimic_config)
    log_dir = Path(config.exp.logdir) / config.name
    log_dir_rm, ckpt_dir_rm, video_dir_rm = TrainUtils.get_exp_dir(expert_robomimic_config)
    dt_string = Path(log_dir_rm).parent.name
    create_model_metadata(
        name=config.name,
        policy_type=PolicyType.ROBOMIMIC,
        policy_identifier=dt_string,
        dataset_path=expert_dataset_path,
        hydra_config=config,
        filepath=log_dir / dt_string / 'metadata.json',
    )
    print(log_dir_rm)
    print(ckpt_dir_rm)
    print(video_dir_rm)

    # Load expert dataset
    ObsUtils.initialize_obs_utils_with_config(expert_robomimic_config)
    expert_shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=expert_robomimic_config.train.data,
        all_obs_keys=expert_robomimic_config.all_obs_keys,
        verbose=True
    )
    expert_trainset, expert_validset = TrainUtils.load_data_for_training(
        expert_robomimic_config, obs_keys=expert_shape_meta["all_obs_keys"])
    expert_train_sampler = expert_trainset.get_dataset_sampler()
    expert_train_loader = DataLoader(
        dataset=expert_trainset,
        sampler=expert_train_sampler,
        batch_size=expert_robomimic_config.train.batch_size,
        shuffle=(expert_train_sampler is None),
        num_workers=expert_robomimic_config.train.num_data_workers,
        generator=torch.Generator(device=device),
        drop_last=True
    )
    expert_obs_normalization_stats = None
    if expert_robomimic_config.train.hdf5_normalize_obs:
        expert_obs_normalization_stats = expert_trainset.get_obs_normalization_stats()
    
    if expert_robomimic_config.experiment.validate:
        num_workers = min(expert_robomimic_config.train.num_data_workers, 1)
        expert_valid_sampler = expert_validset.get_dataset_sampler()
        expert_valid_loader = DataLoader(
            dataset=expert_validset,
            sampler=expert_valid_sampler,
            batch_size=expert_robomimic_config.train.batch_size,
            shuffle=(expert_valid_sampler is None),
            num_workers=num_workers,
            generator=torch.Generator(device=device),
            drop_last=True
        )
    
    # Load suboptimal dataset
    ObsUtils.initialize_obs_utils_with_config(suboptimal_robomimic_config)
    suboptimal_shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=suboptimal_robomimic_config.train.data,
        all_obs_keys=suboptimal_robomimic_config.all_obs_keys,
        verbose=True
    )
    suboptimal_trainset, suboptimal_validset = TrainUtils.load_data_for_training(
        suboptimal_robomimic_config, obs_keys=suboptimal_shape_meta["all_obs_keys"])
    suboptimal_train_sampler = suboptimal_trainset.get_dataset_sampler()
    suboptimal_train_loader = DataLoader(
        dataset=suboptimal_trainset,
        sampler=suboptimal_train_sampler,
        batch_size=suboptimal_robomimic_config.train.batch_size,
        shuffle=(suboptimal_train_sampler is None),
        num_workers=suboptimal_robomimic_config.train.num_data_workers,
        generator=torch.Generator(device=device),
        drop_last=True
    )
    suboptimal_obs_normalization_stats = None
    if suboptimal_robomimic_config.train.hdf5_normalize_obs:
        suboptimal_obs_normalization_stats = suboptimal_trainset.get_obs_normalization_stats()
    
    if expert_robomimic_config.experiment.validate:
        num_workers = min(suboptimal_robomimic_config.train.num_data_workers, 1)
        suboptimal_valid_sampler = suboptimal_validset.get_dataset_sampler()
        suboptimal_valid_loader = DataLoader(
            dataset=suboptimal_validset,
            sampler=suboptimal_valid_sampler,
            batch_size=suboptimal_robomimic_config.train.batch_size,
            shuffle=(suboptimal_valid_sampler is None),
            num_workers=num_workers,
            generator=torch.Generator(device=device),
            drop_last=True
        )


    total_obs_dim = 0
    for shape in expert_shape_meta['all_shapes'].values():
        total_obs_dim += shape[0]

    # Setup data logger 
    data_logger = DataLogger(
        log_dir_rm,
        expert_robomimic_config,
        log_tb=expert_robomimic_config.experiment.logging.log_tb,
        log_wandb=expert_robomimic_config.experiment.logging.log_wandb,
    )

    # Instantiate the discriminator
    discriminator = Discriminator(
        algo_config=expert_robomimic_config.algo,
        obs_config=expert_robomimic_config.observation,
        global_config=expert_robomimic_config,
        obs_key_shapes=expert_shape_meta['all_shapes'],
        ac_dim=1,
        device=device
    )

    best_valid_loss = None

    # Train the discriminator
    for epoch in range(1, expert_robomimic_config.train.num_epochs + 1):
        step_log = run_discriminator_training_epoch(discriminator, expert_train_loader, suboptimal_train_loader, epoch=epoch, expert_obs_normalization_stats=expert_obs_normalization_stats, subopt_obs_normalization_stats=suboptimal_obs_normalization_stats)
        discriminator.on_epoch_end(epoch)

        # setup checkpoint path
        epoch_ckpt_name = "model_epoch_{}".format(epoch)

        should_save_ckpt = False
        if expert_robomimic_config.experiment.save.enabled:
            epoch_check = (expert_robomimic_config.experiment.save.every_n_epochs is not None) and \
                (epoch > 0) and (epoch % expert_robomimic_config.experiment.save.every_n_epochs == 0)
            epoch_list_check = (epoch in expert_robomimic_config.experiment.save.epochs)
            should_save_ckpt = (epoch_check or epoch_list_check)
        
        print("Train Epoch {}".format(epoch))
        print(json.dumps(step_log, sort_keys=True, indent=4))

        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record("Timing_Stats/Train_{}".format(k[5:]), v, epoch)
            else:
                data_logger.record("Train/{}".format(k), v, epoch)


        if expert_robomimic_config.experiment.validate:
            with torch.no_grad():
                val_step_log = run_discriminator_training_epoch(discriminator, expert_valid_loader, suboptimal_valid_loader, epoch=epoch, validate=True, expert_obs_normalization_stats=expert_obs_normalization_stats, subopt_obs_normalization_stats=suboptimal_obs_normalization_stats)
            print(json.dumps(val_step_log, sort_keys=True, indent=4))
                # discriminator.on_epoch_end(epoch, validate=True)
            
            for k, v in val_step_log.items():
                if k.startswith("Time_"):
                    data_logger.record("Timing_Stats/Valid_{}".format(k[5:]), v, epoch)
                else:
                    data_logger.record("Valid/{}".format(k), v, epoch)
            
            valid_check = "Loss" in step_log
            if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                best_valid_loss = step_log["Loss"]
                if expert_robomimic_config.experiment.save.enabled and expert_robomimic_config.experiment.save.on_best_validation:
                    epoch_ckpt_name += "_best_validation_{}".format(best_valid_loss)
                    should_save_ckpt = True


        if should_save_ckpt:
            print(f'Saving model at epoch {epoch} to {os.path.join(ckpt_dir_rm, epoch_ckpt_name)}')
            TrainUtils.save_model(
                model=discriminator,
                config=expert_robomimic_config,
                env_meta=env_meta,
                shape_meta=expert_shape_meta,
                ckpt_path=os.path.join(ckpt_dir_rm, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=expert_obs_normalization_stats,
            )
    data_logger.close()

    max_epochs_saved: int = get_latest_robomimic_policy(log_dir / dt_string / 'models', find_epoch=True)
    if max_epochs_saved >= 100:
        update_metadata_trained(log_dir / dt_string / 'metadata.json')  

def run_discriminator_training_epoch(discriminator, expert_loader, suboptimal_loader, epoch, validate=False, expert_obs_normalization_stats=None, subopt_obs_normalization_stats=None):

    step_log_all = []

    num_steps = max(len(expert_loader), len(suboptimal_loader))
    for _ in tqdm(range(num_steps)):
        expert_batch = next(iter(expert_loader))
        suboptimal_batch = next(iter(suboptimal_loader))

        input_expert_batch = discriminator.process_batch_for_training(expert_batch)
        input_subopt_batch = discriminator.process_batch_for_training(suboptimal_batch)

        input_expert_batch = discriminator.postprocess_batch_for_training(input_expert_batch, obs_normalization_stats=expert_obs_normalization_stats)
        input_subopt_batch = discriminator.postprocess_batch_for_training(input_subopt_batch, obs_normalization_stats=subopt_obs_normalization_stats)

        info = discriminator.train_on_batch(input_expert_batch, input_subopt_batch, epoch=epoch, validate=validate)

        # print('='*50)
        # print(info)
        step_log = discriminator.log_info(info)
        step_log_all.append(step_log)
        # print(step_log)

        # print(info['losses'])
    step_log_dict = {}
    for i in range(len(step_log_all)):
        for k in step_log_all[i]:
            if k not in step_log_dict:
                step_log_dict[k] = []
            step_log_dict[k].append(step_log_all[i][k])
    step_log_all = dict((k, float(np.mean(v))) for k, v in step_log_dict.items())

    return step_log_all



def get_robomimic_config(config: RobomimicDiscriminatorConfig, dataset_path: str):
    train_cfg: ImitationTrainingConfig = config.training
    
    with h5py.File(dataset_path, 'r') as f:
        add_mask = 'mask' not in f
    
    if add_mask:        
        split_train_val_from_hdf5(dataset_path, val_ratio=0.1)
            
    use_image = bool(train_cfg.rgb_keys)
    json_path = find_json(train_cfg, config.env.env_name.get_simple_name(), use_image)
    
    Macros.WANDB_ENTITY = config.exp.entity
    
    if json_path is None:
        robomimic_config = config_factory(train_cfg.algo.value)
        print('No json path found')
    else:
        json_config = json.load(open(json_path, 'r'))
        print(f'Found json path : {json_path}')
        
        json_config['experiment']['name'] = config.name
        json_config['train']['data'] = dataset_path
        json_config['train']['output_dir'] = config.exp.logdir
        json_config['train']['batch_size'] = train_cfg.batch_size
        json_config['train']['num_epochs'] = train_cfg.num_epochs
        json_config['experiment']['epoch_every_n_steps'] = train_cfg.epoch_every_n_steps
        json_config['experiment']['validate'] = not train_cfg.algo.is_offline_rl()
        
        json_config['experiment']['logging']['log_wandb'] = 'wandb' in config.exp.format_strings
        json_config['experiment']['logging']['wandb_proj_name'] = config.exp.project
        json_config['experiment']['logging']['log_tb'] = 'tensorboard' in config.exp.format_strings
        
        json_config['experiment']['rollout']['enabled'] = not train_cfg.algo.is_offline_rl()
        json_config['experiment']['rollout']['horizon'] = train_cfg.rollout_horizon
        json_config['experiment']['rollout']['rate'] = config.exp.eval_interval
        json_config['experiment']['render_video'] = train_cfg.render_video
        
        json_config['observation']['modalities']['obs']['low_dim'] = OmegaConf.to_container(train_cfg.low_dim_keys)
        json_config['observation']['modalities']['obs']['rgb'] = OmegaConf.to_container(train_cfg.rgb_keys)
        json_config['observation']['modalities']['obs']['depth'] = OmegaConf.to_container(train_cfg.depth_keys)
        
        json_config['train']['hdf5_cache_mode'] = train_cfg.cache_mode
        json_config['train']['hdf5_filter_key'] = 'train'
        json_config['train']['hdf5_validation_filter_key'] = 'valid'
        
        if train_cfg.algo.is_offline_rl():
            json_config['algo']['optim_params']['critic']['learning_rate']['initial'] = train_cfg.learning_rate
            json_config['algo']['optim_params']['vf']['learning_rate']['initial'] = train_cfg.learning_rate
            json_config['algo']['target_tau'] = train_cfg.target_tau
            json_config['algo']['vf_quantile'] = train_cfg.vf_quantile
            json_config['algo']['critic']['layer_dims'] = OmegaConf.to_container(train_cfg.layer_dims)
            json_config['algo']['adv']['beta'] = train_cfg.beta
            json_config['algo']['discount'] = train_cfg.discount_rate
            
            if train_cfg.algo != ImitationAlgorithm.IQL_DIFFUSION:
                json_config['algo']['actor']['layer_dims'] = OmegaConf.to_container(train_cfg.layer_dims)
                json_config['algo']['actor']['net']['type'] = train_cfg.net_arch
                json_config['algo']['actor']['net']['gmm']['num_modes'] = train_cfg.gmm_heads
                json_config['algo']['optim_params']['actor']['learning_rate']['initial'] = train_cfg.learning_rate
            else:
                json_config['algo']['multi_step_method'] = train_cfg.multi_step_method.value
                json_config['algo']['optim_params']['policy']['learning_rate']['initial'] = train_cfg.learning_rate
                json_config['algo']['horizon']['observation_horizon'] = train_cfg.observation_horizon
                json_config['algo']['horizon']['action_horizon'] = train_cfg.action_horizon
                json_config['algo']['horizon']['prediction_horizon'] = train_cfg.prediction_horizon

        robomimic_config = config_factory(json_config['algo_name'])
        with robomimic_config.values_unlocked():
            robomimic_config.update(json_config)
    
    return robomimic_config