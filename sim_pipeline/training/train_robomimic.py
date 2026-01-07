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
import sys
import h5py

import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils

from pathlib import Path
from collections import OrderedDict

from omegaconf import OmegaConf
from omegaconf import MissingMandatoryValue
from torch.utils.data import DataLoader

from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.scripts.split_train_val import split_train_val_from_hdf5
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings

from sim_pipeline.configs.constants import PolicyType, ImitationAlgorithm
from sim_pipeline.configs._imitation_learning.robomimic_imitation_base import RobomimicImitationBaseConfig
from sim_pipeline.configs.training._imitation_learning.base_imitation import ImitationTrainingConfig
from sim_pipeline.envs.robosuite_wrapper import RobomimicObsWrapper
from sim_pipeline.envs.make_env import make_env
# from sim_pipeline.configs.constants import PolicyType
from sim_pipeline.data_manager.get_dataset import get_dataset
from sim_pipeline.utils.find_robomimic_json import find_json
from sim_pipeline.utils.experiment import setup_experiment
from sim_pipeline.utils.train_utils import get_latest_robomimic_policy
from sim_pipeline.model_manager.create_model_metadata import create_model_metadata, update_metadata_trained
# from sim_pipeline.eval.rollout import rollout, PossibleVecEnv, RolloutStats
# from sim_pipeline.eval.rollout import RolloutPolicy as SimPipelineRolloutPolicy

# from robomimic.scripts.train import train as robomimic_train


def robomimic_train(config, env_config, exp_config, device, log_dir, ckpt_dir, video_dir):
    """
    Train a model using the algorithm.
    """

    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    torch.set_num_threads(2)

    print("\n============= New Training Run with Config =============")
    print("")

    # if config.experiment.logging.terminal_output_to_txt:
    #     # log stdout and stderr to a text file
    #     logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
    #     sys.stdout = logger
    #     sys.stderr = logger

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

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    # create environment
    envs = OrderedDict()
    config.experiment.rollout.enabled = False
    if config.experiment.rollout.enabled:
        # create environments for validation runs
        env_names = [env_meta["env_name"]]

        if config.experiment.additional_envs is not None:
            for name in config.experiment.additional_envs:
                env_names.append(name)

        for env_name in env_names:
            # env = EnvUtils.create_env_from_metadata(
            #     env_meta=env_meta,
            #     env_name=env_name, 
            #     render=False, 
            #     render_offscreen=config.experiment.render_video,
            #     use_image_obs=shape_meta["use_images"],
            #     use_depth_obs=shape_meta["use_depths"],
            # )
            env: RobomimicObsWrapper = make_env(
                env_config=env_config,
                seed=exp_config.seed,
                device_id=exp_config.device_id,
                flatten_obs=False,
                gymnasium_api=False,
            )        
            
            env = EnvUtils.wrap_env_from_config(env, config=config) # apply environment warpper, if applicable
            envs[env.name] = env
            print(envs[env.name])

    print("")

    # setup for a new training run
    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )

    # save the config as a json file
    with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    # load training data
    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"])
    train_sampler = trainset.get_dataset_sampler(
        data_source=trainset,
        generator=torch.Generator(device=device),
    )
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")
    if validset is not None:
        print("\n============= Validation Dataset =============")
        print(validset)
        print("")

    # maybe retreve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        generator=torch.Generator(device=device),
        drop_last=(not config.train.augment_nearby_states),
    )

    config.experiment.validate = False
    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler(
            data_source=validset,
            generator=torch.Generator(device=device),
        )
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            generator=torch.Generator(device=device),
            drop_last=True
        )
    else:
        valid_loader = None

    # print all warnings before training begins
    print("*" * 50)
    print("Warnings generated by robomimic have been duplicated here (from above) for convenience. Please check them carefully.")
    flush_warnings()
    print("*" * 50)
    print("")

    # main training loop
    best_valid_loss = None
    best_return = {k: -np.inf for k in envs} if config.experiment.rollout.enabled else None
    best_success_rate = {k: -1. for k in envs} if config.experiment.rollout.enabled else None
    last_ckpt_time = time.time()

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    for epoch in range(1, config.train.num_epochs + 1): # epoch numbers start at 1
        if config.train.augment_nearby_states:
            if epoch % 2 == 0:
                train_loader.sampler.set_augmented()
            else:
                train_loader.sampler.set_regular()
        
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
        should_save_ckpt = epoch % config.experiment.save.every_n_epochs == 0
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

        # Evaluate the model by by running rollouts

        # do rollouts at fixed rate or if it's time to save a new ckpt
        video_paths = None
        rollout_check = (epoch % config.experiment.rollout.rate == 0) or (should_save_ckpt and ckpt_reason == "time")
        rollout_check = False
        if config.experiment.rollout.enabled and (epoch > config.experiment.rollout.warmstart) and rollout_check:

            # wrap model as a RolloutPolicy to prepare for rollouts
            rollout_model = RolloutPolicy(model, obs_normalization_stats=obs_normalization_stats)
            # rollout_model = SimPipelineRolloutPolicy(rollout_model, policy_type=PolicyType.ROBOMIMIC)
            
            # rollout_stats = RolloutStats()
            # # Custom rollout ignores possibility of multiple envs (not sure why we would ever need to roll out
            # # a single policy on multiple distinct envs)
            # env = next(iter(envs.values()))
            # _env_name = next(iter(envs.keys()))
            # env = PossibleVecEnv(envs)
            
            # if config.experiment.render_video:
            #     size = (512, 512)
            #     fps = 20
            #     video_path = os.path.join(video_dir, f'{_env_name}_epoch_{epoch}.mp4')
            #     video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
            # else:
            #     video_writer = None


            num_episodes = config.experiment.rollout.n
            # for ep in range(num_episodes):
            #     rollout(
            #         policy=rollout_model,
            #         eval_envs=env,
            #         rollout_stats=rollout_stats,
            #         logger=None,
            #         steps=config.experiment.rollout.horizon,
            #         render_mode='rgb_array',
            #         log_video=config.experiment.render_video,
            #         video_writer=None,
            #         video_skip=config.experiment.get("video_skip", 5),
            #         render_dims=None,
            #         return_trajectory=False,
            #     )
            # rollout_stats.compute_statistics()
            
            all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
                policy=rollout_model,
                envs=envs,
                horizon=config.experiment.rollout.horizon,
                use_goals=config.use_goals,
                num_episodes=num_episodes,
                render=False,
                video_dir=video_dir if config.experiment.render_video else None,
                epoch=epoch,
                video_skip=config.experiment.get("video_skip", 5),
                terminate_on_success=config.experiment.rollout.terminate_on_success,
            )

            # summarize results from rollouts to tensorboard and terminal
            for env_name in all_rollout_logs:
                rollout_logs = all_rollout_logs[env_name]
                for k, v in rollout_logs.items():
                    if k.startswith("Time_"):
                        data_logger.record("Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]), v, epoch)
                    else:
                        data_logger.record("Rollout/{}/{}".format(k, env_name), v, epoch, log_stats=True)

                print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, rollout_logs["time"]))
                print('Env: {}'.format(env_name))
                print(json.dumps(rollout_logs, sort_keys=True, indent=4))

            # checkpoint and video saving logic
            updated_stats = TrainUtils.should_save_from_rollout_logs(
                all_rollout_logs=all_rollout_logs,
                best_return=best_return,
                best_success_rate=best_success_rate,
                epoch_ckpt_name=epoch_ckpt_name,
                save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
                save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
            )
            best_return = updated_stats["best_return"]
            best_success_rate = updated_stats["best_success_rate"]
            epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
            should_save_ckpt = (config.experiment.save.enabled and updated_stats["should_save_ckpt"]) or should_save_ckpt
            if updated_stats["ckpt_reason"] is not None:
                ckpt_reason = updated_stats["ckpt_reason"]

        # Only keep saved videos if the ckpt should be saved (but not because of validation score)
        should_save_video = (should_save_ckpt and (ckpt_reason != "valid")) or config.experiment.keep_all_videos
        if video_paths is not None and not should_save_video:
            for env_name in video_paths:
                os.remove(video_paths[env_name])

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
            
    use_image = bool(train_cfg.rgb_keys)
    json_path = find_json(train_cfg, config.env.env_name.get_simple_name(), use_image)
    
    Macros.WANDB_ENTITY = config.exp.entity
    
    if json_path is None:
        robomimic_config = config_factory(train_cfg.algo.value)
    else:
        json_config = json.load(open(json_path, 'r'))
        
        json_config['experiment']['name'] = config.name
        json_config['train']['data'] = dataset_path
        json_config['train']['output_dir'] = config.exp.logdir
        json_config['train']['batch_size'] = train_cfg.batch_size
        json_config['train']['num_epochs'] = train_cfg.num_epochs
        json_config['train']['pad_seq_length'] = train_cfg.pad_seq_length
        if train_cfg.lipschitz and not train_cfg.lipschitz_slack:
            json_config['train']['dataset_keys'] = json_config['train']['dataset_keys'] + ['entropy']
        json_config['experiment']['epoch_every_n_steps'] = train_cfg.epoch_every_n_steps
        json_config['experiment']['validate'] = not train_cfg.algo.is_offline_rl()
        
        json_config['experiment']['logging']['log_wandb'] = 'wandb' in config.exp.format_strings if not config.debug else False
        json_config['experiment']['logging']['wandb_proj_name'] = config.exp.project
        json_config['experiment']['logging']['log_tb'] = 'tensorboard' in config.exp.format_strings
        
        json_config['experiment']['rollout']['enabled'] = not train_cfg.algo.is_offline_rl()
        json_config['experiment']['rollout']['horizon'] = train_cfg.rollout_horizon
        json_config['experiment']['rollout']['rate'] = config.exp.eval_interval
        json_config['experiment']['render_video'] = train_cfg.render_video
        json_config['experiment']['save']['every_n_epochs'] = config.exp.save_interval
        
        json_config['observation']['modalities']['obs']['low_dim'] = OmegaConf.to_container(train_cfg.low_dim_keys)
        json_config['observation']['modalities']['obs']['rgb'] = OmegaConf.to_container(train_cfg.rgb_keys)
        json_config['observation']['modalities']['obs']['depth'] = OmegaConf.to_container(train_cfg.depth_keys)
        
        json_config['train']['hdf5_cache_mode'] = train_cfg.cache_mode
        json_config['train']['augment_nearby_states'] = train_cfg.action_augmentation
        json_config['train']['distance_threshold'] = train_cfg.distance_threshold
        json_config['train']['num_neighbors'] = train_cfg.num_neighbors
        json_config['algo']['action_augmentation'] = train_cfg.action_augmentation
        json_config['train']['advanced_augmentation'] = train_cfg.advanced_augmentation
        json_config['train']['augment_init_cutoff_thresh'] = train_cfg.augment_init_cutoff_threshold
        json_config['train']['augment_init_cutoff_thresh_expert'] = train_cfg.augment_init_cutoff_thresh_expert
        json_config['train']['augment_play'] = train_cfg.augment_play
        json_config['train']['gripper_key'] = train_cfg.late_fusion_key
        json_config['train']['mask_augmentation'] = train_cfg.mask_augmentation
        json_config['train']['proprio_keys'] = OmegaConf.to_container(train_cfg.low_dim_keys)
        
        if train_cfg.use_dino_features:
            json_config['observation']['encoder']['rgb']['core_class'] = 'DinoV2Core'
            json_config['observation']['encoder']['rgb']['core_kwargs'] = {
                'backbone_name': 'dinov2_vits14',
                'frozen': True,
            }
        elif train_cfg.pretrained:
            json_config['observation']['encoder']['rgb']['core_kwargs']['backbone_kwargs']['pretrained'] = True
            
        if train_cfg.algo.is_offline_rl():
            json_config['algo']['optim_params']['critic']['learning_rate']['initial'] = train_cfg.learning_rate
            json_config['algo']['optim_params']['vf']['learning_rate']['initial'] = train_cfg.learning_rate
            json_config['algo']['target_tau'] = train_cfg.target_tau
            json_config['algo']['vf_quantile'] = train_cfg.vf_quantile
            json_config['algo']['critic']['layer_dims'] = OmegaConf.to_container(train_cfg.layer_dims)
            json_config['algo']['critic']['late_fusion_key'] = train_cfg.late_fusion_key
            json_config['algo']['critic']['late_fusion_layer_index'] = train_cfg.late_fusion_layer_index
            
            if train_cfg.multiply_late_fusion_key and train_cfg.late_fusion_key is not None:
                json_config['algo']['critic']['obs_multiplier_key'] = train_cfg.late_fusion_key
                json_config['algo']['critic']['obs_multiplier'] = train_cfg.multiply_constant
            
            json_config['algo']['adv']['beta'] = train_cfg.beta
            json_config['algo']['discount'] = train_cfg.discount_rate
            
            json_config['algo']['optim_params']['critic']['regularization']['L2'] = train_cfg.l2_value
            json_config['algo']['optim_params']['vf']['regularization']['L2'] = train_cfg.l2_value
            json_config['algo']['optim_params']['policy']['regularization']['L2'] = train_cfg.l2_policy
            
            if not train_cfg.algo.is_diffusion():
                json_config['algo']['actor']['layer_dims'] = OmegaConf.to_container(train_cfg.layer_dims)
                json_config['algo']['actor']['net']['type'] = train_cfg.net_arch
                json_config['algo']['actor']['net']['gmm']['num_modes'] = train_cfg.gmm_heads
                json_config['algo']['optim_params']['actor']['learning_rate']['initial'] = train_cfg.learning_rate
            else:
                # iql diffusion
                json_config['algo']['multi_step_method'] = train_cfg.multi_step_method.value
                json_config['algo']['use_bc'] = train_cfg.use_bc
                if train_cfg.use_bc:
                    json_config['train']['hdf5_load_next_obs'] = False
                json_config['algo']['unet']['down_dims'] = OmegaConf.to_container(train_cfg.down_dims)
                json_config['algo']['unet']['diffusion_step_embed_dim'] = train_cfg.diffusion_step_embed_dim
                json_config['algo']['optim_params']['policy']['learning_rate']['initial'] = train_cfg.learning_rate
                json_config['algo']['optim_params']['policy']['freeze_until_epoch'] = train_cfg.actor_freeze_until_epoch
                json_config['algo']['horizon']['observation_horizon'] = train_cfg.observation_horizon
                json_config['algo']['horizon']['action_horizon'] = train_cfg.action_horizon
                json_config['algo']['horizon']['prediction_horizon'] = train_cfg.prediction_horizon
                json_config['train']['frame_stack'] = train_cfg.observation_horizon
                json_config['train']['seq_length'] = train_cfg.prediction_horizon

                #bottleneck config
                json_config['algo']['bottleneck_value'] = train_cfg.bottleneck_value
                json_config['algo']['bottleneck_policy'] = train_cfg.bottleneck_policy
                json_config['algo']['q_bottleneck_beta'] = train_cfg.q_bottleneck_beta
                json_config['algo']['policy_bottleneck_beta'] = train_cfg.policy_bottleneck_beta
                json_config['algo']['spectral_norm_value'] = train_cfg.spectral_norm_value
                json_config['algo']['spectral_norm_policy'] = train_cfg.spectral_norm_policy
                json_config['algo']['lipschitz'] = train_cfg.lipschitz
                json_config['algo']['lipschitz_constant'] = train_cfg.lipschitz_constant
                json_config['algo']['lipschitz_weight'] = train_cfg.lipschitz_weight
                json_config['algo']['lipschitz_slack'] = train_cfg.lipschitz_slack
                json_config['algo']['lipschitz_denoiser'] = train_cfg.lipschitz_denoiser

        robomimic_config = config_factory(json_config['algo_name'])
        with robomimic_config.values_unlocked():
            robomimic_config.update(json_config)
    
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
        robomimic_train(robomimic_config, config.env, config.exp, device, log_dir_rm, ckpt_dir_rm, video_dir_rm)
    finally:
        max_epochs_saved: int = get_latest_robomimic_policy(log_dir / dt_string / 'models', find_epoch=True)
        if max_epochs_saved >= 100:
            update_metadata_trained(log_dir / dt_string / 'metadata.json')  