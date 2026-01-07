import numpy as np
import hydra
import torch
import cv2
import h5py
import uuid
import json
import dill

import robomimic.utils.file_utils as FileUtils
from omegaconf import open_dict, OmegaConf
from contextlib import nullcontext
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from hydra import compose

from gym.vector.vector_env import VectorEnv
from stable_baselines3.common.vec_env import VecEnv

import robomimic.utils.obs_utils as ObsUtils

from sim_pipeline.eval.rollout_env import PossibleVecEnv
from sim_pipeline.eval.rollout_policies import *
from sim_pipeline.eval.rollout import rollout
from sim_pipeline.configs._evaluation.eval_base import EvaluationConfig
from sim_pipeline.configs.constants import PolicyType, CombinedPolicyType, ImitationAlgorithm, ROLLOUT_DATA_DIR
from sim_pipeline.envs.robosuite_wrapper import RobomimicObsWrapper
from sim_pipeline.envs.make_env import make_env, make_vec_env
from sim_pipeline.eval.rollout_stats import RolloutStats
from sim_pipeline.utils.rollout_utils import add_trajectory_to_dataset, merge_obs_specs
from sim_pipeline.utils.train_utils import get_latest_sb_policy, get_latest_robomimic_policy, get_latest_diffusion_policy, get_latest_ogbench_policy
from sim_pipeline.utils.experiment import setup_experiment
from sim_pipeline.utils.video_writer import video_writer_manager
from sim_pipeline.utils.diffusion_utils import get_policy_class, get_workspace
from sim_pipeline.data_manager.modify_dataset import fill_metadata
from sim_pipeline.data_manager.dataset_metadata_enums import *
from sim_pipeline.data_manager.edit_dataset_description import edit_dataset_description
from sim_pipeline.model_manager.get_model import get_model
from sim_pipeline.data_manager.get_dataset import get_dataset

def get_env(policy_type: PolicyType, config: EvaluationConfig, obs_modality_specs=None, n_envs_override: int = None) -> RobomimicObsWrapper | VecEnv | VectorEnv:
    if n_envs_override is not None:
        n_envs = n_envs_override
    else:
        n_envs = config.exp.num_workers
    if policy_type == PolicyType.ROBOMIMIC:
        if n_envs > 1:
            env: VectorEnv = make_vec_env(
                env_config=config.env,
                num_workers=n_envs,
                seed=config.exp.eval_seed,
                device_id=config.exp.device_id,
                flatten_obs=False,
                use_gym_vec_env=True,
                gymnasium_api=False,
            )
        else:
            env: RobomimicObsWrapper = make_env(
                env_config=config.env,
                seed=config.exp.eval_seed,
                device_id=config.exp.device_id,
                flatten_obs=False,
            )
    elif policy_type == PolicyType.STABLE_BASELINES:
        if n_envs > 1:
            env: VecEnv = make_vec_env(
                env_config=config.env,
                num_workers=n_envs,
                seed=config.exp.eval_seed,
                device_id=config.exp.device_id,
                flatten_obs=True,
                use_gym_vec_env=False,
                mp_context='forkserver',
                obs_modality_specs=obs_modality_specs,
                gymnasium_api=False,
            )
        else:
            env: RobomimicObsWrapper = make_env(
                env_config=config.env,
                seed=config.exp.eval_seed,
                device_id=config.exp.device_id,
                flatten_obs=True,
            )
    elif policy_type == PolicyType.DIFFUSION or policy_type == PolicyType.COMPOSITE or policy_type == PolicyType.OGBENCH:
        if n_envs > 1:
            env: VectorEnv = make_vec_env(
                env_config=config.env,
                num_workers=n_envs,
                seed=config.exp.eval_seed,
                device_id=config.exp.device_id,
                flatten_obs=False,
                use_gym_vec_env=True,
                mp_context='forkserver',
                obs_modality_specs=obs_modality_specs,
                gymnasium_api=False,
            )
        else:
            env: RobomimicObsWrapper = make_env(
                env_config=config.env,
                seed=config.exp.eval_seed,
                device_id=config.exp.device_id,
                flatten_obs=False,
            )
    elif policy_type == PolicyType.PD:
        if n_envs > 1:
            env: VectorEnv = make_vec_env(
                env_config=config.env,
                num_workers=n_envs,
                seed=config.exp.eval_seed,
                device_id=config.exp.device_id,
                flatten_obs=False,
                use_gym_vec_env=True,
                mp_context='forkserver',
                obs_modality_specs=obs_modality_specs,
                gymnasium_api=False,
            )
        else:
            env: RobomimicObsWrapper = make_env(
                env_config=config.env,
                seed=config.exp.eval_seed,
                device_id=config.exp.device_id,
                flatten_obs=False,
            )
    else:
        raise NotImplementedError
    return env


def get_policy_dir(policy_type: PolicyType, config: EvaluationConfig) -> Path | None:
    if config.exp.use_model_manager and not policy_type in [PolicyType.PD, PolicyType.COMPOSITE, PolicyType.STABLE_BASELINES, PolicyType.OGBENCH]:
        identifier = config.exp.date_string
        if identifier:
            model_path = get_model(config.name, identifier)
        else:
            model_path = get_model(config.name)
        return Path(model_path)
    
    if config.exp.model_path:
        return Path(config.exp.model_path)
    
    log_path = config.exp.logdir
    match policy_type:
        case PolicyType.STABLE_BASELINES:
            logdir = Path(log_path) / f'{config.name}' / f'{config.exp.seed}'
        case PolicyType.ROBOMIMIC | PolicyType.DISCRIMINATOR:
            logdir: Path = Path(log_path) / f'{config.name}' / config.exp.date_string
        case PolicyType.DIFFUSION:
            logdir: Path = Path(log_path) / f'{config.name}' / config.exp.date_string
        case PolicyType.OGBENCH:
            logdir: Path = Path(log_path) / f'{config.name}'
        case PolicyType.PD:
            logdir = None
        case PolicyType.COMPOSITE:
            logdir: Path = Path(log_path) / config.name
            if not logdir.exists():
                logdir.mkdir()
        case _:
            raise NotImplementedError
    return logdir


def load_policy(policy_type: PolicyType, config: EvaluationConfig, policy_dir: Path, env, device: torch.device):
    match policy_type:
        case PolicyType.STABLE_BASELINES:
            algo = config.training.algo.get_algo()
            ckpt_path = get_latest_sb_policy(policy_dir / 'policy' / '_step_')
            model = algo.load(ckpt_path, device=device, env=env)
            
            return model
        case PolicyType.ROBOMIMIC | PolicyType.DISCRIMINATOR:
            ckpt_path = get_latest_robomimic_policy(policy_dir / 'models')
            
            policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True,
                                                                 discriminator=policy_type == PolicyType.DISCRIMINATOR)
            
            return policy
        case PolicyType.DIFFUSION:
            ckpt_path = get_latest_diffusion_policy(policy_dir / 'checkpoints')
            payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
            diffusion_cfg = payload['cfg']
            
            lowdim_keys = config.training.low_dim_keys
            rgb_keys = config.training.rgb_keys
            depth_keys = config.training.depth_keys
            
            # adapt old diffusion library configs
            with open_dict(diffusion_cfg):
                diffusion_cfg.debug = False
            
            workspace_cls = get_workspace(rgb_keys, lowdim_keys, depth_keys)
            workspace = workspace_cls(diffusion_cfg, output_dir=policy_dir)
            workspace.load_payload(payload)

            policy = workspace.model
            if diffusion_cfg.training.use_ema:
                policy = workspace.ema_model
            policy.to(device)
            policy.eval()
        case PolicyType.OGBENCH:
            from impls.utils.flax_utils import restore_agent
            from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
            from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import RobomimicReplayLowdimDataset
            from torch.utils.data import DataLoader

            dataset_path = get_dataset(config.data)
            
            if config.training.rgb_keys:
                with h5py.File(dataset_path, 'r') as f:
                    data = f['data']
                    ep_keys = list(data.keys())
                    ep = data[ep_keys[0]]
                    action_shape = ep['actions'].shape[1:]
                    obs_shapes = {}
                    for key in config.training.low_dim_keys:
                        if key not in ep['obs'].keys():
                            raise ValueError(f"Key {key} not found in dataset")
                        obs_shapes[key] = {'shape': ep['obs'][key].shape[1:]}
                    for key in config.training.rgb_keys:
                        if key not in ep['obs'].keys():
                            raise ValueError(f"Key {key} not found in dataset")
                        # switch to channels first
                        obs_shapes[key] = ep['obs'][key].shape[1:]
                        obs_shapes[key] = {
                            'shape': (obs_shapes[key][2], obs_shapes[key][0], obs_shapes[key][1]),
                            'type': 'rgb',
                        }
                        
                    shape_meta = {
                        'obs': obs_shapes,
                        'action': {
                            'shape': action_shape,
                        }
                    }

                dataset = RobomimicReplayImageDataset(
                    shape_meta=shape_meta,
                    dataset_path=dataset_path,
                    horizon=1,
                    pad_before=0,
                    pad_after=0,
                    n_obs_steps=1,
                    abs_action=False,
                    use_cache=True,
                    seed=config.exp.seed,
                    val_ratio=0.0,
                    extra_info=True,
                    sample_goals=True,
                    p_currgoal=config.training.p_currgoal,
                    squeeze=True,
                )
            else:
                dataset = RobomimicReplayLowdimDataset(
                    obs_keys=config.training.low_dim_keys,
                    dataset_path=dataset_path,
                    horizon=1,
                    pad_before=0,
                    pad_after=0,
                    abs_action=False,
                    seed=config.exp.seed,
                    val_ratio=0.0,
                    extra_info=True,
                    sample_goals=True,
                    p_currgoal=config.training.p_currgoal,
                    squeeze=True,
                )

            train_dataloader = DataLoader(
                dataset,
                batch_size=config.training.batch_size,
                num_workers=0,
                shuffle=True,
                generator=torch.Generator(device=device),
            )
                
            example_batch = next(iter(train_dataloader))
            example_batch_np = {
                'observations': example_batch['obs'].numpy(),
                'actions': example_batch['action'].numpy(),
            }
            agent_class = config.training.algo.get_agent()
            agent = agent_class.create(
                seed=config.exp.seed,
                ex_observations=example_batch_np['observations'],
                ex_actions=example_batch_np['actions'],
                config=OmegaConf.to_container(config.training),
            )
            latest_epoch = get_latest_ogbench_policy(policy_dir / 'checkpoints', find_epoch=True)
            policy = restore_agent(agent, str(policy_dir / 'checkpoints'), latest_epoch)
        case PolicyType.PD:
            policy = None
        case _:
            raise NotImplementedError
    return policy
    
    
def create_combined_policy(
    policies,
    combined_policy_type: CombinedPolicyType,
    n_envs: int,
):
    match combined_policy_type:
        case CombinedPolicyType.HEURISTIC_RESET:
            policy = HeuristicResetPolicy(
                policies=policies,
                n_envs=n_envs
            )
        case CombinedPolicyType.HEURISTIC_RESET_SPLIT:
            policy = HeuristicResetSplitPolicy(
                policies=policies,
                n_envs=n_envs
            )
        case _:
            raise NotImplementedError
    return policy

    
def create_rollout_policy(
        policy, 
        policy_type: PolicyType,
        config: EvaluationConfig,
        n_obs_steps: int | None = None,
        action_chunking: bool = False,
        obs_keys: list[str] | None = None,
        lowdim: bool = False,
        manual_lookahead: bool = False,
        opex: bool = False,
        discriminator = None,
        expert_policy = None,
    ) -> RolloutPolicy:
    match policy_type:
        case PolicyType.STABLE_BASELINES:
            return StableBaselinesRolloutPolicy(
                policy, 
            )
        case PolicyType.ROBOMIMIC:
            if opex:
                return RobomimicOpexRolloutPolicy(
                    policy, 
                    action_chunking=action_chunking, 
                    n_obs_steps=n_obs_steps,
                    is_diffusion=config.training.algo.is_diffusion()
                )
            if manual_lookahead:
                return ManualLookaheadRolloutPolicy(
                    policy, 
                    action_chunking=action_chunking, 
                    n_obs_steps=n_obs_steps,
                    is_diffusion=config.training.algo.is_diffusion()
                )
            if config.training.algo == ImitationAlgorithm.IDQL:
                return RobomimicSamplingRolloutPolicy(
                    policy, 
                    action_chunking=action_chunking, 
                    n_obs_steps=n_obs_steps,
                    is_diffusion=config.training.algo.is_diffusion(),
                    discriminator=discriminator,
                    expert_policy=expert_policy,
                )
            return RobomimicRolloutPolicy(
                policy, 
                action_chunking=action_chunking,
                n_obs_steps=n_obs_steps,
                is_diffusion=config.training.algo.is_diffusion()
            )
        case PolicyType.DIFFUSION:
            return DiffusionRolloutPolicy(
                policy, 
                action_chunking=action_chunking, 
                n_obs_steps=n_obs_steps, 
                obs_keys=obs_keys, 
                lowdim=lowdim
            )
        case PolicyType.OGBENCH:
            return OGBenchRolloutPolicy(
                policy,
                action_chunking=action_chunking,
                n_obs_steps=n_obs_steps,
                obs_keys=obs_keys,
                lowdim=lowdim,
            )
        case PolicyType.PD:
            return PDRolloutPolicy(
                policy=None,
                target_pos=config.exp.target_pos,
            )
        case _:
            raise NotImplementedError


def get_policy_info(policy_type: PolicyType, config: EvaluationConfig) -> tuple:
    action_chunking = True if policy_type in [PolicyType.DIFFUSION] else False
    n_obs_steps = config.training.obs_horizon if policy_type in [PolicyType.DIFFUSION] else None
    all_obs_keys = None
    only_lowdim = False
    obs_modality_specs = None
    if policy_type == PolicyType.DIFFUSION or policy_type == PolicyType.OGBENCH:
        lowdim_keys = OmegaConf.to_container(config.training.low_dim_keys)
        rgb_keys = OmegaConf.to_container(config.training.rgb_keys)
        depth_keys = OmegaConf.to_container(config.training.depth_keys)
        all_obs_keys = []
        for key in lowdim_keys + rgb_keys + depth_keys:
            if key in all_obs_keys:
                raise ValueError(f"Duplicate obs key {key}")
            all_obs_keys.append(key)
        only_lowdim = lowdim_keys and not rgb_keys and not depth_keys
        obs_modality_specs = RobomimicObsWrapper.get_specs_from_obs_keys(lowdim_keys, rgb_keys, depth_keys)
    elif policy_type == PolicyType.ROBOMIMIC:
        if config.training.algo in [ImitationAlgorithm.IQL_DIFFUSION, ImitationAlgorithm.IDQL, ImitationAlgorithm.IDQL_QSM]:
            action_chunking = True
            n_obs_steps = config.training.observation_horizon
    elif policy_type == PolicyType.PD:
        obs_modality_specs = {}
    return action_chunking, n_obs_steps, all_obs_keys, only_lowdim, obs_modality_specs


def get_policy(policy_type: PolicyType, config: EvaluationConfig, device: torch.device, discriminator=None, expert_policy=None) -> tuple[RolloutPolicy, dict | None]:
    if policy_type == PolicyType.COMPOSITE:
        all_policies = []
        all_obs_modality_specs = []
        for policy_cfg_str in config.exp.policy_configs:
            policy_cfg: EvaluationConfig = compose(policy_cfg_str)
            rollout_policy, obs_modality_specs = get_policy(policy_cfg.exp.policy_type, policy_cfg, device)
            if obs_modality_specs is not None:
                all_obs_modality_specs.append(obs_modality_specs)
            all_policies.append(rollout_policy)
                
        obs_modality_specs = merge_obs_specs(all_obs_modality_specs)
        
        rollout_policy = create_combined_policy(
            policies=all_policies,
            combined_policy_type=config.exp.combined_policy_type,
            n_envs=config.exp.num_workers,
        )
        
        return rollout_policy, obs_modality_specs

    policy_dir = get_policy_dir(config.exp.policy_type, config)
    action_chunking, n_obs_steps, all_obs_keys, lowdim, obs_modality_specs = get_policy_info(config.exp.policy_type, config)
    
    if policy_type == PolicyType.STABLE_BASELINES:
        # we need env in order to load sb policy
        temp_env = get_env(config.exp.policy_type, config, obs_modality_specs, n_envs_override=1)
    else:
        temp_env = None
        
    policy = load_policy(
        policy_type=config.exp.policy_type, 
        config=config, 
        policy_dir=policy_dir, 
        env=temp_env, 
        device=device
    )
    if policy_type == PolicyType.DISCRIMINATOR:
        return policy, None
    
    if config.exp.use_discriminator:
        assert discriminator is not None, "Discriminator policy not found"
    
    rollout_policy = create_rollout_policy(
        policy, 
        policy_type=config.exp.policy_type,
        config=config,
        obs_keys=all_obs_keys,
        lowdim=lowdim,
        action_chunking=action_chunking,
        n_obs_steps=n_obs_steps,
        manual_lookahead=config.exp.manual_lookahead,
        opex=config.exp.opex,
        discriminator=discriminator,
        expert_policy=expert_policy,
    )
    
    return rollout_policy, obs_modality_specs
    

def setup_eval(config: EvaluationConfig):    
    device = setup_experiment(config.exp, eval=True)

    log_video = config.exp.log_video
    render = config.env.render
    
    policy_dir = get_policy_dir(config.exp.policy_type, config)
    
    if config.exp.use_discriminator:
        import copy
        discriminator_config = copy.deepcopy(config)
        discriminator_config.exp.policy_type = PolicyType.DISCRIMINATOR
        discriminator_config.name = config.exp.discriminator_name
        discriminator, _ = get_policy(PolicyType.DISCRIMINATOR, discriminator_config, device)
        
        expert_config = copy.deepcopy(config)
        expert_config.exp.policy_type = PolicyType.DIFFUSION
        expert_config.exp.use_discriminator = False
        expert_config.exp.discriminator_name = None
        expert_policy, _ = get_policy(PolicyType.DIFFUSION, expert_config, device)
    else:
        discriminator = None

    policy, obs_modality_specs = get_policy(config.exp.policy_type, config, device, discriminator=discriminator)

    if obs_modality_specs is not None:
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)
    env = get_env(config.exp.policy_type, config, obs_modality_specs)
    env = PossibleVecEnv(env)

    # human render takes precedence over video recording
    if log_video and not render:
        video_dir: Path = policy_dir / 'videos_eval'
    
        if not video_dir.exists():
            video_dir.mkdir()
            
        default_size = tuple((s[0] if not env.is_vec_env else s[0][0]) for s in env.get_camera_sizes())
        size = default_size if config.exp.video_dims is None else config.exp.video_dims
        fps = 20
        video_writers = []
        for i in range(config.exp.num_workers):
            video_writer = cv2.VideoWriter(str(video_dir / f'rollout_video_{i}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
            video_writers.append(video_writer)
            
        render_mode = 'rgb_array'
    elif render:
        render_mode = 'human'
        video_writers = None
    else:
        render_mode = None
        video_writers = None
        
    return env, policy, render_mode, video_writers
        
        
@hydra.main(version_base=None, config_path='../configs', config_name="eval_robomimic")
def eval(config: EvaluationConfig): 
    config.env.horizon = config.exp.rollout_horizon
    env, policy, render_mode, video_writers = setup_eval(config)
            
    save_dataset = config.exp.save_to_dataset
    rollout_stats = RolloutStats()
    
    if save_dataset:
        dataset_name = f'{config.exp.dataset_name}.hdf5'
        if not ROLLOUT_DATA_DIR.exists():
            ROLLOUT_DATA_DIR.mkdir()
        dataset_path = ROLLOUT_DATA_DIR / dataset_name
        total_samples = 0
        demo_num = 0
        
    if config.exp.reset_to_data:
        ref_dataset_path = get_dataset(config.data)
    else:
        ref_dataset_path = None
        
    try:
        with (
            video_writer_manager(video_writers) as video_writers, 
            (h5py.File(dataset_path, 'w') if save_dataset else nullcontext()) as dataset_file,
            (h5py.File(ref_dataset_path, 'r') if ref_dataset_path is not None else nullcontext()) as ref_dataset_file
        ):
            if save_dataset:
                data_grp = dataset_file.create_group('data')                
            
            for i in range(config.exp.num_rollouts):
                if ref_dataset_file is not None:
                    ind = 9
                    
                    eef_pos = ref_dataset_file['data'][f'demo_{ind}']['obs']['robot0_eef_pos'] 
                    # find first index where z < 0.5
                    try:
                        idx = np.where(eef_pos[:, 2] < 0.835)[0][0]
                    except IndexError:
                        idx = 0
                    
                    reset_state = ref_dataset_file['data'][f'demo_{ind}']['states'][idx]
                else:
                    reset_state = None
                                        
                traj = rollout(
                    policy=policy,
                    eval_envs=env,
                    rollout_stats=rollout_stats,
                    reset_state=reset_state,
                    logger=None,
                    total_steps=config.exp.rollout_horizon,
                    render_mode=render_mode,
                    log_video=config.exp.log_video,
                    video_writers=video_writers,
                    render_dims=config.exp.video_dims,
                    video_skip=config.exp.video_skip,
                    return_trajectory=save_dataset,
                    end_on_success=False,
                )
                if save_dataset:
                    # save traj
                    demo_num, added_samples = add_trajectory_to_dataset(
                        demo_num,
                        traj,
                        dataset_file
                    )
                    total_samples += added_samples
                
                rollout_stats.compute_statistics()

                if config.exp.verbose:
                    rollout_stats.print_stats(i, total=config.exp.num_rollouts)
        
            if save_dataset:
                data_grp.attrs['total'] = total_samples
                
                hydra_config = HydraConfig.get()
                config_name = hydra_config.job.config_name
                data_grp.attrs['config_name'] = config_name
                
                env_args = env.serialize()
                if "placement_initializer" in env_args['env_kwargs']:
                    env_args['env_kwargs']["placement_initializer"] = None
                data_grp.attrs['env_args'] = json.dumps(env_args, indent=4)

                fill_metadata(
                    dataset_file,
                    dataset_id=str(uuid.uuid4()),
                    dataset_type=DatasetType.RL,
                    action_type=ActionType.RELATIVE,
                    env_type=EnvType.ROBOMIMIC,
                    env_name=config.env.env_name.value,
                    is_real=False,
                    include_date=True,
                    include_creator=True,
                )
                
                print(f"Dataset saved to {dataset_path}")
                print(f"Total samples added to dataset: {total_samples}")
        if save_dataset:
            edit_dataset_description(dataset_path)
    except Exception as e:
        # remove partially completed dataset if it was created
        if save_dataset:
            dataset_path.unlink()
        raise e
