import hydra

from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf
from sim_pipeline.configs._diffusion.base_diffusion import DiffusionBaseConfig
from sim_pipeline.configs.constants import PolicyType
from sim_pipeline.utils.experiment import setup_experiment
from sim_pipeline.data_manager.get_dataset import get_dataset
from sim_pipeline.utils.shape_meta import get_shape_meta, update_diffusion_cfg_with_shape_meta
from sim_pipeline.utils.diffusion_utils import get_policy_class, get_workspace
from sim_pipeline.utils.train_utils import get_latest_diffusion_policy
from sim_pipeline.model_manager.create_model_metadata import create_model_metadata, update_metadata_trained
import wandb


@hydra.main(version_base=None, config_path='../configs', config_name='base_diffusion')
def train(config: DiffusionBaseConfig):
    import traceback
    import sys
    try:
        run(config)
    except:
        traceback.print_exc(file=sys.stderr)
        raise


def run(cfg: DiffusionBaseConfig):
    # device = setup_experiment(cfg.exp)

    wandb.init(
    # set the wandb project where this run will be logged
        project="diffusion",
        entity="kehuang",
    )


    dataset_path = get_dataset(cfg.data)
    
    lowdim_keys = cfg.training.low_dim_keys
    rgb_keys = cfg.training.rgb_keys
    depth_keys = cfg.training.depth_keys
    all_obs_keys = []
    for key in lowdim_keys + rgb_keys + depth_keys:
        if key in all_obs_keys:
            raise ValueError(f"Duplicate obs key {key}")
        all_obs_keys.append(key)
    only_lowdim = lowdim_keys and not rgb_keys and not depth_keys

    obs_shape_meta, action_shape_meta = get_shape_meta(dataset_path, all_obs_keys)
    total_obs_dim = 0
    for shape in obs_shape_meta.values():
        total_obs_dim += shape[0]
    
    default_config_path = cfg.training.default_config_path
    
    default_cfg = OmegaConf.load(default_config_path)
    
    default_cfg.n_action_steps = cfg.training.execution_horizon
    default_cfg.n_obs_steps = cfg.training.obs_horizon
    
    default_cfg.name = cfg.name
    default_cfg.training.seed = cfg.exp.seed
    default_cfg.task.env_runner.test_start_seed = cfg.exp.seed
    
    if 'shape_meta' in default_cfg:
        shape_meta_cfg = default_cfg.shape_meta
        update_diffusion_cfg_with_shape_meta(shape_meta_cfg, obs_shape_meta, action_shape_meta, rgb_keys, lowdim_keys, depth_keys)
    if only_lowdim:
        default_cfg.obs_dim = total_obs_dim
    
    policy_cfg = default_cfg.policy
    
    policy_cfg.n_action_steps = cfg.training.execution_horizon
    policy_cfg.n_obs_steps = cfg.training.obs_horizon
    policy_cfg.horizon = cfg.training.prediction_horizon
        
    if 'shape_meta' in policy_cfg:
        policy_shape_meta_cfg = policy_cfg.shape_meta
        update_diffusion_cfg_with_shape_meta(policy_shape_meta_cfg, obs_shape_meta, action_shape_meta, rgb_keys, lowdim_keys, depth_keys)
    if only_lowdim:
        policy_cfg.obs_dim = total_obs_dim
    
    policy_cls = get_policy_class(rgb_keys, lowdim_keys, depth_keys)
    
    env_runner_cfg = default_cfg.task.env_runner
    
    env_runner_cfg.n_action_steps = cfg.training.execution_horizon
    env_runner_cfg.n_obs_steps = cfg.training.obs_horizon
    
    if only_lowdim:
        default_cfg.task.obs_dim = total_obs_dim
        default_cfg.task.dataset.obs_keys = tuple(all_obs_keys)
        env_runner_cfg.obs_keys = tuple(all_obs_keys)
        default_cfg.task.obs_keys = tuple(all_obs_keys)
        
    default_cfg.task.dataset.dataset_path = dataset_path
    default_cfg.task.dataset_path = dataset_path
    default_cfg.task.env_runner.dataset_path = dataset_path
    
    training_cfg = default_cfg.training
    training_cfg.device = f'cuda:{cfg.exp.device_id}'
    training_cfg.checkpoint_every = cfg.exp.save_interval
    training_cfg.rollout_every = cfg.exp.eval_interval
    training_cfg.num_epochs = cfg.training.num_epochs
    
    default_cfg.dataloader.batch_size = cfg.training.batch_size
    default_cfg.val_dataloader.batch_size = cfg.training.batch_size
    
    default_cfg.logging.project = cfg.exp.project
    default_cfg.logging.entity = cfg.exp.entity
    default_cfg.logging.name = cfg.name
    
    # default_cfg.training.debug = cfg.debug
    env_runner_cfg.debug = cfg.debug
    if cfg.debug:
        default_cfg.task.env_runner.n_envs = 1
        default_cfg.debug = True
    else:
        default_cfg.debug = False
    
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d%H%M%S")
    log_dir = Path(cfg.exp.logdir) / cfg.name / dt_string
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
        
    create_model_metadata(
        name=cfg.name,
        policy_type=PolicyType.DIFFUSION,
        policy_identifier=dt_string,
        dataset_path=dataset_path,
        hydra_config=cfg,
        filepath=log_dir / 'metadata.json',
    )

    try:
        OmegaConf.resolve(default_cfg)
        workspace_cls = get_workspace(rgb_keys, lowdim_keys, depth_keys)
        workspace = workspace_cls(default_cfg, output_dir=log_dir)
        workspace.run()
    finally:
        max_epoch_saved: int = get_latest_diffusion_policy(log_dir / 'checkpoints', find_epoch=True)
        # only consider a training run valid if we have ran for more than 1000 (hardcoded) epochs
        if max_epoch_saved >= 1000:
            update_metadata_trained(log_dir / 'metadata.json')  
