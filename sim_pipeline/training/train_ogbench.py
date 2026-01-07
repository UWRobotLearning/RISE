import hydra
import h5py
import tqdm
import torch 
import wandb

from collections import defaultdict
from pathlib import Path
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from impls.utils.flax_utils import save_agent

from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import RobomimicReplayLowdimDataset

from sim_pipeline.utils.experiment import setup_experiment, setup_wandb
from sim_pipeline.data_manager.get_dataset import get_dataset
from sim_pipeline.configs._offline_rl.ogbench import OGBenchBaseConfig

@hydra.main(version_base=None, config_path='../configs', config_name='crl')
def train(config: OGBenchBaseConfig):
    device = setup_experiment(config.exp)

    if not config.debug:
        setup_wandb(
            config,
            name=f"{config.name}",
            entity=config.exp.entity,
            project=config.exp.project,
        )
    
    log_dir = Path(config.exp.logdir) / config.name / 'checkpoints'
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
        
    dataset_path = get_dataset(config.data)
    
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
        
    if config.training.rgb_keys:
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
    
    for epoch in range(config.training.num_epochs):
        epoch_metrics = defaultdict(list)
        with tqdm.tqdm(train_dataloader, desc=f'Training epoch {epoch}') as tepoch:
            for batch in tepoch:
                # to numpy
                batch_np = {
                    'observations': batch['obs'].numpy(),
                    'actions': batch['action'].numpy(),
                    'terminals': batch['terminals'].numpy(),
                    'valids': batch['valids'].numpy(),
                    'rewards': batch['rewards'].numpy(),
                    'next_observations': batch['next_obs'].numpy(),
                    'value_goals': batch['value_goals'].numpy(),
                    'actor_goals': batch['actor_goals'].numpy(),
                }
                agent, update_info = agent.update(batch_np)
                
                for k, v in update_info.items():
                    epoch_metrics[f'training/{k}'].append(v)
        # avg over epoch
        epoch_metrics = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}
        if not config.debug:
            wandb.log(epoch_metrics, step=epoch)
            
        if epoch % config.exp.save_interval == 0:
            save_agent(agent, log_dir, epoch)
    save_agent(agent, log_dir, epoch)