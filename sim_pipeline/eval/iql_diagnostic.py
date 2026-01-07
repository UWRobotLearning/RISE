import hydra

import torch
import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.train_utils as TrainUtils
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from torch.utils.data import DataLoader

from omegaconf import OmegaConf
from robomimic.algo.iql import IQL

from sim_pipeline.configs.constants import PolicyType
from sim_pipeline.configs._evaluation.eval_base import EvaluationConfig
from sim_pipeline.data_manager.get_dataset import get_dataset
from sim_pipeline.eval.eval import get_policy
from sim_pipeline.training.train_robomimic import get_robomimic_config
from sim_pipeline.eval.rollout_policies.rollout_policy import RolloutPolicy
from sim_pipeline.utils.experiment import setup_experiment

@hydra.main(version_base=None, config_path='../configs', config_name="eval_robomimic")
def plot_advantages(config: EvaluationConfig): 
    device = setup_experiment(config.exp, True)
    if config.exp.policy_type != PolicyType.ROBOMIMIC:
        raise ValueError("This script is only for Robomimic policies")
    rollout_policy: RolloutPolicy
    rollout_policy, _ = get_policy(config, config, device)
    policy: IQL = rollout_policy.policy.policy
    
    dataset_path = get_dataset(config.data)
    robomimic_config = get_robomimic_config(config, dataset_path)
    
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path,
        all_obs_keys=robomimic_config.all_obs_keys,
    )
    
    trainset, validset = TrainUtils.load_data_for_training(
        robomimic_config, obs_keys=shape_meta["all_obs_keys"])
    train_sampler = trainset.get_dataset_sampler()
    
    data_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=100,
        shuffle=(train_sampler is None),
        generator=torch.Generator(device=device),
        num_workers=robomimic_config.train.num_data_workers,
        drop_last=True
    )
    
    all_pos = []
    all_next_pos = []
    all_adv = []
        
    count = 0
    for batch in data_loader:
        input_batch = policy.process_batch_for_training(batch)
        input_batch = policy.postprocess_batch_for_training(input_batch, obs_normalization_stats=None)
        
        obs = input_batch['obs']
        next_obs = input_batch['next_obs']
    
        _, _, critic_info = policy._compute_critic_loss(input_batch)

        q_pred = critic_info['vf/q_pred']
        v_pred = critic_info['vf/v_pred']
        
        # adv = v_pred
        adv = q_pred - v_pred
        adv_weights = policy._get_adv_weights(adv)
        
        # To - 1
        object_pos = obs['object'][:, 1, :2]
        next_object_pos = next_obs['object'][:, 1, :2]
        
        all_pos.append(object_pos.detach().cpu().numpy())
        all_next_pos.append(next_object_pos.detach().cpu().numpy())
        all_adv.append(q_pred.detach().cpu().numpy())
        count += 1
        if count > 250:
            break
    all_pos = np.concatenate(all_pos, axis=0)
    all_next_pos = np.concatenate(all_next_pos, axis=0)
    all_adv = np.concatenate(all_adv, axis=0)
    
    # filtered = all_adv < 0.8
    # all_pos = all_pos[filtered]
    # all_next_pos = all_next_pos[filtered]
    # all_adv = all_adv[filtered]
    
    # Compute reasonable min and max values for the colorbar
    val_min = np.percentile(all_adv, 5)  # 5th percentile
    val_max = np.percentile(all_adv, 95)  # 95th percentile
    norm = Normalize(vmin=val_min, vmax=val_max)

    goal = (-0.1125, 0.1675)
    
    vectors = all_next_pos - all_pos
    fig, ax = plt.subplots()
    quiver = ax.quiver(all_pos[:, 0], all_pos[:, 1], vectors[:, 0], vectors[:, 1], all_adv, norm=norm, angles='xy')
    cbar = plt.colorbar(quiver)
    # cbar.set_label('adv')

    ax.scatter(x=goal[0], y=goal[1], c='r', marker='o', s=50)

    # Adjust colorbar ticks to show min, max, and clipped ranges
    cbar.ax.set_yticklabels([f'<{val_min:.2f}', f'{val_min:.2f}', 
                            f'{(val_min+val_max)/2:.2f}', f'{val_max:.2f}', 
                            f'>{val_max:.2f}'])
    plt.show()
    # plt.savefig('advantages.svg')
    
        
if __name__ == "__main__":
    plot_advantages()