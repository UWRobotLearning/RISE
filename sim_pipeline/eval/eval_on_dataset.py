import numpy as np
import h5py
import torch
import click

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.tensor_utils as TensorUtils


def eval_on_dataset(
    data_path: str,
    policy,
    device: torch.device,
):
    with h5py.File(data_path, 'r') as f:        
        data = f['data']
        for ep_key in data:
            if ep_key != 'demo_1':
                continue
            ep = data[ep_key]
            obs = ep['obs']
            # Create batch of observations for entire episode
            obs_dict = {k: obs[k][:] for k in obs.keys() if k in ['front_image', 'wrist_image', 'states']}
            # Prepare observation history with a length of 2
            obs_history = {k: np.zeros((len(obs['front_image']), 2) + obs_dict[k].shape[1:]) for k in obs_dict.keys()}
            for t in range(len(obs['front_image'])):
                # Collect the current and previous observations
                for k in obs_dict.keys():
                    if t == 0:
                        # If it's the first timestep, duplicate the observation
                        obs_history[k][t, 0] = obs_dict[k][t]
                        obs_history[k][t, 1] = obs_dict[k][t]
                    else:
                        obs_history[k][t, 0] = obs_dict[k][t-1]
                        obs_history[k][t, 1] = obs_dict[k][t]

            # Reshape image keys to (C, H, W) and scale
            for k in obs_history.keys():
                if 'image' in k:
                    obs_history[k] = obs_history[k].transpose(0, 1, 4, 2, 3)
                    obs_history[k] = obs_history[k].astype(float) / 255.0

            ob = TensorUtils.to_tensor(obs_history)
            ob = TensorUtils.to_device(ob, device)
            ob = TensorUtils.to_float(ob)
            # Get predictions for all states at once
            predictions = policy.policy.get_action(ob)
            
            import ipdb; ipdb.set_trace()
            
            print(predictions)
            
def load_policy(policy_path: str, device: torch.device):
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=policy_path, device=device, verbose=True)
    return policy

@click.command()
@click.option('--dataset', '-d', help='Datset file path')
@click.option('--policy_path', '-p', help='Policy path')              
def run(dataset, policy_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = load_policy(policy_path, device)

    eval_on_dataset(dataset, policy, device)
    
if __name__ == '__main__':
    run()