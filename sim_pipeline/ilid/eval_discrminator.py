import os
import h5py
import torch
import click

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from sim_pipeline.ilid.discriminator import Discriminator


def classify_states(
    data_path: str,
    discriminator: Discriminator, 
    device: torch.device,
):
    with h5py.File(data_path, 'r') as f:        
        data = f['data']
        for ep_key in data:
            ep = data[ep_key]
            obs = ep['obs']
            # Create batch of observations for entire episode
            obs_dict = {k: obs[k][:] for k in obs.keys() if k in ['agentview_image', 'robot0_eye_in_hand_image', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']}

            # reshape image keys to (C, H, W)
            for k in obs_dict.keys():
                if 'image' in k:
                    obs_dict[k] = obs_dict[k].transpose(0, 3, 1, 2)
                    # scale from (0, 255) to (0, 1)
                    obs_dict[k] = obs_dict[k].astype(float) / 255.0
            ob = TensorUtils.to_tensor(obs_dict)
            ob = TensorUtils.to_device(ob, device)
            ob = TensorUtils.to_float(ob)
            # import ipdb; ipdb.set_trace()
            # Get predictions for all states at once
            predictions = discriminator.policy.get_action(ob)
            
            print(predictions)
                            
def load_discriminator(discriminator_path: str, device: torch.device):
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=discriminator_path, device=device, verbose=True, discriminator=True)
    return policy

@click.command()
@click.option('--dataset', '-d', help='Datset file path')
@click.option('--discriminator_path', '-dp', help='Discriminator path')              
def run(dataset, discriminator_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    discriminator = load_discriminator(discriminator_path, device)

    classify_states(dataset, discriminator, device)
    
if __name__ == '__main__':
    run()