import os
import h5py
import torch
import click
from tqdm import tqdm
import uuid
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from sim_pipeline.ilid.discriminator import Discriminator


def classify_states(
    filtered_data_path: str,
    suboptimal_data_path: str, 
    discriminator: Discriminator, 
    device: torch.device,
    threshold: float = 0.0,
):
    with h5py.File(filtered_data_path, 'w') as f:
        data_grp = f.create_group('data')
        
        demo_counter = 0
        
        with h5py.File(suboptimal_data_path, 'r') as suboptimal_data:
            data = suboptimal_data['data']
            # Create progress bar for episodes
            for ep_key in tqdm(data, desc="Processing episodes"):
                ep = data[ep_key]
                obs = ep['obs']
                expert_like_states = []
                # Create batch of observations for entire episode
                obs_dict = {k: obs[k][:] for k in obs.keys()}
                # reshape image keys to (C, H, W)
                for k in obs_dict.keys():
                    if 'image' in k:
                        obs_dict[k] = obs_dict[k].transpose(0, 3, 1, 2)
                        # scale from (0, 255) to (0, 1)
                        obs_dict[k] = obs_dict[k].astype(float) / 255.0
                ob = TensorUtils.to_tensor(obs_dict)
                ob = TensorUtils.to_device(ob, device)
                ob = TensorUtils.to_float(ob)
                # Get predictions for all states at once
                predictions = discriminator.policy.get_action(ob)
                                
                # Find indices where prediction exceeds threshold
                expert_like_states = [i for i, pred in enumerate(predictions) if pred > threshold]
                                        
                if expert_like_states:
                    last_index = expert_like_states[-1]
                    
                    # Create new trajectory up to the last expert state
                    new_obs = {k: obs[k][:last_index+1] for k in obs.keys()}
                    new_actions = ep['actions'][:last_index+1]
                    new_rewards = ep['rewards'][:last_index+1]
                    new_dones = ep['dones'][:last_index+1]
                    next_obs = ep['next_obs']
                    new_states = ep['states'][:last_index+1]
                    new_next_obs = {k: next_obs[k][:last_index+1] for k in next_obs.keys()}
                    
                    # Save the new trajectory to the new dataset
                    ep_grp = data_grp.create_group(f'demo_{demo_counter}')
                    obs_grp = ep_grp.create_group('obs')
                    for k, v in new_obs.items():
                        obs_grp.create_dataset(k, data=v)
                    ep_grp.create_dataset('actions', data=new_actions)
                    ep_grp.create_dataset('rewards', data=new_rewards)
                    ep_grp.create_dataset('dones', data=new_dones)
                    ep_grp.create_dataset('states', data=new_states)
                    next_obs_grp = ep_grp.create_group('next_obs')
                    for k, v in new_next_obs.items():
                        next_obs_grp.create_dataset(k, data=v)
                    demo_counter += 1
                    
                    # copy episode attributes
                    for key, val in ep.attrs.items():
                        ep_grp.attrs[key] = val

            # copy dataset attributes
            for key, val in data.attrs.items():
                if key == 'dataset_id':
                    data_grp.attrs[key] = str(uuid.uuid4())
                else:
                    data_grp.attrs[key] = val
                
def load_discriminator(discriminator_path: str, device: torch.device):
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=discriminator_path, device=device, verbose=True, discriminator=True)
    return policy

@click.command()
@click.option('--dataset', '-d', help='Datset file path')
@click.option('--output_name', '-o', help='Output file name')
@click.option('--discriminator_path', '-dp', help='Discriminator path')              
@click.option('--threshold', '-t', type=float, help='Threshold for classifying states', default=0.0)
def run(dataset, output_name, discriminator_path, threshold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    discriminator = load_discriminator(discriminator_path, device)
    output_path = os.path.join(os.path.dirname(dataset), output_name)

    classify_states(output_path, dataset, discriminator, device, threshold=threshold)