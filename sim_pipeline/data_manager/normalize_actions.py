import click
import os
import h5py
import uuid
import numpy as np
from tqdm import tqdm

from contextlib import nullcontext
from robomimic.scripts.split_train_val import split_train_val_from_hdf5

@click.command()
@click.option('--dataset', '-d', help='Datset file path')
@click.option('--output_name', '-o', help='Output file name')
def normalize_dataset_actions(dataset, output_name):
    print(f'Normalizing actions in dataset {dataset}')
    # output file in same directory as input file
    output_path = os.path.join(os.path.dirname(dataset), output_name)

    with (
        h5py.File(dataset, 'r') as f,
        h5py.File(output_path, 'w') as f_normalized,
    ):
        data = f['data']
        out_data = f_normalized.create_group("data")
        
        # First pass - compute dataset-wide action statistics
        print("Computing action statistics...")
        action_list = []
        for ep_key in tqdm(data):
            ep = data[ep_key]
            actions = ep['actions'][:]
            action_list.append(actions)
        
        all_actions = np.concatenate(action_list, axis=0)
        
        # Compute min and max for all dims except gripper
        action_min = np.min(all_actions[:, :-1], axis=0)  # shape: (n_dims-1,)
        action_max = np.max(all_actions[:, :-1], axis=0)  # shape: (n_dims-1,)
        
        # Save normalization stats
        stats_grp = f_normalized.create_group("action_normalization_stats")
        stats_grp.create_dataset("min", data=action_min)
        stats_grp.create_dataset("max", data=action_max)
                
        # Second pass - normalize and save actions
        print("Normalizing actions...")
        demo_counter = 0
        for ep_key in tqdm(data):
            ep = data[ep_key]
            
            data.copy(
                source=ep,
                dest=out_data,
                name=f'demo_{demo_counter}'
            )
            out_data_grp = out_data[f'demo_{demo_counter}']
            
            # Get actions and normalize all dims except gripper
            actions = out_data_grp['actions'][:]
            # Scale to [0,1] then to [-0.5, 0.5]
            actions[:, :-1] = (actions[:, :-1] - action_min) / (action_max - action_min) - 0.5
            
            # Save normalized actions
            del out_data_grp['actions']
            out_data_grp.create_dataset('actions', data=actions)
            
            demo_counter += 1
            
            # copy episode attributes
            for key, val in ep.attrs.items():
                out_data_grp.attrs[key] = val

        # copy dataset attributes
        for key, val in data.attrs.items():
            if key == 'dataset_id':
                out_data.attrs[key] = str(uuid.uuid4())
            else:
                out_data.attrs[key] = val
                
    # recreate the train-validation split
    split_train_val_from_hdf5(hdf5_path=output_path, val_ratio=0.1)
    
if __name__ == "__main__":
    normalize_dataset_actions()