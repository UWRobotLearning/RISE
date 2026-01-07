import click
import os
import h5py
import uuid
import numpy as np

from contextlib import nullcontext
from robomimic.scripts.split_train_val import split_train_val_from_hdf5

@click.command()
@click.option('--dataset', '-d', help='Datset file path')
@click.option('--output_name', '-o', help='Output file name')
@click.option('--const_reward', '-c', type=float, help='Constant reward', default=0.0)
@click.option('--copy_from_dataset', '-cf', type=str, help='Copy rewards from this dataset path', default=None)
@click.option('--monotonic_increase', '-m', is_flag=True, help='Whether to monotonically increase rewards')
def relabel_dataset_rewards(dataset, output_name, const_reward, copy_from_dataset, monotonic_increase):
    print(f'Relabeling rewards in dataset {dataset}')
    # output file in same directory as input file
    output_path = os.path.join(os.path.dirname(dataset), output_name)

    with (
        h5py.File(dataset, 'r') as f,
        h5py.File(output_path, 'w') as f_relabeled,
        h5py.File(copy_from_dataset, 'r') if copy_from_dataset is not None else nullcontext() as f_copy_from,
    ):
        data = f['data']
        out_data = f_relabeled.create_group("data")
                
        demo_counter = 0
        for ep_key in data:
            ep = data[ep_key]
            
            data.copy(
                source=ep,
                dest=out_data,
                name=f'demo_{demo_counter}'
            )
            out_data_grp = out_data[f'demo_{demo_counter}']
            if copy_from_dataset is not None:
                copy_from_data = f_copy_from['data']
                copy_from_ep = copy_from_data[ep_key]
                out_data_grp['rewards'][:] = copy_from_ep['rewards'][:]
            else:
                if 'rewards' not in out_data_grp:
                    out_data_grp.create_dataset('rewards', shape=(ep['actions'].shape[0],), dtype=np.float32)
                
                if monotonic_increase:
                    # Create linearly increasing rewards from 1 to 2
                    num_steps = ep['actions'].shape[0]
                    rewards = np.linspace(1.0, 2.0, num_steps)
                    out_data_grp['rewards'][:] = rewards
                else:
                    out_data_grp['rewards'][:] = np.ones_like(out_data_grp['rewards']) * const_reward
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
    relabel_dataset_rewards()