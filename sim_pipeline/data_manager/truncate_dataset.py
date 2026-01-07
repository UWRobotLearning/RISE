import click
import os
import h5py
import uuid
import numpy as np
from tqdm import tqdm

from robomimic.scripts.split_train_val import split_train_val_from_hdf5

@click.command()
@click.option('--dataset', '-d', help='Dataset file path')
@click.option('--output_name', '-o', help='Output file name')
@click.option('--proportion', '-p', type=click.FloatRange(0, 1), help='Proportion of dataset to keep', default=1.0)
@click.option('--amount', '-n', type=click.IntRange(0), help='Number of steps to remove from the start of each demo', default=None)
def truncate_dataset(dataset, output_name, proportion, amount):
    print(f'Truncating dataset {dataset}')
    if proportion == 0:
        raise ValueError("No demos to keep")
    # output file in same directory as input file
    output_path = os.path.join(os.path.dirname(dataset), output_name)

    with (
        h5py.File(dataset, 'r') as f,
        h5py.File(output_path, 'w') as f_filtered
    ):
        data = f['data']
        out_data = f_filtered.create_group("data")
        
        num_demos = len(data.keys())
        demos_to_keep = int(num_demos * proportion)
        new_demos = []
        new_total = 0
        demo_counter = 0
        
        for ep_key in tqdm(data, desc="Processing demos", total=demos_to_keep):
            if demo_counter >= demos_to_keep:
                break
            ep = data[ep_key]
            new_demos.append(ep_key)
            
            # Determine the number of steps to keep
            num_steps = ep['actions'].shape[0]
            steps_to_keep = max(0, num_steps - (amount if amount is not None else 0))
            start_index = amount if amount is not None else 0
            new_total += steps_to_keep
            
            # Create a new group for the truncated demo
            out_data_grp = out_data.create_group(f'demo_{demo_counter}')
            
            # Copy truncated data
            for key in ep.keys():
                if isinstance(ep[key], h5py.Dataset):
                    out_data_grp.create_dataset(
                        key, 
                        data=ep[key][start_index:num_steps]
                    )
                else:
                    ep.copy(key, out_data_grp)
            
            # Copy and update episode attributes
            for key, val in ep.attrs.items():
                if key == 'num_samples':
                    out_data_grp.attrs[key] = steps_to_keep
                else:
                    out_data_grp.attrs[key] = val
            
            demo_counter += 1

        # Copy dataset attributes
        for key, val in data.attrs.items():
            if key == 'total':
                out_data.attrs[key] = new_total
            elif key == 'dataset_id':
                out_data.attrs[key] = str(uuid.uuid4())
            elif key == 'description':
                out_data.attrs[key] = val + f' truncated w/ {demos_to_keep} demos'
            else:
                out_data.attrs[key] = val
                
    # Recreate the train-validation split
    split_train_val_from_hdf5(hdf5_path=output_path, val_ratio=0.1)
    
if __name__ == "__main__":
    truncate_dataset()