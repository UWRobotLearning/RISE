import click
import os
import h5py
import uuid
import numpy as np
import random

from robomimic.scripts.split_train_val import split_train_val_from_hdf5

@click.command()
@click.option('--dataset', '-d', help='Dataset file path')
@click.option('--output_name', '-o', help='Output file name')
@click.option('--proportion', '-p', type=click.FloatRange(0, 1), help='Proportion of dataset to keep', default=1.0)
@click.option('--amount', '-n', type=click.IntRange(0), help='Number of demos to keep', default=None)
@click.option('--randomize', '-r', is_flag=True, help='Randomize the selection of episodes')
def chop_dataset(dataset, output_name, proportion, amount, randomize):
    print(f'Chopping dataset {dataset}')
    if proportion == 0 or (amount is not None and amount == 0):
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
        if amount is not None:
            demos_to_keep = min(amount, num_demos)
        else:
            demos_to_keep = int(num_demos * proportion)
        
        demo_keys = list(data.keys())
        if randomize:
            random.shuffle(demo_keys)
        
        new_demos = []
        new_total = 0
        demo_counter = 0
        for ep_key in demo_keys:
            if demo_counter >= demos_to_keep:
                break
            ep = data[ep_key]
            new_demos.append(ep_key)
            new_total += ep['actions'].shape[0]
            
            data.copy(
                source=ep,
                dest=out_data,
                name=f'demo_{demo_counter}'
            )
            out_data_grp = out_data[f'demo_{demo_counter}']
            demo_counter += 1
            
            # copy episode attributes
            for key, val in ep.attrs.items():
                out_data_grp.attrs[key] = val

        # copy dataset attributes
        for key, val in data.attrs.items():
            if key == 'total':
                out_data.attrs[key] = new_total
            elif key == 'dataset_id':
                out_data.attrs[key] = str(uuid.uuid4())
            elif key == 'description':
                out_data.attrs[key] = val + f' chopped w/ {demos_to_keep} demos'
            else:
                out_data.attrs[key] = val
                
    # recreate the train-validation split
    split_train_val_from_hdf5(hdf5_path=output_path, val_ratio=0.1)
    
if __name__ == "__main__":
    chop_dataset()