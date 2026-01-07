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
def relabel_dataset_rewards(dataset, output_name):
    print(f'Relabeling rewards in dataset {dataset}')
    # output file in same directory as input file
    output_path = os.path.join(os.path.dirname(dataset), output_name)

    with (
        h5py.File(dataset, 'r') as f,
        h5py.File(output_path, 'w') as f_relabeled,
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

            # Reshape image from (episode_len, H, W, 3) to (episode_len, 3, 1, 80, 80)
            image = out_data_grp['obs']['agentview_image'][:]
            # First reshape to move channels dimension
            image = np.transpose(image, (0, 3, 1, 2))
            # Get current dimensions
            _, _, h, w = image.shape
            # Center crop to 80x80
            h_start = (h - 80) // 2
            w_start = (w - 80) // 2
            image = image[:, :, h_start:h_start+80, w_start:w_start+80]
            # Add singleton dimension
            image = np.expand_dims(image, axis=2)
            out_data_grp['obs']['agentview_image'][:] = image
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