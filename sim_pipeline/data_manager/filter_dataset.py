import click
import os
import h5py
import uuid
import numpy as np

from robomimic.scripts.split_train_val import split_train_val_from_hdf5

@click.command()
@click.option('--dataset', '-d', help='Datset file path')
@click.option('--output_name', '-o', help='Output file name')
def filter_dataset(dataset, output_name):
    print(f'Filtering dataset {dataset}')
    # output file in same directory as input file
    output_path = os.path.join(os.path.dirname(dataset), output_name)

    with (
        h5py.File(dataset, 'r') as f,
        h5py.File(output_path, 'w') as f_filtered
    ):
        data = f['data']
        out_data = f_filtered.create_group("data")
        
        new_demos = []
        new_total = 0
        demo_counter = 0
        for ep_key in data:
            ep = data[ep_key]
            # end_obj_pos = ep['obs']['object'][-1]
            # filter if object xy position is within the range
            # if -0.14 <= end_obj_pos[0] <= 0.0 and 0.11 <= end_obj_pos[1] <= 0.2:
            obs_keys = ep['obs'].keys()
            include = True
            for key in obs_keys:
                if key not in ['123622270810_rgb', '207322251049_rgb', 'action', 'lang_embed', 'language_instruction', 'lowdim_ee', 'lowdim_qpos']:
                    include = False
                    break
            for key in ['123622270810_rgb', '207322251049_rgb']:
                if key not in obs_keys:
                    include = False
                    break
                
            if include:
                new_demos.append(ep_key)
                new_total += ep['actions'].shape[0]
                
                data.copy(
                    source=ep,
                    dest=out_data,
                    name=f'demo_{demo_counter}'
                )
                out_data_grp = out_data[f'demo_{demo_counter}']
                demo_counter += 1
                
                out_data_grp['obs'].move('123622270810_rgb', 'eye_in_hand_image')
                out_data_grp['obs'].move('207322251049_rgb', 'agentview_image')
                
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
                out_data.attrs[key] = val + ' filtered'
            else:
                out_data.attrs[key] = val
                
    # recreate the train-validation split
    split_train_val_from_hdf5(hdf5_path=output_path, val_ratio=0.0)
    
if __name__ == "__main__":
    filter_dataset()