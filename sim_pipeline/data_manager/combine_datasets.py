import h5py
import json
import numpy as np
import click
import uuid

from sim_pipeline.configs.constants import COMBINED_DATA_DIR

def copy_group(in_group, out_group):
    for key in in_group.keys():
        item = in_group[key]
        if isinstance(item, h5py.Group):
            new_group = out_group.create_group(key)
            for attr_key, attr_value in item.attrs.items():
                new_group.attrs[attr_key] = attr_value
            copy_group(item, new_group)
        else:
            # is a dataset
            out_dataset = out_group.create_dataset(key, data=item[:])
            for attr_key, attr_value in item.attrs.items():
                out_dataset.attrs[attr_key] = attr_value

                
def normalize(arr, stats):
    min_val, max_val = stats["min"], stats["max"]
    return 2 * (arr - min_val) / (max_val - min_val + 1e-7) - 1


def unnormalize(arr, stats):
    min_val, max_val = stats["min"], stats["max"]
    return 0.5 * (arr + 1) * (max_val - min_val) + min_val


@click.command()
@click.option('--datasets', '-d', help='Dataset file path(s)', multiple=True)
@click.option('--output_name', '-o', help='Output file name')
def combine_datasets_command(datasets, output_name):
    combine_datasets(datasets, output_name, default_stem=True)

def combine_datasets(input_files: list[str], output_file: str, default_stem: bool = False):
    # combine hdf5 datasets together. to combine attrs, all copies should be stored
    # in a list with the same keys. 
    
    if default_stem:
        output_file = f'{COMBINED_DATA_DIR}/{output_file}'
    
    with h5py.File(output_file, 'w') as out_f:
        data_group = out_f.create_group('data')
        mask_group = out_f.create_group('mask')
        combined_attrs = {}
        episode_counter = {}
        combined_masks = {}
        
        for input_file in input_files:
            with h5py.File(input_file, 'r') as in_f:
                in_data_group = in_f['data']
                old_key_to_new = {}
                
                for episode_key in in_data_group.keys():
                    # i.e. base_key = 'demo
                    base_key = episode_key.split('_')[0]
                    if base_key not in episode_counter:
                        episode_counter[base_key] = 0
                    else:
                        episode_counter[base_key] += 1
    
                    new_key = f'{base_key}_{episode_counter[base_key]}'
                    old_key_to_new[episode_key] = new_key
                    
                    episode_group = data_group.create_group(new_key)
                    in_episode_group = in_data_group[episode_key]
                    # Copy attributes
                    for attr_key, attr_value in in_episode_group.attrs.items():
                        episode_group.attrs[attr_key] = attr_value
                    
                    copy_group(in_episode_group, episode_group)
                    
                for attr_key, attr_value in in_data_group.attrs.items():
                    if attr_key in ['derived_from', 'combined']:
                        continue
                    
                    if attr_key == 'description':
                        if attr_key not in combined_attrs:
                            combined_attrs[attr_key] = 'Combined: '
                        combined_attrs[attr_key] += attr_value + ' | '
                        combined_attrs[attr_key] = combined_attrs[attr_key].replace('\n', '')
                        continue
                    
                    if attr_key == 'dataset_id':
                        attr_key = 'constituent_dataset_id'

                    if attr_key == 'env_args':
                        combined_attrs[attr_key] = attr_value
                        continue
                                             
                    if attr_key in ['total', 'size']:
                        if attr_key not in combined_attrs:
                            combined_attrs[attr_key] = 0
                        combined_attrs[attr_key] += attr_value
                        continue   
                        
                    if attr_key not in combined_attrs:
                        combined_attrs[attr_key] = []
                    try:
                        attr_value = json.loads(attr_value)
                    except:
                        pass
                    if isinstance(attr_value, list):
                        combined_attrs[attr_key].extend(attr_value)
                    else:
                        combined_attrs[attr_key].append(attr_value)
                
                if 'mask' in in_f:
                    in_mask_group = in_f['mask']
                    for mask_key in in_mask_group.keys():
                        if mask_key not in combined_masks:
                            combined_masks[mask_key] = []
                        combined_masks[mask_key].extend([old_key_to_new[episode_key.decode('utf-8')] for episode_key in in_mask_group[mask_key]])
                
        
        for mask_key, mask_values in combined_masks.items():
            mask_group.create_dataset(mask_key, data=np.array(mask_values).astype('S8'))
        
        # Add combined attributes to the output file
        for attr_key, attr_values in combined_attrs.items():
            if attr_key == 'total':
                data_group.attrs[attr_key] = attr_values
                continue
            if isinstance(attr_values, str):
                data_group.attrs[attr_key] = attr_values
                continue
            try:
                data_group.attrs[attr_key] = json.dumps(attr_values)
            except TypeError:
                # is_real is numpy bool_ for some reason
                data_group.attrs[attr_key] = json.dumps([bool(i) for i in attr_values])
        
        data_group.attrs['combined'] = True
        data_group.attrs['dataset_id'] = str(uuid.uuid4())
        
if __name__ == '__main__':
    combine_datasets_command()