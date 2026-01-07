import click
import torch
import os
import h5py
import uuid
import numpy as np

from tqdm import tqdm
from contextlib import nullcontext
from robomimic.scripts.split_train_val import split_train_val_from_hdf5
from robomimic.models.obs_core import DinoV2Core

def relabel_key(ep, out_data_grp, key, feature_model, batch_size, model_device):
    obs = ep[key]
    obs_dict = {k: obs[k][:] for k in obs.keys() if 'image' in k}

    for k in obs_dict.keys():
        obs_dict[k] = obs_dict[k].transpose(0, 3, 2, 1)
        obs_dict[k] = obs_dict[k].astype(float) / 255.0

    ep_features = []
    for image_key in sorted(obs_dict.keys()):
        all_img_features = []
        all_imgs = obs_dict[image_key]
        for index in tqdm(range(0, all_imgs.shape[0], batch_size), desc='processing batch', leave=False):
            img_batch = all_imgs[index:min(index+batch_size, all_imgs.shape[0])]
            img_batch = torch.from_numpy(img_batch).to(model_device).float()
            img_batch_features = feature_model(img_batch).detach().cpu().numpy()
            all_img_features.append(img_batch_features)
        all_img_features = np.concatenate(all_img_features, axis=0)
        ep_features.append(all_img_features)
    ep_features = np.concatenate(ep_features, axis=1)

    feature_shape = ep_features.shape[1]
    out_data_grp[key].create_dataset('dino_features', shape=(ep['actions'].shape[0], feature_shape), dtype=np.float32)
    out_data_grp[key]['dino_features'][:] = ep_features
    for image_key in obs_dict:
        del out_data_grp[key][image_key]

@click.command()
@click.option('--dataset', '-d', help='Datset file path')
@click.option('--output_name', '-o', help='Output file name')
@click.option('--batch_size', '-b', default=4, type=int, help='Batch size for feature model')
def relabel_dino_features(dataset, output_name, batch_size):
    print(f'Relabeling images with dino features in dataset {dataset}')
    # output file in same directory as input file
    output_path = os.path.join(os.path.dirname(dataset), output_name)
    
    feature_model = DinoV2Core(input_shape=(3, 84, 84), backbone_name='dinov2_vits14', concatenate=True)
    model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_model.to(model_device)
    feature_model.eval()

    with (
        h5py.File(dataset, 'r') as f,
        h5py.File(output_path, 'w') as f_relabeled,
    ):  
        data = f['data']
        out_data = f_relabeled.create_group("data")
                
        demo_counter = 0
        for ep_key in tqdm(data, desc='processing episodes'):
            ep = data[ep_key]
            
            data.copy(
                source=ep,
                dest=out_data,
                name=f'demo_{demo_counter}'
            )
            out_data_grp = out_data[f'demo_{demo_counter}']
            
            relabel_key(ep, out_data_grp, 'obs', feature_model, batch_size, model_device)
            relabel_key(ep, out_data_grp, 'next_obs', feature_model, batch_size, model_device)
            
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
    relabel_dino_features()