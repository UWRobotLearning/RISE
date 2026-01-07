import hydra
import h5py
import tempfile
import subprocess

from sim_pipeline.configs._data_summary.data_summary import DataSummaryConfig
from sim_pipeline.data_manager.get_dataset import get_dataset



@hydra.main(config_path="../sim_pipeline/configs", config_name="data_summary", version_base=None)
def main(data_summary_cfg: DataSummaryConfig):
    data_cfg = data_summary_cfg.data
    dataset_path = get_dataset(data_cfg)

    edit_dataset_description(dataset_path)
    
def edit_dataset_description(dataset_path: str):
    with h5py.File(dataset_path, 'r+') as f:
        data = f['data']
        if 'description' not in data.attrs:
            data.attrs['description'] = '<please enter dataset description here>'
            
        with tempfile.NamedTemporaryFile(suffix='.txt', mode='w+') as tmp:
            tmp.write(data.attrs['description'])
            tmp.flush()
            
            subprocess.call(['vim', '+1', '-c set filetype=txt', tmp.name])
            
            with open(tmp.name, 'r') as tmp_f:
                data.attrs['description'] = tmp_f.read()