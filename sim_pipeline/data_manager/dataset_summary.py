import h5py
import hydra
import numpy as np
from sim_pipeline.data_manager.dataset_metadata_enums import *
from sim_pipeline.configs._data_summary.data_summary import DataSummaryConfig
from sim_pipeline.data_manager.get_dataset import get_dataset

@hydra.main(config_path="../configs", config_name="data_summary", version_base=None)
def dataset_summary(data_summary_cfg: DataSummaryConfig):
    data_cfg = data_summary_cfg.data
    dataset_path = get_dataset(data_cfg)
    get_dataset_summary(dataset_path)

def get_dataset_summary(file_path: str):
    entropies = []
    with h5py.File(file_path, 'r') as f:
        data = f['data']
        attrs = data.attrs
        
        ep_keys = list(data.keys())
        
        traj_lengths = []
        action_min = np.inf
        action_max = -np.inf
        for ep in ep_keys:
            traj_lengths.append(data[ep]['actions'].shape[0])
            action_min = min(action_min, np.min(data[ep]['actions'][()]))
            action_max = max(action_max, np.max(data[ep]['actions'][()]))
        traj_lengths = np.array(traj_lengths)
        
        if 'obs' in data[ep_keys[0]]:
            processed_status = ProcessedStatus.PROCESSED
        elif 'mask' not in f:
            processed_status = ProcessedStatus.RAW_TELEOP
        else:
            processed_status = ProcessedStatus.PRE_PROCESSED
            
        print(f'FILE: {file_path}')

        for ep in ep_keys:
            if 'entropy' in data[ep]:
                entropies.append(data[ep]['entropy'][()])
        
        import ipdb; ipdb.set_trace()
                                    
        print('==== Dataset Metadata ====')
        print(f'ID: {attrs["dataset_id"]}')
        print('Processed Status:', processed_status)
        print(f'env_name: {attrs["env_name"]}')
        print(f'env_type: {attrs["env_type"]}')
        print(f'data_type: {attrs["dataset_type"]}')
        print(f'action_type: {attrs["action_type"]}')
        if processed_status == ProcessedStatus.PROCESSED:
            print(f'reward_type: {attrs["reward_type"]}')
        print(f'is_real: {attrs["is_real"]}')
        if 'combined' in attrs and attrs['combined']:
            print(f'This is a combined dataset with constituents{attrs["constituent_dataset_id"]}')
        if 'date' in attrs:
            print(f'creation date: {attrs["date"]}')
        if 'creator' in attrs:
            print(f'creator: {attrs["creator"]}')

        print("==== Trajectory Summary ====")
        print(f"total transitions: {np.sum(traj_lengths)}")
        print(f"total trajectories: {traj_lengths.shape[0]}")
        print(f"traj length mean: {np.mean(traj_lengths)}")
        print(f"traj length std: {np.std(traj_lengths)}")
        print(f"traj length min: {np.min(traj_lengths)}")
        print(f"traj length max: {np.max(traj_lengths)}")
        print(f"action min: {action_min}")
        print(f"action max: {action_max}")
        print("")

        # traj length processing
        num_flagged = 0
        flagged = []
        for i in range(len(traj_lengths)):
            if traj_lengths[i] > 1.5 * np.std(traj_lengths) + np.mean(traj_lengths):
                num_flagged += 1
                print(i, traj_lengths[i])
                flagged.append(i)
        print("total flagged", num_flagged)
        print(flagged)

        if processed_status != ProcessedStatus.PROCESSED:
            print('Dataset is not processed, and thus does not have observations')
            return 
        
        print("==== Dataset Structure ====")
        for ep in ep_keys:
            print(f"episode {ep} with {data[ep].attrs['num_samples']} transitions")
            for k in data[ep]:
                if k in ["obs", "next_obs"]:
                    print(f"    key: {k}")
                    for obs_k in data[ep][k]:
                        shape = data[ep][k][obs_k].shape
                        print(f"        observation key {obs_k} with shape {shape}")
                elif isinstance(data[ep][k], h5py.Dataset):
                    key_shape = data[ep][k].shape
                    print(f"    key: {k} with shape {key_shape}")
            break
        
        if 'description' in data.attrs:
            print('==== Dataset Description ====')
            print(data.attrs['description'])
            print('')

    # entropies2 = []
    # for entropy in entropies:
    #     entropy = np.array(entropy)
    #     entropy *= 1#1 / (1 + np.exp(-10*(entropy - 0.5)))
    #     entropies2.append(entropy)
    # entropies = entropies2
    # import matplotlib.pyplot as plt
    # if entropies:
    #     max_length = max(len(e) for e in entropies)
    #     avg_entropies = []
    #     std_entropies = []
    #     for t in range(max_length):
    #         valid_entropies = [e[t] for e in entropies if len(e) > t]
    #         if valid_entropies:
    #             avg_entropies.append(np.mean(valid_entropies))
    #             std_entropies.append(np.std(valid_entropies))
    #         else:
    #             avg_entropies.append(np.nan)
    #             std_entropies.append(np.nan)
        
    #     plt.figure()
    #     plt.plot(avg_entropies, label='Average Entropy')
    #     plt.fill_between(range(max_length), 
    #                      np.array(avg_entropies) - np.array(std_entropies), 
    #                      np.array(avg_entropies) + np.array(std_entropies), 
    #                      color='b', alpha=0.2, label='Std Deviation')
    #     plt.xlabel('Timestep')
    #     plt.ylabel('Entropy')
    #     plt.title('Average Entropy Across All Demos')
    #     plt.legend()
    #     plt.savefig('average_entropy.png')
    # import matplotlib.pyplot as plt
    # plt.plot(entropies[6])
    # plt.savefig('entropy.png')
