import h5py
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.io as pio

def load_dataset(dataset_name_with_full_path):

    # Not sure why, but you have to initialize the ObsUtils with a "fake" observation spec.
    # Note, we don't actually perform the "regenerate observations from dataset" routine
    # So we are sticking with the observations that are already in the dataset, rather than generating new ones
    dummy_spec = dict(
            obs=dict(
                    low_dim=["robot0_eef_pos"],
                    rgb=[],
                ),
        )
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)
    dataset = h5py.File(dataset_name_with_full_path, "r")

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_name_with_full_path)
    env = EnvUtils.create_env_from_metadata(
         env_meta=env_meta,
         render=True,
         render_offscreen=False,
         use_image_obs=False,
    )

    # == Get some information about what's inside the dataset ===

    # print the environment meta data 
    print("\n===Environment Metadata===\n")
    print(env_meta)

    # reorder and print the list of individual trajectory demonstrations 
    print("\n===Demos List===\n")
    demos = list(dataset["data"].keys())
    idxs = np.argsort([int(elem[5:]) for elem in demos])
    demo_idxs = [demos[i] for i in idxs]
    print(demo_idxs)


    # Remind yourself what observations are available 
    print("\n===Available Observations:===\n")
    print(dataset['data']["demo_0"]['obs'].keys())


    # Returns a constructed env, the meta_data for the env, a list of demo names in order, and  the full dataset
    return env, env_meta, demo_idxs, dataset

def visualize_eepos_trajectories_iteratively(dataset, demo_idxs):
    for demo_idx in demo_idxs:
        ee_positions =  dataset['data'][demo_idx]['obs']['robot0_eef_pos']

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(ee_positions[:,0], ee_positions[:,1], ee_positions[:,2], label='End Effector Trajectory')
        plt.show()

def visualize_objectpos_trajectories_iteratively(dataset, demo_idxs):
    for demo_idx in demo_idxs:
        # Note, the 14 dimensional object obs vector is: [world pos, world quat, ee_frame_pos, ee_frame_quat]
        object_positions =  dataset['data'][demo_idx]['obs']['object'][:,:3] # first 3 are world pos

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(object_positions[:,0], object_positions[:,1], object_positions[:,2], label='Object Trajectory')
        plt.show()

def visualize_plotly_objectpos_trajectories_iteratively(dataset, demo_idxs):
    fig = go.Figure()

    for demo_idx in demo_idxs:
        # Note, the 14 dimensional object obs vector is: [world pos, world quat, ee_frame_pos, ee_frame_quat]
        object_positions =  dataset['data'][demo_idx]['obs']['object'][:,:3] # first 3 are world pos
        fig.data = []  # Clear the existing data
        fig.add_trace(go.Scatter3d(
            x=object_positions[:,0],
            y=object_positions[:,1],
            z=object_positions[:,2],
            mode='markers+lines',
            marker=dict(
                size=5,
                color=np.arange(len(object_positions)),
                colorscale='Viridis',   # Choose a colorscale
                opacity=0.8
            ),
            hovertext=np.arange(len(object_positions)),
        ))

        # Update layout details
        fig.update_layout(
            title=f'{demo_idx} object traj',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
    
        # Display the figure
        pio.show(fig)
        input('press enter for next')

def visualize_eepos_trajectories_overlayed(dataset, demo_idxs, demo_idx_range=(0,10)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(*demo_idx_range):    
        traj = dataset['data'][demo_idxs[i]]['obs']['robot0_eef_pos']

        #cmap_rbg = LinearSegmentedColormap.from_list('rbg_grad', [(1,0,0), (0,0,1), (0,1,0)] , N=len(traj))
        #orp = [tuple((1/256)*e for e in t) for t in [(121,0,178), (255,0,71), (255,95,19)]]
        #cmap_orp = LinearSegmentedColormap.from_list('orp_grad', orp , N=len(traj))
        
        colormap_dict = {
            0: 'Blues',
            1: 'BuGn',
            2: 'BuPu',
            3: 'GnBu',
            4: 'Greens',
            5: 'Greys',
            6: 'Oranges',
            7: 'OrRd',
            8: 'PuBu',
            9: 'PuBuGn',
            10: 'PuRd',
            11: 'Purples',
            12: 'RdPu',
            13: 'Reds',
            14: 'YlGn',
            15: 'YlGnBu',
            16: 'YlOrBr',
            17: 'YlOrRd'
        }

        # Use modulo operation to ensure the index is within the bounds of the colormap_dict
        cmap_selected = colormap_dict.get(i % len(colormap_dict), 'viridis')

        ax.scatter3D(traj[:,0],traj[:,1],traj[:,2], c=np.arange(len(traj)), cmap=cmap_selected)

    plt.show()

def run():
    #dataset_name = "/home/share_act/WRK/robomimic/datasets/square/ph/low_dim_v141.hdf5"
    dataset_name = "/home/kevin/ood/sim_pipeline/sim_pipeline/data/combined/push_manual_combined_tl_rew.hdf5"
    env, env_meta, demo_idxs, dataset = load_dataset(dataset_name)

    # Step the Env once, just for fun
    test_action = np.zeros(env.action_dimension)
    env.step(test_action)

    #flattened_dataset = np.concatenate([v['obs']['robot0_eef_pos'][()] for v in dataset['data'].values()])

    #visualize_plotly_objectpos_trajectories_iteratively(dataset, demo_idxs)
    visualize_eepos_trajectories_overlayed(dataset, demo_idxs, (0,10))

if __name__ == "__main__":
    run()