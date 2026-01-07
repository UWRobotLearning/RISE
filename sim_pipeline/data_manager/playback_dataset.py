"""
A script to visualize dataset trajectories by loading the simulation states
one by one or loading the first state and playing actions back open-loop.
The script can generate videos as well, by rendering simulation frames
during playback. The videos can also be generated using the image observations
in the dataset (this is useful for real-robot datasets) by using the
--use-obs argument.

Example usage below:

    # force simulation states one by one, and render agentview and wrist view cameras to video
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --render_image_names agentview robot0_eye_in_hand \
        --video_path /tmp/playback_dataset.mp4

    # playback the actions in the dataset, and render agentview camera during playback to video
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use-actions --render_image_names agentview \
        --video_path /tmp/playback_dataset_with_actions.mp4

    # use the observations stored in the dataset to render videos of the dataset trajectories
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use-obs --render_image_names agentview_image \
        --video_path /tmp/obs_trajectory.mp4

    # visualize depth observations along with image observations
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use-obs --render_image_names agentview_image \
        --render_depth_names agentview_depth \
        --video_path /tmp/obs_trajectory.mp4

    # visualize initial states in the demonstration data
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --first --render_image_names agentview \
        --video_path /tmp/dataset_task_inits.mp4
"""

import os
import json
import h5py
import click
import imageio
import numpy as np

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.utils.vis_utils import depth_to_rgb
from robomimic.envs.env_base import EnvBase, EnvType


# Define default cameras to use for each env type
DEFAULT_CAMERAS = {
    EnvType.ROBOSUITE_TYPE: ["agentview"],
    EnvType.IG_MOMART_TYPE: ["rgb"],
    EnvType.GYM_TYPE: ValueError("No camera names supported for gym type env!"),
}


def playback_trajectory_with_env(
    env, 
    initial_state, 
    states, 
    actions=None, 
    render=False, 
    video_writer=None, 
    video_skip=5, 
    camera_names=None,
    first=False,
):
    """
    Helper function to playback a single trajectory using the simulator environment.
    If @actions are not None, it will play them open-loop after loading the initial state. 
    Otherwise, @states are loaded one by one.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load
        actions (np.array): if provided, play actions back open-loop instead of using @states
        render (bool): if True, render on-screen
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    assert isinstance(env, EnvBase)

    write_video = (video_writer is not None)
    video_count = 0
    assert not (render and write_video)

    # load the initial state
    env.reset()
    env.reset_to(initial_state)

    traj_len = states.shape[0]
    action_playback = (actions is not None)
    if action_playback:
        assert states.shape[0] == actions.shape[0]

    for i in range(traj_len):
        if action_playback:
            env.step(actions[i])
            if i < traj_len - 1:
                # check whether the actions deterministically lead to the same recorded states
                state_playback = env.get_state()["states"]
                if not np.all(np.equal(states[i + 1], state_playback)):
                    err = np.linalg.norm(states[i + 1] - state_playback)
                    print("warning: playback diverged by {} at step {}".format(err, i))
        else:
            env.reset_to({"states" : states[i]})
            
        # on-screen render
        if render:
            env.render(mode="human", camera_name=camera_names[0])
            
        # video render
        if write_video:
            if video_count % video_skip == 0:
                video_img = []
                for cam_name in camera_names:
                    video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                video_writer.append_data(video_img)
            video_count += 1

        if first:
            break


def playback_trajectory_with_obs(
    traj_grp,
    video_writer, 
    video_skip=5, 
    image_names=None,
    depth_names=None,
    first=False,
):
    """
    This function reads all "rgb" (and possibly "depth") observations in the dataset trajectory and
    writes them into a video.

    Args:
        traj_grp (hdf5 file group): hdf5 group which corresponds to the dataset trajectory to playback
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        image_names (list): determines which image observations are used for rendering. Pass more than
            one to output a video with multiple image observations concatenated horizontally.
        depth_names (list): determines which depth observations are used for rendering (if any).
        first (bool): if True, only use the first frame of each episode.
    """
    assert image_names is not None, "error: must specify at least one image observation to use in @image_names"
    video_count = 0

    if depth_names is not None:
        # compute min and max depth value across trajectory for normalization
        depth_min = { k : traj_grp["obs/{}".format(k)][:].min() for k in depth_names }
        depth_max = { k : traj_grp["obs/{}".format(k)][:].max() for k in depth_names }

    traj_len = traj_grp["actions"].shape[0]
    for i in range(traj_len):
        if video_count % video_skip == 0:
            # concatenate image obs together
            im = [traj_grp["obs/{}".format(k)][i] for k in image_names]
            depth = [depth_to_rgb(traj_grp["obs/{}".format(k)][i], depth_min=depth_min[k], depth_max=depth_max[k]) for k in depth_names] if depth_names is not None else []
            frame = np.concatenate(im + depth, axis=1)
            video_writer.append_data(frame)
        video_count += 1

        if first:
            break


@click.command()
@click.option('--dataset', type=str, help='Path to hdf5 dataset', required=True)
@click.option('--filter_key', type=str, default=None, help='Filter key to select subset of trajectories')
@click.option('--n', type=int, default=None, help='Stop after n trajectories')
@click.option('--use-obs', is_flag=True, help='Visualize trajectories with dataset image observations')
@click.option('--use-actions', is_flag=True, help='Use open-loop action playback')
@click.option('--render', is_flag=True, help='On-screen rendering')
@click.option('--video_path', type=str, default=None, help='Path to save video')
@click.option('--video_skip', type=int, default=5, help='Render frames to video every n steps')
@click.option('--render_image_names', type=str, multiple=True, default=None, help='Camera names/image observations for rendering')
@click.option('--render_depth_names', type=str, multiple=True, default=None, help='Depth observations for rendering')
@click.option('--first', is_flag=True, help='Use first frame of each episode')
def playback_dataset(dataset, filter_key, n, use_obs, use_actions, render, video_path, video_skip, render_image_names, render_depth_names, first):
    # some arg checking
    write_video = (video_path is not None)
    assert not (render and write_video) # either on-screen or video but not both

    # Auto-fill camera rendering info if not specified
    if not render_image_names:
        # We fill in the automatic values
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset)
        env_type = EnvUtils.get_env_type(env_meta=env_meta)
        render_image_names = DEFAULT_CAMERAS[env_type]

    if render:
        # on-screen rendering can only support one camera
        assert len(render_image_names) == 1

    if use_obs:
        assert write_video, "playback with observations can only write to video"
        assert not use_actions, "playback with observations is offline and does not support action playback"

    if render_depth_names:
        assert use_obs, "depth observations can only be visualized from observations currently"

    # create environment only if not playing back with observations
    if not use_obs:
        # need to make sure ObsUtils knows which observations are images, but it doesn't matter 
        # for playback since observations are unused. Pass a dummy spec here.
        dummy_spec = dict(
            obs=dict(
                    low_dim=["robot0_eef_pos"],
                    rgb=[],
                ),
        )
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset)
        env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=render, render_offscreen=write_video)

        # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
        is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    f = h5py.File(dataset, "r")

    # list of all demonstration episodes (sorted in increasing number order)
    if filter_key is not None:
        print("using filter key: {}".format(filter_key))
        demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(filter_key)])]
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if n is not None:
        demos = demos[:n]

    # maybe dump video
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(video_path, fps=20)

    for ind in range(len(demos)):
        ep = demos[ind]
        print("Playing back episode: {}".format(ep))

        if use_obs:
            playback_trajectory_with_obs(
                traj_grp=f["data/{}".format(ep)], 
                video_writer=video_writer, 
                video_skip=video_skip,
                image_names=render_image_names,
                depth_names=render_depth_names,
                first=first,
            )
            continue

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

        # supply actions if using open-loop action playback
        actions = None
        if use_actions:
            actions = f["data/{}/actions".format(ep)][()]

        playback_trajectory_with_env(
            env=env, 
            initial_state=initial_state, 
            states=states, actions=actions, 
            render=render, 
            video_writer=video_writer, 
            video_skip=video_skip,
            camera_names=render_image_names,
            first=first,
        )

    f.close()
    if write_video:
        video_writer.close()


if __name__ == "__main__":
    playback_dataset()
