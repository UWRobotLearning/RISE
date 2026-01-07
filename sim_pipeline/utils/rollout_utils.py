import h5py
import cv2
import numpy as np

from collections import deque

from sim_pipeline.eval.rollout_env import PossibleVecEnv


def _render_frame(env: PossibleVecEnv, render_mode: str, log_video: bool, default_camera_height: int, video_writers: list[cv2.VideoWriter] | None, dims: tuple[int, int] | None = None) -> np.ndarray:
    if render_mode == 'human' and env.is_vec_env:
        raise ValueError("Cannot render human mode for vectorized environments")
    # for robomimic: frame is (w, h, c), 0-255
    # for sb3 VecEnv: frame is (w * num_envs, h, c)
    # for gym VectorEnv, frame is (num_envs, w, h, c), where outer dim is list
    if env.is_sb_vec_env or dims is None:
        frame: np.ndarray[np.uint8] = env.render(mode=render_mode)
    else:
        frame = env.render(mode=render_mode, height=dims[0], width=dims[1])
    if render_mode == 'rgb_array' and log_video:
        # each frame has shape (num_envs, w, h, c)
        height = default_camera_height if dims is None else dims[0]
        if env.is_gym_vec_env:
            frame = np.array(frame)
        elif env.is_sb_vec_env:
            frame = np.array(np.split(frame, frame.shape[0] // height))
        else:
            frame = np.array([frame])

        if video_writers is not None:
            num_envs = env.num_envs if env.is_vec_env else 1
            for i in range(num_envs):
                video_frame = frame[i]
                video_writer = video_writers[i]
                video_frame = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR)
                video_writer.write(video_frame)
    return frame


def add_trajectory_to_dataset(
        ep_number: int,
        trajectory: dict[str, list[np.ndarray | float | bool] | list[list[np.ndarray | float | bool]]],
        dataset_file: h5py.File,
    ) -> tuple[int, int]:

    ep_number_after = ep_number + 1
    added_samples = 0

    data_grp = dataset_file['data']

    if isinstance(trajectory['states'][0], list):
        for i in range(len(trajectory['states'])):
            ep_data_grp = data_grp.create_group(f"demo_{ep_number + i}")

            ep_data_grp.create_dataset('states', data=np.array(trajectory['states'][i]))
            ep_data_grp.create_dataset('actions', data=np.array(trajectory['actions'][i]))
            ep_data_grp.create_dataset('rewards', data=np.array(trajectory['rewards'][i]))
            ep_data_grp.create_dataset('dones', data=np.array(trajectory['dones'][i]))

            ep_data_grp.attrs['num_samples'] = len(trajectory['actions'][i])
            ep_number_after = ep_number + i + 1
            added_samples += len(trajectory['actions'][i])
    else:
        ep_data_grp = data_grp.create_group(f"demo_{ep_number}")
        ep_data_grp.create_dataset('states', data=np.array(trajectory['states']))
        ep_data_grp.create_dataset('actions', data=np.array(trajectory['actions']))
        ep_data_grp.create_dataset('rewards', data=np.array(trajectory['rewards']))
        ep_data_grp.create_dataset('dones', data=np.array(trajectory['dones']))

        if 'model' in trajectory['initial_state_dict']:
            ep_data_grp.attrs['model_file'] = trajectory['initial_state_dict']['model']

        n_samples = len(trajectory['actions'])
        ep_data_grp.attrs['num_samples'] = n_samples
        added_samples += n_samples

    return ep_number_after, added_samples


def add_batch_dim(obs: np.ndarray | dict[str, np.ndarray]) -> np.ndarray | dict[str, np.ndarray]:
    if isinstance(obs, dict):
        return {key: np.expand_dims(val, 0) for key, val in obs.items()}
    return np.expand_dims(obs, 0)


def _get_and_pad_n_obs(all_obs: list[np.ndarray], n_obs_steps: int) -> np.ndarray:
    result = np.zeros((n_obs_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
    
    start_idx = -min(n_obs_steps, len(all_obs))
    result[start_idx:] = np.array(all_obs[start_idx:])
    
    # if all_obs is not long enough to fill result
    if n_obs_steps > len(all_obs):
        # pad with the earliest observation
        result[:start_idx] = result[start_idx]
        
    result = np.swapaxes(result, 0, 1)
    return result


def stack_last_n_obs(all_obs: deque[np.ndarray | dict[str, np.ndarray]], n_obs_steps: int) -> np.ndarray | dict[str, np.ndarray]:
    """
    assuming diffusion, each obs has shape (n_envs, d)
    
    need to stack the last n_obs_steps observations in middle dim, so we output shape (n_envs, n_obs_steps, d)
    """
    if len(all_obs) == 0:
        raise ValueError("observation history is empty")

    all_obs = list(all_obs)

    if isinstance(all_obs[0], dict):
        keys = all_obs[0].keys()
        return {key: _get_and_pad_n_obs([obs[key] for obs in all_obs], n_obs_steps) for key in keys}
        
    return _get_and_pad_n_obs(all_obs, n_obs_steps)


def filter_obs_keys(obs: dict[str, np.ndarray] | np.ndarray, obs_keys: list[str] | None) -> dict[str, np.ndarray] | np.ndarray:
    if isinstance(obs, dict) and obs_keys is not None:
        return {key: obs[key] for key in obs_keys}
    return obs


def merge_obs_specs(obs_specs: list[dict[str, list[str]]]):
    """Recursively merge Robomimic obs specs, concatenating values with the same keys.

    Args:
        obs_specs (list[dict[str, list[str]]]): list of obs specs to merge
    """
    
    def merge_recursive(d1, d2):
        for key, value in d2.items():
            if isinstance(value, dict):
                d1[key] = merge_recursive(d1.get(key, {}), value)
            elif isinstance(value, list):
                if key in d1:
                    d1[key] = list(set(d1[key] + value))
                else:
                    d1[key] = value
            else:
                raise ValueError(f"Unexpected value type {type(value)} in dict {d2}")
        return d1
    
    result = {}
    for d in obs_specs:
        result = merge_recursive(result, d)
    return result
