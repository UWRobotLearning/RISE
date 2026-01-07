import numpy as np
import torch
import cv2
from pynput import keyboard 

from collections import deque
from sim_pipeline.eval.rollout_env import PossibleVecEnv
from sim_pipeline.eval.rollout_policies.rollout_policy import RolloutPolicy
from sim_pipeline.eval.rollout_stats import RolloutStats
from sim_pipeline.utils.logger import Video
from sim_pipeline.utils.logger import Logger
from sim_pipeline.utils.rollout_utils import _render_frame, stack_last_n_obs, add_batch_dim


def rollout(
        policy: RolloutPolicy, 
        eval_envs: PossibleVecEnv, 
        rollout_stats: RolloutStats,
        reset_state: np.ndarray | None = None,
        logger: Logger | None = None, 
        total_steps: int | None = 200, 
        render_mode: str | None = 'rgb_array', 
        log_video: bool = True,
        video_writers: list[cv2.VideoWriter] | None = None,
        render_dims: tuple[int, int] | None = None,
        video_skip: int = 5,
        return_trajectory: bool = False,
        end_on_success: bool = True,
        tag: str = 'eval'
    ) -> None | dict[str, list[np.ndarray | float | bool] | list[list[np.ndarray | float | bool]]]:
    """
    General function for rolling out a policy in an environment.   

    Args:
        policy: supports .predict(action)
        eval_envs: must have gymnasium api, so `step` returns (obs, rew, term, trunc, info) and `reset` returns (obs, info)
        rollout_stats: RolloutStats object to store episode statistics
        logger: supports .record(key, value)
        total_steps : number of steps to rollout. If None, roll out
            until done=True (for all envs) and assumes all environments terminate at the same time.
        render_mode:
            'human' - render to screen. no videos can be saved. `eval_envs` can be
                a single environment, or if vectorized, only the first environment is rendered.
            'rgb_array' - `eval_envs` is a vectorized environment. videos are logged if `log_video` 
                is True.
            None - no rendering
        log_video: whether to save videos of the rollout. If video_writer is not None, writes to video_writer,
            and if logger is not None, logs the video to the logger.
        video_writer: a cv2.VideoWriter object to write the video to. Must have `log_video` set to True to save
            videos.
        render_dims: dimensions to render the video in. If None, uses the default camera size of the environment.
            Ignored if vectorized environment.
        return_trajectory: whether to return the trajectory of the rollout.
        
    Returns:
        if `return_trajectory` is True, returns a dictionary:
            {
                'states': list of states, 
                'actions': list of actions, 
                'rewards': list of rewards, 
                'dones': list of dones
            }
            if `eval_envs` is a vectorized environment, each list is a list of lists, where each inner list
            corresponds to a different env's rollout. Otherwise, each list is a list of the rollout trajectory.
        else, returns None
    """
    assert isinstance(policy, RolloutPolicy)
    
    render = torch.cuda.device_count() > 0 and render_mode is not None
    eval_envs.render_mode = render_mode
    if eval_envs.is_sb_vec_env:
        # not correct
        camera_height = eval_envs.get_camera_sizes()[0][0][0]
    else:
        camera_height = eval_envs.get_camera_sizes()[0][0]
        
    if return_trajectory:
        if eval_envs.is_vec_env:
            trajectory = {
                "states": [[] for _ in range(eval_envs.num_envs)],
                "actions": [[] for _ in range(eval_envs.num_envs)],
                "rewards": [[] for _ in range(eval_envs.num_envs)],
                "dones": [[] for _ in range(eval_envs.num_envs)],
            }
        trajectory = {
            "states": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "obs": [],
        }

    ########################################
    # obs shapes:
    # SB3:
    #   non-vectorized: (D, ) flat
    #   obs: (N, D) flat
    #   policy handles either
    # robomimic:
    #   non-vectorized: dict -> (d,)
    #   vectorized: dict -> (N, d)
    #   -> does not support vectorized eval b/c robomimic policy auto-pads batch dim
    # diffusion:
    #   non-vectorized: dict -> (d, )
    #   vectorized: dict -> (N, d)
    #   policy:
    #       lowdim expects (N, H, D)
    #       rgb expects dict -> (N, H, d)
    # H - obs horizon, D - total obs dim, d - obs key dim, N - num envs
    ########################################
    if reset_state is not None and not eval_envs.is_vec_env:
        eval_envs.reset()
        obs = eval_envs.reset_to({'states': reset_state})
    else:
        obs = eval_envs.reset()
    policy.reset()
    if return_trajectory:
        # doesn't work when eval_envs is VecEnv
        state_dict = eval_envs.get_state()
        state = state_dict['states']
        trajectory['initial_state_dict'] = state_dict
    dones = False
    if render_mode == 'rgb_array' and log_video:
        frames: list[np.ndarray] = []
    if render:
        frame = _render_frame(eval_envs, render_mode, log_video, camera_height, video_writers=video_writers, dims=render_dims)
        
        if render_mode == 'rgb_array' and log_video and video_writers is None:
            frames.append(frame)

    # TODO: Don't assume all eval_envs terminate at the same time. If VecEnv, 
    # steps is not specified, and a env has done = True before others, will error out. 
    step_count = 0
    # for diffusion policies/policies that do action chunking, execute one action at a time
    action_queue = deque()
    if not eval_envs.is_vec_env:
        obs = add_batch_dim(obs)
    if policy.n_obs_steps is not None:
        obs_stack: deque[np.ndarray | dict[str, np.ndarray]] = deque(maxlen=policy.n_obs_steps+1)
        obs_stack.append(obs)
    else:
        obs_stack = None
        
    global should_stop, random_sample
    should_stop = False
    random_sample = False
    def on_press(key):
        global should_stop, random_sample
        try:
            if key.char == 'q':
                should_stop = True
            elif key.char == 'r':
                random_sample = True
        except AttributeError:
            pass
            
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    while (total_steps is None and not np.all(dones)) or (total_steps is not None and step_count < total_steps):
        if should_stop:
            break
        
        obs_input = obs if policy.n_obs_steps is None else stack_last_n_obs(obs_stack, policy.n_obs_steps)
        if policy.action_chunking:
            if not action_queue:
                actions = policy.predict(obs_input, random_sample=random_sample)
                # actions has dim (n_envs, execution_horizon, action_dim)
                for i in range(actions.shape[1]):
                    action_queue.append(actions[:, i])
            actions = action_queue.popleft()
        else:
            # actions has dim (n_envs, action_dim)
            actions = policy.predict(obs_input, random_sample=random_sample)
            
        if policy.is_composite:
            switched = policy.switch_policy(obs, actions, eval_envs)
            # empty action queue b/c the new policy might not have action chunking
            if switched:
                action_queue.clear()
            
        if not eval_envs.is_vec_env:
            # for non vectorized envs, remove batch dim
            actions = actions[0]
        next_obs, rewards, dones, truncated, infos = eval_envs.step(actions)
        
        if not eval_envs.is_vec_env:
            next_obs = add_batch_dim(next_obs)
        if obs_stack is not None:
            obs_stack.append(next_obs)
        
        success: list[bool] | bool = eval_envs.get_success()
            
        rollout_stats.add_step(
            reward=rewards,
            success=success,
        )
        
        if return_trajectory:
            if eval_envs.is_vec_env:
                for i in range(eval_envs.num_envs):
                    trajectory['states'][i].append(state[i])
                    trajectory['actions'][i].append(actions[i])
                    trajectory['rewards'][i].append(rewards[i])
                    trajectory['dones'][i].append(dones[i])
            else:
                trajectory['states'].append(state)
                trajectory['actions'].append(actions)
                trajectory['rewards'].append(rewards)
                trajectory['dones'].append(dones)
                trajectory['obs'].append(obs)
            state = eval_envs.get_state()['states']

        if render:
            if step_count % video_skip == 0:
                frame = _render_frame(eval_envs, render_mode, log_video, camera_height, video_writers=video_writers, dims=render_dims)
                
                if render_mode == 'rgb_array' and log_video and video_writers is None:
                    frames.append(frame)
            
        if end_on_success:
            if isinstance(success, list | tuple) and all(success):
                break
            elif isinstance(success, bool) and success:
                break
                
        obs = next_obs
        step_count += 1

    # Record episode statistics
    rollout_stats.finalize_episode()
    if logger is not None:
        logger.record(f"{tag}/batch_avg_return", rollout_stats.avg_ep_reward)
        logger.record(f"{tag}/success", rollout_stats.avg_ep_success)
        logger.record(f"{tag}/final_success", rollout_stats.avg_ep_final_success)

    # Log videos
    if render and render_mode == 'rgb_array' and log_video:        
        if logger is not None:
            # -> (b, t, 3, h, w) + downsample
            video: np.ndarray[np.uint8] = np.stack(frames).transpose(1, 0, 4, 2, 3)[..., ::2, ::2]
            logger.record(
                f"{tag}/trajectory/env/camera",
                Video(video, fps=20),
                exclude=["stdout"],
            )
            
    if return_trajectory:
        return trajectory