import h5py
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecEnv
from sim_pipeline.utils.obs_dict_tools import flatten_obs_dict
from sim_pipeline.configs.env.robomimic_base import RobomimicEnvConfig
from sim_pipeline.envs.robosuite_wrapper import RobomimicObsWrapper

def init_buffer(buffer: ReplayBuffer, dataset_path: str, env: RobomimicObsWrapper | VecEnv, env_config: RobomimicEnvConfig | None):
    """
    Given a ReplayBuffer, loads a dataset from a given path and adds the data to the buffer.
    """
    dataset = h5py.File(dataset_path, 'r')
    
    episode_names: list[str] = list(dataset['data'].keys())
    n_envs: int = buffer.n_envs
    
    # to add to sb3 buffer, needs to be in chunks of n_envs, so there will be leftovers.
    # try to group together all leftovers from all episodes so that as little data is wasted as possible
    leftover_obs = []
    leftover_next_obs = []
    leftover_actions = []
    leftover_rewards = []
    leftover_dones = []
    for ep_name in episode_names:
        episode = dataset['data'][ep_name]
        
        obs_dict = episode['obs']
        next_obs_dict = episode['next_obs']
        if isinstance(env, VecEnv):
            robot_prefix = env.env_method('get_prefix')[0]
        else:
            robot_prefix = env.get_prefix()
        # Shape (ep_len, obs_dim)
        obs = flatten_obs_dict(obs_dict, robot_prefix, env_config)
        next_obs = flatten_obs_dict(next_obs_dict, robot_prefix, env_config)
        # Shape (ep_len, act_dim)
        actions = episode['actions']
        # Shape (ep_len, )
        rewards = episode['rewards']
        dones = episode['dones']
                
        # convert to (n_env, dim) chunks        
        ep_len = obs.shape[0]
        n_chunks = ep_len // n_envs
        leftover = ep_len % n_envs
                
        if leftover > 0:
            obs_chunks = np.split(obs[:-leftover], n_chunks)
            next_obs_chunks = np.split(next_obs[:-leftover], n_chunks)
            actions_chunks = np.split(actions[:-leftover], n_chunks)
            rewards_chunks = np.split(rewards[:-leftover], n_chunks)
            dones_chunks = np.split(dones[:-leftover], n_chunks)
            
            leftover_obs.append(obs[-leftover:])
            leftover_next_obs.append(next_obs[-leftover:])
            leftover_actions.append(actions[-leftover:])
            leftover_rewards.append(rewards[-leftover:])
            leftover_dones.append(dones[-leftover:])
        else:
            obs_chunks = np.split(obs, n_chunks)
            next_obs_chunks = np.split(next_obs, n_chunks)
            actions_chunks = np.split(actions, n_chunks)
            rewards_chunks = np.split(rewards, n_chunks)
            dones_chunks = np.split(dones, n_chunks)
                    
        infos = [{} for _ in range(n_envs)]
        
        for i in range(len(obs_chunks)):
            buffer.add(
                obs_chunks[i], 
                next_obs_chunks[i], 
                actions_chunks[i], 
                rewards_chunks[i], 
                dones_chunks[i], 
                infos
            )
    
    # Deal with leftovers
    if leftover_obs:
        leftover_obs = np.concatenate(leftover_obs, axis=0)
        leftover_next_obs = np.concatenate(leftover_next_obs, axis=0)
        leftover_actions = np.concatenate(leftover_actions, axis=0)
        leftover_rewards = np.concatenate(leftover_rewards, axis=0)
        leftover_dones = np.concatenate(leftover_dones, axis=0)
        
        num_leftover = leftover_obs.shape[0]
        n_chunks = num_leftover // n_envs
        leftover = num_leftover % n_envs
        
        if leftover > 0:
            obs_chunks = np.split(leftover_obs[:-leftover], n_chunks)
            next_obs_chunks = np.split(leftover_next_obs[:-leftover], n_chunks)
            actions_chunks = np.split(leftover_actions[:-leftover], n_chunks)
            rewards_chunks = np.split(leftover_rewards[:-leftover], n_chunks)
            dones_chunks = np.split(leftover_dones[:-leftover], n_chunks)
        else:
            obs_chunks = np.split(leftover_obs, n_chunks)
            next_obs_chunks = np.split(leftover_next_obs, n_chunks)
            actions_chunks = np.split(leftover_actions, n_chunks)
            rewards_chunks = np.split(leftover_rewards, n_chunks)
            dones_chunks = np.split(leftover_dones, n_chunks)
            
        infos = [{} for _ in range(n_envs)]
        
        for i in range(len(obs_chunks)):
            buffer.add(
                obs_chunks[i], 
                next_obs_chunks[i], 
                actions_chunks[i], 
                rewards_chunks[i], 
                dones_chunks[i], 
                infos
            )
        
        print(f'Wasted {leftover} samples')
    print('Wasted no samples!')