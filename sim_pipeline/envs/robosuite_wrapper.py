import gymnasium as gym
# needed b/c diffusion policy uses orig gym
import gym as og_gym
import numpy as np
import robomimic.utils.obs_utils as ObsUtils

from sim_pipeline.configs.env.robomimic_base import RobomimicEnvConfig
from sim_pipeline.utils.obs_dict_tools import flatten_obs_dict
from robomimic.envs.env_robosuite import EnvRobosuite

class RobomimicObsWrapper(EnvRobosuite, gym.Env):
    def __init__(self, *args, env_config=None, flatten_obs=True, gymnasium_api=True, obs_modality_specs=None, seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.env_config: RobomimicEnvConfig = env_config
        if seed:
            np.random.seed(seed)

        self.flatten_obs = flatten_obs
        self.gymnasium_api = gymnasium_api
        
        # we store this value in the class rather than robomimic's global var
        # due to multiprocessing
        self.obs_modality_specs = obs_modality_specs
        if self.obs_modality_specs is not None:
            ObsUtils.initialize_obs_utils_with_obs_specs(self.obs_modality_specs)
        
        if self.env.has_offscreen_renderer and not self.env.has_renderer:
            self.render_mode = 'rgb_array'
        elif self.env.has_renderer:
            self.render_mode = 'human'
            
        # this is auto set as True in robomimic
        self.env.ignore_done = env_config.ignore_done
        
        #TODO hardcoded action dim, can pull from bc dataset perhaps
        if self.flatten_obs:
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
            obs = self.get_observation()
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32)
        elif self.obs_modality_specs is not None:
            observation_space = og_gym.spaces.Dict()
            obs = self.get_observation()
            for key, value in obs.items():
                if key.endswith('image'):
                    low = 0.0
                    high = 1.0
                else:
                    low = -1.0
                    high = 1.0
                observation_space[key] = og_gym.spaces.Box(low=low, high=high, shape=value.shape, dtype=np.float32)
            self.observation_space = observation_space  
            self.action_space = og_gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
                
    @staticmethod
    def init_for_data_processing(camera_names):
        image_modalities = [f'{cn}_image' for cn in camera_names]

        obs_modality_specs = {
            "obs": {
                "low_dim": [], # technically unused, so we don't have to specify all of them
                "rgb": image_modalities,
            }
        }
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)
        
    @staticmethod
    def get_specs_from_obs_keys(lowdim_keys: list[str], rgb_keys: list[str], depth_keys: list[str]):
        obs_modality_specs = {
            "obs": {
                "low_dim": lowdim_keys,
                "rgb": rgb_keys,
                "depth": depth_keys,
            }
        }
        return obs_modality_specs

    def __getattr__(self, name):
        return getattr(self.env, name)
    
    def get_observation(self, di=None) -> np.ndarray:
        if not self.flatten_obs:
            return super().get_observation(di)
        
        obs_dict = self.env._get_observations(force_update=True)
        
        pf = self.get_prefix()
                    
        return flatten_obs_dict(obs_dict, pf, self.env_config)
    
    def get_prefix(self):
        # only supports one robot for now
        for robot in self.env.robots:
            return robot.robot_model.naming_prefix

    def get_success(self):
        return self.env._check_success()
    
    def is_done(self):
        # need to override EnvRobosuite, which auto sets done to False
        return self.env.done
    
    def get_camera_sizes(self):
        return self.env.camera_heights, self.env.camera_widths
    
    def render(self, mode=None, height=None, width=None, camera_name="agentview"):
        if mode is None:
            mode = self.render_mode

        if height is None:
            height = self.env.camera_heights[0]
        if width is None:
            width = self.env.camera_widths[0]

        return super().render(mode=mode, height=height, width=width, camera_name=camera_name)
    
    def step(self, action, *args, **kwargs):
        obs, rew, done, info = super().step(action, *args, **kwargs)
        
        if self.gymnasium_api:
            return obs, rew, done, False, info 
        return obs, rew, done, info

    def reset(self, seed=None):
        obs = super().reset()
        if self.gymnasium_api:
            return obs, None
        return obs