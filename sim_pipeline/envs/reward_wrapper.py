from typing import Any
import gymnasium as gym

class RewardWrapper(gym.Env):
    def __init__(self, env, reward_func=None, success_func=None, gymnasium_api=True):
        self.env = env
        self.reward_func = reward_func if reward_func else self.default_reward
        self.success_func = success_func if success_func else self.default_success
        self.gymnasium_api = gymnasium_api

    def __getattribute__(self, name: str) -> Any:
        """
        Ignores subclass methods. Needed to avoid SB3's requirement of needing to inherit from gym.Env
        """
        instance_dict = object.__getattribute__(self, '__dict__')
        if name in RewardWrapper.__dict__ or name in instance_dict:
            return object.__getattribute__(self, name)
        
        env = object.__getattribute__(self, 'env')
        return getattr(env, name)
        
    def step(self, action):
        step_results = self.env.step(action)
        try:
            obs, reward, done, info = step_results
            truncated = False
        except ValueError:
            obs, reward, done, truncated, info = step_results
        if self.gymnasium_api:
            return obs, self.reward_func(obs, action, self.env), done, truncated, info
        return obs, self.reward_func(obs, action, self.env), done, info
    
    def get_success(self):
        obs = self.env.get_observation()
        return self.success_func(obs, self.env)

    def get_reward(self):
        obs = self.env.get_observation()
        return self.reward_func(obs, None, self.env)

    def default_reward(self, obs, act, env):
        return 0.0
    
    def default_success(self, obs, env):
        return self.env.get_success()