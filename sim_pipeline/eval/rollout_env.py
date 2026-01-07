from functools import partial

from gym.vector.vector_env import VectorEnv
from stable_baselines3.common.vec_env import VecEnv


class PossibleVecEnv:
    def __init__(self, env):
        """ 
        Wrapper to easily handle both single and sb3 vectorized environments for rollouts.

        env must have gymnasium format, so `step` returns (obs, rew, term, trunc, info)
        and `reset` returns (obs, info)
        """
        self.env = env
        self.is_sb_vec_env = isinstance(env, VecEnv)
        self.is_gym_vec_env = isinstance(env, VectorEnv)
        self.is_vec_env = self.is_sb_vec_env or self.is_gym_vec_env

    def __getattr__(self, name):
        if self.is_sb_vec_env:
            try:
                env_method = getattr(self.env, name)
            except AttributeError:
                env_method = partial(self.env.env_method, name)
            return env_method
        elif self.is_gym_vec_env:
            try:
                env_method = getattr(self.env, name)
            except AttributeError:
                env_method = partial(self.env.call, name)
            return env_method
        return getattr(self.env, name)

    def get_success(self) -> list[bool] | bool:
        if self.is_vec_env:
            try:
                return self.__getattr__('get_success')()
            except AttributeError:
                return [False] * self.env.num_envs
        try:
            # if defined, i.e. for robomimic envs
            return self.env.get_success()
        except AttributeError:
            return [False]

    def step(self, actions):
        # SB3 is stupid and VecEnv expects the wrapped env to have truncated,
        # but the actual VecEnv doesn't return truncated
        # gym VectorEnv needs to have gym api due to being inherited from diffusion policy's
        # implementation, which uses gym api
        if self.is_sb_vec_env or self.is_gym_vec_env:
            next_obs, rew, dones, infos = self.env.step(actions)
            return next_obs, rew, dones, None, infos
        return self.env.step(actions)

    def reset(self):
        # likewise, SB3 is stupid, and VecEnv expects the wrapped env's reset to return
        # a tuple (obs, info), but the actual VecEnv doesn't return info
        if self.is_sb_vec_env or self.is_gym_vec_env:
            return self.env.reset()
        return self.env.reset()[0]