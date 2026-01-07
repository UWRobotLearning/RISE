import numpy as np

from sim_pipeline.eval.rollout_policies.rollout_policy import RolloutPolicy

class CompositeRolloutPolicy(RolloutPolicy):
    def __init__(self, policies: list[RolloutPolicy], init_policy_index: int = 0, n_envs: int = 1):
        self.policies = policies
        self.n_policies = len(policies)
        self.n_envs = n_envs
        self.init_policy_index = init_policy_index
        self.env_to_policy_idx = {i: self.init_policy_index for i in range(n_envs)}
        
        self.action_chunking_dict = {i: policy.action_chunking for i, policy in enumerate(policies)}
        self.n_obs_steps_dict = {i: policy.n_obs_steps for i, policy in enumerate(policies)}
        self.action_chunking = self.action_chunking_dict[self.init_policy_index]
        self.n_obs_steps = self.n_obs_steps_dict[self.init_policy_index]
        self.is_composite = True

    def predict(self, obs, *args, **kwargs):
        # faster for not vectorized case
        if self.n_envs == 1:
            return self.policies[self.env_to_policy_idx[0]].predict(obs, *args, **kwargs)
        
        result = [None for _ in range(self.n_envs)]
        for policy_idx in self.env_to_policy_idx.values():
            envs_with_policy = np.array([env_idx for env_idx, p_idx in self.env_to_policy_idx.items() if p_idx == policy_idx])
            if isinstance(obs, dict):
                obs_env = {
                    key: obs[key][envs_with_policy] for key in obs.keys()
                }
            else:
                obs_env = obs[envs_with_policy]
                
            act_env = self.policies[policy_idx].predict(obs_env, *args, **kwargs)
            
            for env_idx, act in zip(envs_with_policy, act_env):
                result[env_idx] = act
        return np.array(result)

    def reset(self):
        for policy in self.policies:
            policy.reset()
        self.env_to_policy_idx = {i: self.init_policy_index for i in range(self.n_envs)}
        self.action_chunking = self.action_chunking_dict[self.init_policy_index]
        self.n_obs_steps = self.n_obs_steps_dict[self.init_policy_index]

    def switch_policy(self, obs, act, env):
        raise NotImplementedError
    