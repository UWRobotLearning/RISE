from sim_pipeline.eval.rollout_policies.rollout_policy import RolloutPolicy
from sim_pipeline.eval.rollout_policies.pd_rollout_policy import PDRolloutPolicy

from sim_pipeline.eval.rollout_policies.composite_rollout_policy import CompositeRolloutPolicy

class HeuristicResetPolicy(CompositeRolloutPolicy):
    def __init__(self, policies: list[RolloutPolicy], n_envs=1, reset_steps=100, init_run_steps=7):
        assert isinstance(policies[1], PDRolloutPolicy)
        super().__init__(policies=policies, init_policy_index=0, n_envs=n_envs)
        self.reset_steps = reset_steps
        self.init_run_steps = init_run_steps
        self.steps = 0
        self.curr_policy = 0
        
    def reset(self):
        super().reset()
        self.steps = 0
        self.curr_policy = 0
        
    def predict(self, obs, *args, **kwargs):
        self.steps += 1
        return super().predict(obs, *args, **kwargs)

    def switch_policy(self, obs, act, env) -> bool:
        switched = False
        if self.steps < self.init_run_steps:
            switched = False
        elif self.init_run_steps <= self.steps < self.init_run_steps + self.reset_steps:
            self.env_to_policy_idx = {env: 1 for env in range(self.n_envs)}
            self.n_obs_steps = self.n_obs_steps_dict[1]
            self.action_chunking = self.action_chunking_dict[1]
            if self.curr_policy != 1:
                self.curr_policy = 1
                switched = True
        elif self.steps >= self.init_run_steps + self.reset_steps:
            self.env_to_policy_idx = {env: 0 for env in range(self.n_envs)}
            self.n_obs_steps = self.n_obs_steps_dict[0]
            self.action_chunking = self.action_chunking_dict[0]
            if self.curr_policy != 0:
                self.curr_policy = 0
                switched = True
        else:
            raise ValueError(f'Invalid state {self.steps}')
        return switched