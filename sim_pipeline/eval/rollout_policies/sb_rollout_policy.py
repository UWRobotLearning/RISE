from sim_pipeline.eval.rollout_policies.rollout_policy import RolloutPolicy

class StableBaselinesRolloutPolicy(RolloutPolicy):
    def __init__(
            self,
            policy,
            action_chunking: bool = False,
            n_obs_steps: int | None = None,
        ):
        super().__init__(policy, action_chunking=action_chunking, n_obs_steps=n_obs_steps)

    def reset(self):
        pass
    
    def predict(self, obs, *args, **kwargs):
        act, _state = self.policy.predict(obs)
        return act
