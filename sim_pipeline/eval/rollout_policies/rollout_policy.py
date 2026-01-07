class RolloutPolicy:
    def __init__(
            self, 
            policy,
            action_chunking: bool = False,
            n_obs_steps: int | None = None,
            is_composite: bool = False,
            action_normalization_stats: dict | None = None,
        ):
        self.policy = policy
        self.action_chunking = action_chunking
        self.n_obs_steps = n_obs_steps
        
        self.is_composite = False
        self.action_normalization_stats = action_normalization_stats
        
    def __getattr__(self, name):
        return getattr(self.policy, name)

    def reset(self):
        raise NotImplementedError

    def predict(self, obs, *args, **kwargs):
        raise NotImplementedError
    
    def unnormalize_actions(self, actions):
        if self.action_normalization_stats is None:
            return actions
        # Actions are normalized to [-0.5, 0.5], so unnormalize by:
        # 1. Add 0.5 to get [0,1]
        # 2. Scale by (max-min) and add min
        action_min = self.action_normalization_stats['min']
        action_max = self.action_normalization_stats['max']
        # Don't normalize the gripper dimension
        actions_copy = actions.copy()
        actions_copy[..., :-1] = (actions[..., :-1] + 0.5) * (action_max - action_min) + action_min
        return actions_copy