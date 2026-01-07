import numpy as np

from impls.utils.evaluation import supply_rng

from diffusion_policy.common.pytorch_util import dict_apply
from sim_pipeline.utils.rollout_utils import filter_obs_keys
from sim_pipeline.eval.rollout_policies.rollout_policy import RolloutPolicy

class OGBenchRolloutPolicy(RolloutPolicy):
    def __init__(
            self,
            policy,
            action_chunking: bool = True,
            n_obs_steps: int | None = 2,
            obs_keys: list[str] | None = None,
            lowdim: bool = False,
        ):
        import jax

        super().__init__(policy, action_chunking=action_chunking, n_obs_steps=n_obs_steps)
        self.obs_keys = obs_keys
        self.lowdim = lowdim
        
        self.actor_fn = supply_rng(self.policy.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))

    def reset(self):
        pass

    def predict(self, obs, *args, **kwargs):
        np_obs_dict = filter_obs_keys(obs, obs_keys=self.obs_keys)
        if self.lowdim:
            obs = np.concatenate([
                np_obs_dict[key] for key in self.obs_keys
            ], axis=-1)

        # hardcoded for now
        goal = np.array([1.14608096e-01, 1.52866992e-01, 8.29978946e-01, 0.0,
            0.0,  5.80254076e-01,  8.14435515e-01,  2.09027917e-03,
            4.97338057e-02,  4.99918345e-04, -7.93045580e-01,  6.07315481e-01,
            -3.82225290e-02,  2.80285198e-02,
            # eef
            -0.11330734,  0.20262865,  0.83037664, 
            #quat
             9.98281814e-01, -3.44512741e-02,  4.73931190e-02,  6.49702759e-04,
             0.04, -0.04])
        goal = goal.reshape(1, -1)
        action = self.actor_fn(observations=obs, goals=goal, temperature=0)
        return np.array(action)
    