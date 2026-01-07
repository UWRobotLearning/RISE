import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from sim_pipeline.eval.rollout_policies.rollout_policy import RolloutPolicy


class RobomimicRolloutPolicy(RolloutPolicy):
    def __init__(
            self,
            policy,
            action_chunking: bool = False,
            n_obs_steps: int | None = None,
            is_diffusion: bool = False,
            action_normalization_stats: dict | None = None
        ):
        super().__init__(policy, action_chunking=action_chunking, n_obs_steps=n_obs_steps, action_normalization_stats=action_normalization_stats)

        if is_diffusion and self.policy.policy.ema is not None:
            self.policy.policy.ema.copy_to(self.policy.policy.nets['policy'].parameters())

    def reset(self):
        pass

    def predict(self, obs, *args, **kwargs):
        # we must copy robomimic.algo.algo.RolloutPolicy.__call__() here
        # b/c we override _prepare_observation() to assume batch dimension
        ob = self._robomimic_prepare_observations(obs)
        if 'goal' in kwargs and kwargs['goal'] is not None:
            goal = self._robomimic_prepare_observations(kwargs['goal'])
        else:
            goal = None
        ac = self.policy.policy.get_action(obs_dict=ob, goal_dict=goal)
        ac = TensorUtils.to_numpy(ac)
        return self.unnormalize_actions(ac)
        
    def _robomimic_prepare_observations(self, ob):
        """
        Prepare raw observation dict from environment for policy. 
        Copied from robomimic.algo.algo.RolloutPolicy._prepare_observation().
        
        We override this b/c robomimic assumes that inputs do not have a batch dimension, but
        we add in the batch dimension in `rollout.py` in order to standardize everything between
        vectorized and non-vectorized environments. Thus, the input to robomimic rollout policies
        here *must* have a batch dimension.

        Args:
            ob (dict): single observation dictionary from environment (with batch dimension, 
                and np.array values for each key)
        """
        ob = TensorUtils.to_tensor(ob)
        # we just remove this line to prevent adding batch dimension twice.
        # ob = TensorUtils.to_batch(ob)
        ob = TensorUtils.to_device(ob, self.policy.policy.device)
        ob = TensorUtils.to_float(ob)
        if self.obs_normalization_stats is not None:
            # ensure obs_normalization_stats are torch Tensors on proper device
            obs_normalization_stats = TensorUtils.to_float(TensorUtils.to_device(TensorUtils.to_tensor(self.obs_normalization_stats), self.policy.policy.device))
            # limit normalization to obs keys being used, in case environment includes extra keys
            ob = { k : ob[k] for k in self.policy.policy.global_config.all_obs_keys }
            ob = ObsUtils.normalize_obs(ob, obs_normalization_stats=obs_normalization_stats)
        return ob
