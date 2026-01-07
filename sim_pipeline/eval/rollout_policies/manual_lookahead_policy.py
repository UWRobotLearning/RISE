import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import numpy as np
import torch

from sim_pipeline.eval.rollout_policies.rollout_policy import RolloutPolicy


class ManualLookaheadRolloutPolicy(RolloutPolicy):
    def __init__(
            self,
            policy,
            action_chunking: bool = False,
            n_obs_steps: int | None = None,
            is_diffusion: bool = False
        ):
        super().__init__(policy, action_chunking=action_chunking, n_obs_steps=n_obs_steps)
        
        self.counter = 0

        if is_diffusion and self.policy.policy.ema is not None:
            self.policy.policy.ema.copy_to(self.policy.policy.nets['policy'].parameters())

    def reset(self):
        self.counter = 0

    def predict(self, obs, *args, **kwargs):
        self.counter += 1
        # we must copy robomimic.algo.algo.RolloutPolicy.__call__() here
        # b/c we override _prepare_observation() to assume batch dimension
        ob = self._robomimic_prepare_observations(obs)
        
        if 'goal' in kwargs and kwargs['goal'] is not None:
            goal = self._robomimic_prepare_observations(kwargs['goal'])
        else:
            goal = None
        
        if ob['robot0_eef_pos'][0, -1, 2] < 0.835:
            ac = self.policy.policy.get_action(obs_dict=ob, goal_dict=goal)
        else:
            # print('manual sampling')
            act_min = np.array([-0.2, -0.2, -0.02, 0.0, 0.0, 0.0, -1.0])
            act_max = np.array([0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
            
            size = 30000
            
            random_action_sample = np.random.uniform(act_min, act_max, size=(size, 7))
            random_action_sample = TensorUtils.to_tensor(random_action_sample)
            random_action_sample = TensorUtils.to_device(random_action_sample, self.policy.policy.device)
            random_action_sample = TensorUtils.to_float(random_action_sample)
            
            for key in ob:
                ob[key] = ob[key][:, -1, :]
                ob[key] = torch.tile(ob[key], (size, 1))
            pred_qs = [critic(obs_dict=ob, acts=random_action_sample, goal_dict=goal)
                    for critic in self.policy.policy.nets["critic"]]
            pred_q1 = pred_qs[0].detach().cpu().numpy()
            pred_q2 = pred_qs[1].detach().cpu().numpy()
            pred_q = np.concatenate([pred_q1, pred_q2], axis=0)
            
            best_idx = np.argmax(pred_q)
            if best_idx >= size:
                best_idx -= size
            ac = random_action_sample[best_idx].reshape(1, 1, -1)
        # vf_pred = self.nets["vf"](obs_dict=obs, goal_dict=goal)

        # ac = self.policy.policy.get_action(obs_dict=ob, goal_dict=goal)
        ac = TensorUtils.to_numpy(ac)
        return ac
        
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
