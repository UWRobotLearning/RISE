import torch
import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils

from sim_pipeline.eval.rollout_policies.rollout_policy import RolloutPolicy


class RobomimicOpexRolloutPolicy(RolloutPolicy):
    def __init__(
            self,
            policy,
            action_chunking: bool = False,
            n_obs_steps: int | None = None,
            is_diffusion: bool = False
        ):
        super().__init__(policy, action_chunking=action_chunking, n_obs_steps=n_obs_steps)
        
        self.counter = 0
        self.grad_norms = []

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
        ac = self.policy.policy.get_action(obs_dict=ob, goal_dict=goal)
        
        # if obs['robot0_eef_pos'][0, -1, 2] < 0.835:
        #     print('touched', self.counter)
        if self.counter > 20:
            print('starting')
            ac = ac.detach()[:, 0, :]
            for key in ob:
                ob[key] = ob[key][:, -1, :]

            ac.requires_grad = True
            
            for i in range(10):
                ac_old = ac.clone().detach()
                ac = ac.detach().requires_grad_(True)
                pred_qs = [critic(obs_dict=ob, acts=ac, goal_dict=goal)
                        for critic in self.policy.policy.nets["critic"]]
                q_pred, _ = torch.cat(pred_qs, dim=1).min(dim=1, keepdim=True)
                q_pred.backward()
                
                # self.grad_norms.append(ac.grad.norm().detach().cpu().numpy())
                # np.savez('grad_norms.npz', grad_norms=np.array(self.grad_norms))
                
                eta = 0.05
                ac = ac + eta * ac.grad
                ac = ac.detach()
                ac[:, 2:] = ac_old[:, 2:]
            
            ac = torch.unsqueeze(ac, 1)
        
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
