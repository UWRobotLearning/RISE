import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils

from sim_pipeline.eval.rollout_policies.rollout_policy import RolloutPolicy
from scipy.spatial import cKDTree

class RobomimicSamplingRolloutPolicy(RolloutPolicy):
    def __init__(
            self,
            policy,
            action_chunking: bool = False,
            n_obs_steps: int | None = None,
            is_diffusion: bool = False, 
            discriminator=None,
            expert_policy=None,
            action_normalization_stats: dict | None = None
        ):
        super().__init__(policy, action_chunking=action_chunking, n_obs_steps=n_obs_steps, action_normalization_stats=action_normalization_stats)
        
        self.counter = 0
        
        self.all_states = []
        self.all_actions = []
        self.Q_vals = []
        
        self.discriminator = discriminator
        self.expert_policy = expert_policy
        
        if is_diffusion and self.policy.policy.ema is not None:
            self.policy.policy.ema.copy_to(self.policy.policy.nets['policy'].parameters())

    def reset(self):
        self.counter = 0
        # np.savez('states_spectral_0.5.npz', states=np.array(self.all_states), actions=np.array(self.all_actions), Q_vals=np.array(self.Q_vals))
        
    def entropy(self, samples, k=5):
        """
        Estimate entropy of distribution given iid samples using k-nearest neighbors.
        """
        N, D = samples.shape
        tree = cKDTree(samples)
        # Find distances to kth nearest neighbor for each point
        distances, _ = tree.query(samples, k=k+1)  # k+1 because first neighbor is self
        knn_distances = distances[:, k]  # Get distances to kth neighbor
        
        # Calculate entropy estimate
        # H = digamma(N) - digamma(k) + D * np.mean(np.log(knn_distances)) + np.log(2)
        H = -digamma(k) + digamma(N) + D * np.mean(np.log(knn_distances))
        
        return H

    def predict(self, obs, *args, **kwargs):
        self.counter += 1
        # we must copy robomimic.algo.algo.RolloutPolicy.__call__() here
        # b/c we override _prepare_observation() to assume batch dimension
        
        # duplicate obs to get multiple predictions on same obs
        ob = self._robomimic_prepare_observations(obs)
        num_samples = 128
        if 'random_sample' in kwargs and kwargs['random_sample']:
            num_samples = 512
            print('random sample')
        multiplier = 1
        ob = {k: ob[k].repeat(num_samples, *([1] * (len(ob[k].shape) - 1))) for k in ob}
        if 'goal' in kwargs and kwargs['goal'] is not None:
            goal = self._robomimic_prepare_observations(kwargs['goal'])
        else:
            goal = None
        
        ob_unstacked = {k: ob[k].clone() for k in ob}
        for key in ob_unstacked:
            ob_unstacked[key] = ob_unstacked[key][:, -1, :]
            
        Qs = []
        all_actions = []
        all_actions_chunked = []
        for i in range(multiplier):
            if 'random_sample' in kwargs and kwargs['random_sample']:
                act_min = np.array([-0.2, -0.2, -0.02, 0.0, 0.0, 0.0, -1.0])
                act_max = np.array([0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 1.0])

                actions = np.random.uniform(act_min, act_max, size=(num_samples, 7)).astype(np.float32)
                actions = TensorUtils.to_tensor(actions)
                actions = TensorUtils.to_device(actions, self.policy.policy.device)
                
                all_actions.append(actions.detach().cpu().numpy())
            else:
                actions = self.policy.policy.get_action(obs_dict=ob, goal_dict=goal)
                all_actions_chunked.append(actions.detach().cpu().numpy())
                actions = actions[:, 0, :]
                all_actions.append(actions.detach().cpu().numpy())

            pred_qs = [critic(obs_dict=ob_unstacked, acts=actions, goal_dict=goal)
                    for critic in self.policy.policy.nets["critic"]]
            pred_q1 = pred_qs[0].detach().cpu().numpy()
            pred_q2 = pred_qs[1].detach().cpu().numpy()

            min_q = np.minimum(pred_q1, pred_q2)
            Qs.append(min_q)
    
        Qs = np.concatenate(Qs, axis=0)
        actions = np.concatenate(all_actions, axis=0)
        if not ('random_sample' in kwargs and kwargs['random_sample']):
            all_actions_chunked = np.concatenate(all_actions_chunked, axis=0)
        # entropy = self.entropy(actions[:, :3])
        # print(entropy)
        # if ob['robot0_eef_pos'][0, -1, 2] > 0.835:
        #     best_idx = np.argmax(min_q)
        # else:
        best_idx = np.argmax(Qs)
        
        if self.discriminator is not None:
            ob2 = self._robomimic_prepare_observations(obs)
            # get only the last state from each obs key
            ob2 = {k: ob2[k][:, -1, ...] for k in ob2}
            predict = self.discriminator.policy.nets['policy'](obs=ob2)
            print(predict)
        
        # if ob['robot0_eef_pos'][0, -1, 2] > 0.855:
        #     # pass
        #     self.all_states.append(obs['robot0_eef_pos'][0, -1, :3])
        #     self.object_state = obs['object'][0, -1, :3]
        #     self.all_actions.append(actions[:, :3])
        #     self.Q_vals.append(Qs)
        # else:
        #     self.all_states.append(obs['robot0_eef_pos'][0, -1, :3])
        #     self.object_state = obs['object'][0, -1, :3]
        #     self.all_actions.append(actions[:, :3])
        #     self.Q_vals.append(Qs)
        
        # if not ('random_sample' in kwargs and kwargs['random_sample']):
        #     np.savez('states_idql_two_piece_1.npz', states=np.array(self.all_states), actions=np.array(self.all_actions), Q_vals=np.array(self.Q_vals))

        # if ob['robot0_eef_pos'][0, -1, 2] < 0.835:
        #     if self.counter % 10 == 0:
        #         self.all_states.append(obs['object'][0, -1, :2])
        #         self.all_actions.append(actions[:, :2])
        #         self.Q_vals.append(Qs)
        
        if 'random_sample' in kwargs and kwargs['random_sample']:
            ac = actions[best_idx].reshape(1, 1, -1)    
        else:
            ac = all_actions_chunked[best_idx][None, :8]

        # if self.counter > 0:
        #     print('bc!!!')
        #     ob2 = self._robomimic_prepare_observations(obs)
        #     ac = self.policy.policy.get_action(obs_dict=ob2, goal_dict=goal)
            # print(ac[:, 0, -1])
            # if ac[:, 0, -1] > 0.0:
            #     import ipdb; ipdb.set_trace()

            # ob2 = self._robomimic_prepare_observations(obs)
            # ob2_unstacked = {k: ob2[k].clone() for k in ob2}
            # for key in ob2_unstacked:
            #     ob2_unstacked[key] = ob2_unstacked[key][:, -1, :]
            # crit = self.policy.policy.nets["critic"][0]
            # import torch
            # act = torch.tensor(ac[:, 0, :]).to(self.policy.policy.device)
            # import ipdb; ipdb.set_trace()
            
        # if self.counter > 190:
        #     import ipdb; ipdb.set_trace()
                
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
