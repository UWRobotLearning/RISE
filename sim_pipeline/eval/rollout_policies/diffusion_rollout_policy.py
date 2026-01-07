import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from sim_pipeline.utils.rollout_utils import filter_obs_keys
from sim_pipeline.eval.rollout_policies.rollout_policy import RolloutPolicy

class DiffusionRolloutPolicy(RolloutPolicy):
    def __init__(
            self,
            policy,
            action_chunking: bool = True,
            n_obs_steps: int | None = 2,
            obs_keys: list[str] | None = None,
            lowdim: bool = False,
        ):
        super().__init__(policy, action_chunking=action_chunking, n_obs_steps=n_obs_steps)
        self.obs_keys = obs_keys
        self.lowdim = lowdim

    def reset(self):
        self.policy.reset()

    def predict(self, obs, *args, **kwargs):
        if not isinstance(obs, dict):
            raise ValueError("Diffusion policy expects dict input")
        np_obs_dict = filter_obs_keys(obs, obs_keys=self.obs_keys)
        if self.lowdim:
            obs = np.concatenate([
                np_obs_dict[key] for key in self.obs_keys
            ], axis=-1)
            np_obs_dict = {'obs': obs}

        # device transfer
        obs_dict = dict_apply(np_obs_dict,
            lambda x: torch.from_numpy(x).to(
                device=self.policy.device))

        with torch.no_grad():
            action_dict = self.policy.predict_action(obs_dict)

        # device_transfer
        np_action_dict = dict_apply(action_dict,
            lambda x: x.detach().to('cpu').numpy())

        action = np_action_dict['action']
        if not np.all(np.isfinite(action)):
            print(action)
            raise RuntimeError("Nan or Inf action")

        # TODO: must enable to support abs action            
        # if self.abs_action:
        #     action = self.undo_transform_action(action)

        # shape is (n_envs, execution_horizon, action_dim)
        return action
    