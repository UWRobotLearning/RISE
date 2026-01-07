from collections import OrderedDict

import torch
from torch import nn
from robomimic.models.obs_nets import MIMO_MLP
from robomimic.algo import register_algo_factory_func, PolicyAlgo

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils

class Discriminator_MLP(MIMO_MLP):
    def __init__(self, obs_shapes, hidden_dim=256):
        # requires robomimic OBS_KEYS_TO_MODALITIES to be defined
        obs_group_shapes = {'obs': obs_shapes}
        obs_group_shapes = OrderedDict(obs_group_shapes)
        output_shapes = OrderedDict(expert=(1))
        layer_dims = [hidden_dim, hidden_dim, 1]
        super().__init__(
            input_obs_group_shapes=obs_group_shapes,
            output_shapes=output_shapes,
            layer_dims=layer_dims,
            activation=nn.ReLU,
            output_activation=None,
            layer_func=nn.Linear,
        )
    
    def forward(self, **inputs):
        # import ipdb; ipdb.set_trace()
        enc_outputs = self.nets["encoder"](**inputs)
        mlp_outputs = self.nets["mlp"](enc_outputs)
        return mlp_outputs


class Discriminator(PolicyAlgo):
    def __init__(self, config, obs_shapes, hidden_dim=256, device='cpu'):  
        self.optim_params = config.algo.optim_params
        self.config = config
        self.device = device

        self.nets = nn.ModuleDict()
        self.nets['policy'] = Discriminator_MLP(obs_shapes, hidden_dim=hidden_dim)
        self.nets = self.nets.float().to(self.device)
        
        self._create_optimizers()     

    def train_on_batch(self, batch, epoch, validate):
        with TorchUtils.maybe_no_grad(no_grad=validate):
            # print(f'expert_batch : {expert_batch.keys()}')
            info = OrderedDict()
            predictions = self._forward_training(batch)
            losses = self._compute_losses(predictions, batch['labels'])

            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)
        
        return info

    def _forward_training(self, batch):
        predictions = OrderedDict()
        preds = self.nets['policy'](obs=batch['obs'])
        predictions['preds'] = preds
        return predictions
    
    
    def get_action(self, obs_dict, goal_dict=None):
        preds = self.nets['policy'](obs=obs_dict)
        return preds

    def _compute_losses(self, preds, labels):
        loss = nn.BCEWithLogitsLoss()(preds['preds'], labels)
        
        # Compute accuracy by comparing predicted class (>0.5 after sigmoid) with true labels
        with torch.no_grad():
            predictions = torch.sigmoid(preds['preds']) > 0.5
            accuracy = (predictions == labels).float().mean()

        losses = OrderedDict()
        losses["prediction_loss"] = [loss]
        losses["accuracy"] = [accuracy]
        return losses

    def _train_step(self, losses):

        # gradient step
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["prediction_loss"][0],
        )
        info["prediction_loss"] = losses["prediction_loss"][0]
        info["accuracy"] = losses["accuracy"][0]
        # info["policy_grad_norms"] = policy_grad_norms
        return info

    def process_batch_for_training(self, batch):
        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]}
        # for k in input_batch["obs"]:
        #     if 'image' in k:
        #         # switch to channels last
        #         input_batch['obs'][k] = input_batch['obs'][k].permute(0, 2, 3, 1)
        input_batch["labels"] = batch["rewards"][:, 0].reshape(-1, 1)
        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def _create_optimizers(self):
        self.optimizers = dict()
        self.lr_schedulers = dict()
        self.optimizers["policy"] = torch.optim.Adam(self.nets["policy"].parameters(),lr=0.0001, weight_decay=1e-3)
        
    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = OrderedDict()

        log["bce_loss"] = info["prediction_loss"].item()
        log["accuracy"] = info["accuracy"].item()
        return log