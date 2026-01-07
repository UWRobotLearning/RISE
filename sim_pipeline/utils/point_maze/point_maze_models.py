import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

import numpy as np
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self, input_shape, action_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mu = nn.Linear(64, action_dim)
        self.log_std = nn.Linear(64, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        return mu, log_std

class GaussianActorNetwork(nn.Module):
    def __init__(self, input_shape=6, action_dim=2, fixed_std=False, init_std=0.3, mean_limits=(-1,1), std_limits=(1e-3, 1), std_activation='softplus', low_noise_eval=True):
        super(GaussianActorNetwork, self).__init__()
        self.net = MLP(input_shape, action_dim)
        self.fixed_std = fixed_std
        self.init_std = init_std
        self.mean_limits = np.array(mean_limits)
        self.std_limits = np.array(std_limits)
        self.low_noise_eval = low_noise_eval
    
        def softplus_scaled(x):
            out = F.softplus(x)
            out - out * (self.init_std / F.softplus(torch.zeros(1).to(x.device)))
            return out

        self.std_activations = {
            None: lambda x: x,
            'softplus': softplus_scaled,
        }
        assert std_activation in self.std_activations
        self.std_activation = std_activation if not self.fixed_std else None

        # with torch.no_grad():
        #     for name, layer in self.net.named_children():
        #         nn.init.uniform_(layer.weight, -1, 1)
        #         nn.init.uniform_(layer.bias, -1, 1)
        
    def forward_train(self, x):
        mean, scale = self.net(x)
        scale = scale if not self.fixed_std else torch.ones_like(scale) * self.init_std

        mean = torch.clamp(mean, min=self.mean_limits[0], max=self.mean_limits[1])

        if self.low_noise_eval and not self.training:
            scale = torch.clamp(scale, min=self.std_limits[0], max=self.std_limits[1])
        else:
            scale = self.std_activations[self.std_activation](scale)
            scale = torch.clamp(scale, min=self.std_limits[0], max=self.std_limits[1])
        
        dists = D.Normal(loc=mean, scale=scale)
        dists = D.Independent(dists, 1)
        return dists

    def forward(self, obs):
        dist = self.forward_train(obs)
        if self.low_noise_eval and not self.training:
            return dist.mean
        return dist.sample()
    
class GaussianCNN(nn.Module):
    def __init__(self, input_shape=(3, 160, 160), action_dim=2, fixed_std=False, init_std=0.3, mean_limits=(-1,1), std_limits=(1e-3, 1), std_activation='softplus', low_noise_eval=True):
        super().__init__()
        feature_size = input_shape[1]
        input_channels = input_shape[0]
        self.conv1 = nn.Conv2d(input_channels, 32, 8, stride=4)
        feature_size = (feature_size - 8) // 4 + 1
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        feature_size = (feature_size - 4) // 2 + 1
        self.conv3 = nn.Conv2d(64, 3, 3, stride=1)
        feature_size = (feature_size - 3) // 1 + 1
        self.encoding_size = 256
                
        self.fc1 = nn.Linear(feature_size * feature_size * 3, self.encoding_size)
        self.mu = nn.Linear(self.encoding_size, action_dim)
        self.log_std = nn.Linear(self.encoding_size, action_dim)
        self.fixed_std = fixed_std
        self.init_std = init_std
        self.mean_limits = np.array(mean_limits)
        self.std_limits = np.array(std_limits)
        self.low_noise_eval = low_noise_eval
    
        def softplus_scaled(x):
            out = F.softplus(x)
            out - out * (self.init_std / F.softplus(torch.zeros(1).to(x.device)))
            return out

        self.std_activations = {
            None: lambda x: x,
            'softplus': softplus_scaled,
        }
        assert std_activation in self.std_activations
        self.std_activation = std_activation if not self.fixed_std else None

        # with torch.no_grad():
        #     for name, layer in self.net.named_children():
        #         nn.init.uniform_(layer.weight, -1, 1)
        #         nn.init.uniform_(layer.bias, -1, 1)
        
    def forward_train(self, x, return_encoding=False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        if return_encoding:
            return x
        mean = self.mu(x)
        scale = self.log_std(x)
        scale = scale if not self.fixed_std else torch.ones_like(scale) * self.init_std

        mean = torch.clamp(mean, min=self.mean_limits[0], max=self.mean_limits[1])

        if self.low_noise_eval and not self.training:
            scale = torch.clamp(scale, min=self.std_limits[0], max=self.std_limits[1])
        else:
            scale = self.std_activations[self.std_activation](scale)
            scale = torch.clamp(scale, min=self.std_limits[0], max=self.std_limits[1])
        
        dists = D.Normal(loc=mean, scale=scale)
        dists = D.Independent(dists, 1)
        return dists
    
    def forward(self, obs):
        dist = self.forward_train(obs)
        if self.low_noise_eval and not self.training:
            return dist.mean
        return dist.sample()
 
class BC_Gaussian(nn.Module):
    def __init__(self, images=False, input_shape=6, action_dim=2, fixed_std=False, init_std=0.3, mean_limits=(-1,1), std_limits=(1e-3, 1), std_activation='softplus', low_noise_eval=True):
        super(BC_Gaussian, self).__init__()
        if images:
            self.actor = GaussianCNN(input_shape, action_dim, fixed_std, init_std, mean_limits, std_limits, std_activation, low_noise_eval)
        else:
            self.actor = GaussianActorNetwork(input_shape, action_dim, fixed_std, init_std, mean_limits, std_limits, std_activation, low_noise_eval)
        self.optimizer = Adam(self.actor.parameters(), lr=1e-3)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.98)
    
    def forward_step(self, batch):
        dist = self.actor.forward_train(batch["obs"])
        log_probs = dist.log_prob(batch["action"])
        # with torch.no_grad():
        #     for name, layer in self.net.named_children():
        #         nn.init.uniform_(layer.weight, -1, 1)
        #         nn.init.uniform_(layer.bias, -1, 1)

        return log_probs
    
    def train_step(self, obs, action):
        self.optimizer.zero_grad()
        log_probs = self.forward_step({"obs": obs, "action": action})
        loss = -log_probs.mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def forward(self, obs):
        return self.actor.forward(obs)
    
    def return_encoding(self, obs):
        return self.actor.forward_train(obs, return_encoding=True)