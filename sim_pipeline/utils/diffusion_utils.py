from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy

from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace
from diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace import TrainDiffusionUnetLowdimWorkspace
from diffusion_policy.workspace.train_diffusion_unet_image_workspace import TrainDiffusionUnetImageWorkspace
from diffusion_policy.workspace.base_workspace import BaseWorkspace

def get_policy_class(rgb_keys: list[str], lowdim_keys: list[str], depth_keys: list[str]) -> type:
    if len(rgb_keys) > 0 and len(lowdim_keys) > 0:
        return DiffusionUnetHybridImagePolicy
    if len(rgb_keys) > 0:
        return DiffusionUnetImagePolicy
    if len(lowdim_keys) > 0:
        return DiffusionUnetLowdimPolicy
    
def get_workspace(rgb_keys: list[str], lowdim_keys: list[str], depth_keys: list[str]) -> type[BaseWorkspace]:
    if len(rgb_keys) > 0 and len(lowdim_keys) > 0:
        return TrainDiffusionUnetHybridWorkspace
    if len(rgb_keys) > 0:
        return TrainDiffusionUnetImageWorkspace
    if len(lowdim_keys) > 0:
        return TrainDiffusionUnetLowdimWorkspace