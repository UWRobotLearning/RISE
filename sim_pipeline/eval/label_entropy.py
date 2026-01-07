import hydra
import numpy as np
import h5py

from tqdm import tqdm
from omegaconf import open_dict
from scipy.spatial import cKDTree
from scipy.special import digamma

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils

from sim_pipeline.data_manager.get_dataset import get_dataset
from sim_pipeline.eval.eval import get_policy
from sim_pipeline.eval.rollout_policies.robomimic_sampling_rollout_policy import RobomimicSamplingRolloutPolicy
from sim_pipeline.configs._evaluation.eval_base import EvaluationConfig
from sim_pipeline.utils.experiment import setup_experiment
from sim_pipeline.configs.constants import PolicyType


def entropy(samples: np.ndarray, k: int = 5) -> float:
    """
    Estimate entropy of the distribution given i.i.d. samples using k-nearest neighbors.
    Adapted from RobomimicSamplingRolloutPolicy.
    """
    if len(samples.shape) != 2:
        raise ValueError(f"Expected samples to have shape (N, D), got {samples.shape}")
    N, D = samples.shape
    tree = cKDTree(samples)
    # Find distances to kth nearest neighbor for each point
    distances, _ = tree.query(samples, k=k + 1)  # k+1 because the first neighbor is the point itself
    knn_distances = distances[:, k]  # Distances to the kth nearest neighbor
    
    # Calculate KNN-based entropy estimate
    H = -digamma(k) + digamma(N) + D * np.mean(np.log(knn_distances))
    return float(H)


def compute_entropy_for_single_obs(
    rollout_policy: RobomimicSamplingRolloutPolicy,
    obs_dict: dict,
    num_samples: int = 16,
    multiplier: int = 4,
    k_for_entropy: int = 5,
) -> float:
    """
    Given a single (unbatched) observation dictionary (each key -> shape (...)),
    replicate it multiple times, sample actions from the policy,
    and compute the entropy of all sampled actions.
    """

    prepared_obs = {}
    for k, v in obs_dict.items():
        v = v[None, None, ...]  
        prepared_obs[k] = v
    # Stack the same observation H times along time dimension (dim=1)
    # Each key is currently shape (1, 1, D1, D2, ...)
    H = rollout_policy.n_obs_steps
    if H is not None and H > 1:
        for k, v in prepared_obs.items():
            # Repeat the same observation H times along dim 1
            prepared_obs[k] = v.repeat(H, axis=1)
            
    # Use the rollout_policy's own internal method for normalizing / shaping
    ob = rollout_policy._robomimic_prepare_observations(prepared_obs)

    # replicate each key num_samples times along the batch dimension
    # so shape (1, T, feat_dim) -> (num_samples, T, feat_dim)
    ob = {k: ob[k].repeat(num_samples, 1, 1) for k in ob}
    
    # sample multiple sets of actions
    all_actions = []
    for _ in range(multiplier):
        # This calls Robomimic's model to get an action for each of the num_samples copies
        # shape of actions: (num_samples, 1, action_dim)
        actions = rollout_policy.policy.policy.get_action(obs_dict=ob, goal_dict=None)
        # remove time dimension
        actions = actions[:, 0, :]
        all_actions.append(actions.detach().cpu().numpy())

    actions_np = np.concatenate(all_actions, axis=0)  # shape: (num_samples*multiplier, action_dim)
    return entropy(actions_np, k=k_for_entropy)

@hydra.main(version_base=None, config_path="../configs", config_name="eval_robomimic_iql_diffusion")
def annotate_entropy(config: EvaluationConfig):
    """
    Example script that:
      1) Loads a Robomimic policy (e.g., IQL, IDQL) using sim_pipeline's utilities.
      2) Loads a Robomimic-format dataset.
      3) For each demonstration and each timestep, computes an action-entropy value by
         sampling policy actions multiple times from that state.
      4) Stores the entropy values in a new 'entropy' field in the dataset.
      5) Normalizes entropy values across the entire dataset to range [0,1].

    Make sure your config has the correct dataset, environment, and policy checkpoint info.
    """
    # Ensure we are on the correct device
    device = setup_experiment(config.exp, eval=True)

    # Load the policy (for IQL/IDQL).
    # This returns a generic RolloutPolicy, which we can cast to RobomimicSamplingRolloutPolicy if needed.
    rollout_policy, _ = get_policy(config.exp.policy_type, config, device=device)

    if not isinstance(rollout_policy, RobomimicSamplingRolloutPolicy):
        # If we are using IDQL or a variant, the easiest way to replicate the multi-sample logic
        # is to use the RobomimicSamplingRolloutPolicy class. Otherwise you can adapt as needed.
        print(
            "WARNING: The loaded policy is not a RobomimicSamplingRolloutPolicy. "
            "This script is tailored to sample multiple actions from IDQL or similar. "
            "Proceeding might require code adaptation."
        )

    # Fetch the dataset path from the config
    dataset_path = get_dataset(config.data)

    # Initialize any normalizations needed by robomimic
    # For IDQL, if your checkpoint has normalization stats, they are typically loaded
    # automatically by the policy. If you need explicit calls, do them here:
    # ObsUtils.initialize_obs_utils_with_obs_specs(specs)  # if needed

    # First pass: compute entropies and collect statistics
    all_entropies = []
    with h5py.File(dataset_path, "r+") as hf:
        data_grp = hf["data"]
        demos = list(data_grp.keys())

        # Create progress bar for demos
        demo_pbar = tqdm(demos, desc="Computing entropies")
        for demo_name in demo_pbar:
            demo_grp = data_grp[demo_name]
            
            # Create or overwrite a dataset for entropy
            T = demo_grp["actions"].shape[0]
            if "entropy" in demo_grp:
                demo_pbar.write(f"Overwriting existing entropy data in {demo_name}...")
                del demo_grp["entropy"]
            ent_dset = demo_grp.create_dataset("entropy", shape=(T,), dtype=np.float32)

            # We'll assume lowdim keys under demo_grp['obs']. Each key -> shape (T, <some_dim>)
            obs_keys = list(demo_grp["obs"].keys())

            # Create progress bar for timesteps
            ts_pbar = tqdm(range(T), desc=f"Processing timesteps for {demo_name}", leave=False)
            demo_entropies = []
            for t in ts_pbar:
                # Build an obs_dict suitable for the policy
                obs_dict = {}
                for k in obs_keys:
                    obs_dict[k] = demo_grp["obs"][k][t]

                # Compute entropy for this timestep
                state_entropy = compute_entropy_for_single_obs(
                    rollout_policy=rollout_policy,
                    obs_dict=obs_dict,
                    num_samples=16,      # number of repeats
                    multiplier=4,        # how many times to sample
                    k_for_entropy=5,     # K in KNN
                )
                demo_entropies.append(state_entropy)
                ent_dset[t] = state_entropy

            all_entropies.extend(demo_entropies)
            demo_pbar.write(f"Finished computing entropy for {demo_name} with {T} frames.")

        # Compute normalization statistics
        all_entropies = np.array(all_entropies)
        entropy_min = np.min(all_entropies)
        entropy_max = np.max(all_entropies)
        entropy_range = entropy_max - entropy_min

        # Second pass: normalize entropies to [0,1]
        demo_pbar = tqdm(demos, desc="Normalizing entropies")
        for demo_name in demo_pbar:
            demo_grp = data_grp[demo_name]
            demo_entropies = demo_grp["entropy"][:]
            
            # Normalize to [0,1] range
            normalized_entropies = (demo_entropies - entropy_min) / entropy_range
            demo_grp["entropy"][:] = normalized_entropies

            demo_pbar.write(f"Normalized entropy for {demo_name}")

        # Store normalization stats in dataset root
        if "entropy_stats" in hf:
            del hf["entropy_stats"]
        stats_grp = hf.create_group("entropy_stats")
        stats_grp.create_dataset("min", data=entropy_min)
        stats_grp.create_dataset("max", data=entropy_max)

    print(f"Done annotating and normalizing dataset. Dataset saved at: {dataset_path}")
    print(f"Entropy statistics - Min: {entropy_min:.4f}, Max: {entropy_max:.4f}")


if __name__ == "__main__":
    annotate_entropy()
