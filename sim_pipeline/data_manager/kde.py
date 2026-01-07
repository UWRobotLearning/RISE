import click
import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.neighbors import KernelDensity
from mpl_toolkits.mplot3d import Axes3D  # necessary for 3D plotting

@click.command()
@click.option('--dataset', '-d', required=True, type=click.Path(exists=True), help='Path to the HDF5 dataset.')
@click.option('--sample_percentage', '-p', default=100.0, help='Percentage of states to sample (0-100).')
@click.option('--use-states', '-s', is_flag=True, default=True, help='Plot state distribution only. If False, plot state-action distribution instead.')
@click.option('--query-state', '-q', default=[-0.2, 0.2, 0.9], multiple=True, help='Query state for state-action distribution.')
@click.option('--state-bandwidth', '-sb', default=1.0, help='Bandwidth for state-action query, to specify neighbhorhood-ness')
@click.option('--kde-bandwidth-state', '-ks', default=0.01, help='Bandwidth for state KDE')
@click.option('--kde-bandwidth-action', '-ka', default=0.01, help='Bandwidth for action KDE')        
def main(dataset, sample_percentage, use_states, query_state, state_bandwidth, kde_bandwidth_state, kde_bandwidth_action):
    # -------------------------------
    # 1. Load your data
    # -------------------------------
    with h5py.File(dataset, 'r') as f:
        data = f['data']

        # gather states and actions from all episodes
        states_list = []
        actions_list = []

        for ep_key in data:
            ep = data[ep_key]
            # Replace this with your actual dataset fields for (X, Y, Z) state data
            ep_states = ep['obs']['robot0_eef_pos'][:]
            # valid_indices = np.where((ep_states[:, 0] < -0.1) & (ep_states[:, 1] > 0.1))[0]
            filtered_states = ep_states[:]
            filtered_actions = ep['actions'][:, :3][:]

            states_list.append(filtered_states)
            actions_list.append(filtered_actions)

        # combine all episodes into single arrays
        states = np.concatenate(states_list, axis=0)
        actions = np.concatenate(actions_list, axis=0)

    # # Sample only p% of the states if needed
    if sample_percentage < 100.0:
        p = sample_percentage / 100.0
        total_states = len(states)
        sample_size = int(total_states * p)
        indices = np.random.choice(total_states, sample_size, replace=False)
        states = states[indices]
        actions = actions[indices]
        
    if use_states:
        state_distribution(states, kde_bandwidth_state)
    else:
        state_action_distribution(states, actions, query_state, state_bandwidth, kde_bandwidth_action)

    
def state_distribution(states, kde_bandwidth_state):
    # -------------------------------
    # 2. Run KDE on the state positions
    # -------------------------------
    bandwidth = kde_bandwidth_state
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(states)

    # Evaluate the density at each state position
    log_density = kde.score_samples(states)  # returns log density
    density = np.exp(log_density)            # convert to actual density values

    # -------------------------------
    # 3. Visualize the data in 3D
    # -------------------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the state positions, coloring by density
    scatter = ax.scatter(states[:, 0], states[:, 1], states[:, 2],
                         c=density, cmap='viridis', s=50, alpha=0.8)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Estimated Density')

    # Label axes and add a title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D State KDE')

    plt.show()

    # -------------------------------
    # 4. Visualize pairwise 2D slices
    # -------------------------------
    state_labels = ['X', 'Y', 'Z']
    pairs = [(0, 1), (0, 2), (1, 2)]

    fig2, axs = plt.subplots(1, 3, figsize=(18, 5))
    for ax2, (i, j) in zip(axs, pairs):
        s_i = states[:, i]
        s_j = states[:, j]
        
        # Define grid limits for the current two dimensions (with a margin)
        margin = 0.5
        s_i_min, s_i_max = s_i.min() - margin, s_i.max() + margin
        s_j_min, s_j_max = s_j.min() - margin, s_j.max() + margin
        
        # Create a grid over the two dimensions
        grid_i = np.linspace(s_i_min, s_i_max, 100)
        grid_j = np.linspace(s_j_min, s_j_max, 100)
        S_i, S_j = np.meshgrid(grid_i, grid_j)
        
        # For the full 3D state KDE, fix the remaining dimension to its mean
        k = [dim for dim in range(3) if dim not in [i, j]][0]
        mean_k = np.mean(states[:, k])
        
        # Build grid points in 3D: set dimensions i and j from the grid, dimension k fixed
        grid_points = np.zeros((S_i.size, 3))
        grid_points[:, i] = S_i.ravel()
        grid_points[:, j] = S_j.ravel()
        grid_points[:, k] = mean_k

        # Evaluate the density on the grid (in log space)
        log_dens = kde.score_samples(grid_points)
        dens = np.exp(log_dens).reshape(S_i.shape)
        
        # Plot the contour density
        cont = ax2.contourf(S_i, S_j, dens, levels=30, cmap='viridis')
        ax2.set_xlabel(state_labels[i])
        ax2.set_ylabel(state_labels[j])
        ax2.set_title(f'Pairwise KDE ({state_labels[i]}, {state_labels[j]})')

    fig2.colorbar(cont, ax=axs, fraction=0.03, pad=0.04, label='Density')
    plt.suptitle('Pairwise 2D State KDE Slices', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
def state_action_distribution(states, actions, query_state, state_bandwidth, kde_bandwidth_action):
    # -------------------------------
    # 2. Specify query state and "nearbiness" in state space
    # -------------------------------
    # The state bandwidth controls how quickly the influence of a data point decays with distance.
    state_bw = state_bandwidth

    # -------------------------------
    # 3. Compute weights based on state proximity
    # -------------------------------
    # Using a Gaussian kernel on the state-space distance:
    diffs = states - query_state              # (n_samples, 3)
    dists_sq = np.sum(diffs**2, axis=1)
    weights = np.exp(-dists_sq / (2 * state_bw**2))

    # Set weights of actions with magnitude less than 0.1 to 0
    action_magnitudes = np.linalg.norm(actions, axis=1)
    weights[action_magnitudes < 0.1] = 0

    # (Optional) Normalize the weights (not strictly necessary for KDE)
    if np.sum(weights) > 0:
        weights /= np.sum(weights)
    
    # -------------------------------
    # 4. Estimate the marginal action distribution with weighted KDE
    # -------------------------------
    # We’re estimating a KDE on the 3D action space.
    action_bw = kde_bandwidth_action
    kde_action = KernelDensity(kernel='gaussian', bandwidth=action_bw)
    kde_action.fit(actions, sample_weight=weights)  # Note: scikit-learn supports sample_weight in recent versions

    # -------------------------------
    # 5. Visualize the marginal action distribution (pairwise 2D slices)
    # -------------------------------
    # Because our action space is 3D, we plot 2D slices for each pair of action dimensions.
    action_labels = ['dx', 'dy', 'dz']
    pairs = [(0, 1), (0, 2), (1, 2)]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (i, j) in zip(axs, pairs):
        # Extract the corresponding action dimensions for plotting
        a_i = actions[:, i]
        a_j = actions[:, j]
        
        # Define grid limits for the current two dimensions (with a margin)
        margin = 0.5
        a_i_min, a_i_max = a_i.min() - margin, a_i.max() + margin
        a_j_min, a_j_max = a_j.min() - margin, a_j.max() + margin
        
        # Create a grid over the two dimensions
        grid_i = np.linspace(a_i_min, a_i_max, 100)
        grid_j = np.linspace(a_j_min, a_j_max, 100)
        A_i, A_j = np.meshgrid(grid_i, grid_j)
        
        # For the full 3D action KDE, we must provide a value for the third (omitted) dimension.
        # A good choice is its weighted mean (in the neighborhood).
        k = [dim for dim in range(3) if dim not in [i, j]][0]
        weighted_mean_k = np.average(actions[:, k], weights=weights)
        
        # Build grid points in 3D: set dimensions i and j from the grid, and dimension k fixed.
        grid_points = np.zeros((A_i.size, 3))
        grid_points[:, i] = A_i.ravel()
        grid_points[:, j] = A_j.ravel()
        grid_points[:, k] = weighted_mean_k

        # Evaluate the density on the grid (in log space)
        log_dens = kde_action.score_samples(grid_points)
        dens = np.exp(log_dens).reshape(A_i.shape)
        
        # Plot the contour density and overlay the actual (projected) action samples.
        cont = ax.contourf(A_i, A_j, dens, levels=30, cmap='viridis')
        # ax.scatter(a_i, a_j, c='red', s=10, alpha=0.5)
        ax.set_xlabel(action_labels[i])
        ax.set_ylabel(action_labels[j])
        ax.set_title(f'p({action_labels[i]}, {action_labels[j]} | s ~ {query_state})')

    # Add a colorbar common to all subplots.
    fig.colorbar(cont, ax=axs, fraction=0.03, pad=0.04, label='Density')
    plt.suptitle('Marginal Action Distribution p(a|s) in a State Neighborhood', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



if __name__ == "__main__":
    main()