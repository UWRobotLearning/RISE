import numpy as np


class RolloutStats:
    def __init__(self):
        self.all_reward = []
        self.all_successes = []
        self.all_final_successes = []
        self.num_episodes = 0

        self.episode_returns = []
        self.episode_successes = []

    def reset_episode(self):
        self.episode_returns = []
        self.episode_successes = []

    def add_step(self, reward: float | np.float64 | np.ndarray[np.float32], success: bool | list[bool]):
        # reward is a float when not vectorized, and a np array of floats when vectorized (from sb3 VecEnv)
        # success is bool when not vectorized, list of bools when vectorized
        # fun fact: robomimic envs by default returns float64 for rew
        self.episode_returns.append(reward)
        self.episode_successes.append(success)

    def finalize_episode(self):
        # shape (ep_len, num_envs) or (ep_len,)
        stacked_rew: np.ndarray[np.float64 | np.float32] = np.stack(self.episode_returns)
        # if vec, shape = (num_envs,)
        total_reward: np.ndarray[np.float32] | np.float64 = np.sum(stacked_rew, axis=0)

        # avg over all envs, if any
        self.avg_ep_reward: np.float32 | np.float64 = np.mean(total_reward)

        # avg success is proportion of envs that were successful at any point in the episode
        # avg final success is proportion of envs that were successful at the end of the episode
        self.avg_ep_success: np.float64 = np.mean(np.any(np.stack(self.episode_successes), axis=0))
        self.avg_ep_final_success: np.float64 = np.mean(np.stack(self.episode_successes)[-1])

        self.all_reward.append(self.avg_ep_reward)
        self.all_successes.append(self.avg_ep_success)
        self.all_final_successes.append(self.avg_ep_final_success)

        self.num_episodes += 1

        self.reset_episode()

    def compute_statistics(self):
        if self.num_episodes == 0:
            return

        self.avg_reward = np.mean(self.all_reward)
        self.avg_success = np.mean(self.all_successes)
        self.avg_final_success = np.mean(self.all_final_successes)

        self.std_reward = np.std(self.all_reward)
        self.std_success = np.std(self.all_successes)
        self.std_final_success = np.std(self.all_final_successes)

    def get_episode_stats(self):
        return {
            'reward': self.avg_ep_reward,
            'success': self.avg_ep_success,
        }

    def get_overall_stats(self):
        return {
            'avg_reward': self.avg_reward,
            'std_reward': self.std_reward,
            'avg_success': self.avg_success,
            'std_success': self.std_success,
        }
        
    def get_avg_stats(self):
        return {
            'avg_reward': self.avg_reward,
            'avg_success': self.avg_success,
        }

    def print_stats(self, i: int, total=None):
        ep_stats = self.get_episode_stats()
        all_stats = self.get_overall_stats()
        total_lines = 2 + len(ep_stats) + len(all_stats)
        total_lines = total_lines + 1 if total is not None else total_lines
        # Move cursor to the beginning of the line
        print('\r', end='')
        if total is not None:
            print(f'progress: {i}/{total} ({i/total*100:.2f}%)')
        print(f"===== Rollout {i} Stats =====")
        for k, v in ep_stats.items():
            print(f"{k}: {v}")
        print(f'===== Overall Stats =====')
        for k, v in all_stats.items():
            print(f"{k}: {v}")
        # Move cursor up to beginning of stats to override
        print(f'\033[{total_lines}A', end='')