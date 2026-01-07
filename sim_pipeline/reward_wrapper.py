import gymnasium as gym
import mujoco
import numpy as np
from enum import Enum

class TaskRewardType:
    PICK_PLACE = 'pick_place'
    PUSH = 'push'

class RewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_name: TaskRewardType=None, **kwargs):
        super().__init__(env)

        if reward_name == TaskRewardType.PICK_PLACE:
            self.reward_fn = self.pick_place_reward
        elif reward_name == TaskRewardType.PUSH:
            self.reward_fn = self.push_reward
        else:
            self.reward_fn = self.default_reward

    def step(self, action):
        obs, orig_rew, done, info = self.env.step(action)

        reward = self.reward_fn(action)

        return obs, reward + orig_rew, done, info
    
    def default_reward(self, action):
        return 0.0
    
    def pick_place_reward(self, action):
        # from maniskill
        reward = 0.0

        # reaching reward
        ee_pos = self.env.unwrapped._robot.get_ee_pose()[:2]
        obj_pos = self.env.get_obj_pose()
        dist = np.linalg.norm(ee_pos - obj_pos)
        reaching_reward = 1 - np.tanh(5 * dist)
        reward += reaching_reward

        # grasp reward, need to implement this somehow
        # is_grasped = self.agent.check_grasp(self.obj, max_angle=30)
        # if is_grasped:
        #     reward += 0.25

        # if is_grasped:
        obj_to_goal_dist = np.linalg.norm(np.r_[self.goal_obj_pose['x'], self.goal_obj_pose['y'], 0.0] - obj_pos)
        place_reward = 1 - np.tanh(5 * obj_to_goal_dist)
        reward += place_reward

        return reward

    def push_reward(self, action):
        # from patrick's code
        obj_pose = self.env.get_obj_pose()
        ee_pos = self.env.unwrapped._robot.get_ee_pose()[:2]
        goal_ee_pos = obj_pose[:2]
        goal_ee_pos[0] -= 0.035
        ee_dist_to_goal = np.linalg.norm(ee_pos - goal_ee_pos)
        obj_dist_to_goal = np.linalg.norm(self.goal_obj_pose['x'] - obj_pose[0]) 
        goal_reached = obj_dist_to_goal < .025
        obj_off_table = obj_pose[2] <= 0.
        return - ee_dist_to_goal - obj_dist_to_goal + goal_reached - obj_off_table

