import numpy as np
from copy import deepcopy
from robosuite.robots.single_arm import SingleArm
from robosuite.environments.robot_env import RobotEnv

class InitializationWrapper:
    def __init__(self, env: RobotEnv, init_ee_range, seed=None):
        assert isinstance(env, RobotEnv)
        
        self.env = env
        self.init_ee_range: list[list[float]] = init_ee_range
        
        if seed is None:
            seed = np.random.randint(0, 1000)
        np.random.seed(seed)
                
    def __getattr__(self, name):
        return getattr(self.env, name)
    
    def reset(self):
        self.env.reset()
        
        ### initialization: move to random ee position within range
                
        # randomly sample target ee pos
        target_ee_pos = np.array([np.random.uniform(self.init_ee_range[0][0], self.init_ee_range[0][1]),
                                  np.random.uniform(self.init_ee_range[1][0], self.init_ee_range[1][1]),
                                  np.random.uniform(self.init_ee_range[2][0], self.init_ee_range[2][1])])
        
        curr_action_dim = self.env.action_dim
        self.env._action_dim = 4
                
        # switch to absolute ee pos controller    

        # TODO support multiple robots
        robots: list[SingleArm] = self.env.robots
        for robot in robots:
            curr_controller = robot.controller
            
            robot._load_controller()
            abs_controller = robot.controller
            abs_controller.use_delta = False
            abs_controller.use_ori = False
            abs_controller.control_dim = 3
            abs_controller.input_max = abs_controller.input_max[:abs_controller.control_dim]
            abs_controller.input_min = abs_controller.input_min[:abs_controller.control_dim]
            abs_controller.output_max = abs_controller.output_max[:abs_controller.control_dim]
            abs_controller.output_min = abs_controller.output_min[:abs_controller.control_dim]

        robot.controller.reset_goal()
        
        # obtained from running robosuite
        starting_z = 1.011049
        
        # to ensure robot does not collide with objects, execute xy movement before z movement
        xy_goal = target_ee_pos[:2]
        
        # TODO automatically determine when done
        num_steps = 25
        for i in range(num_steps):
            action = np.r_[xy_goal, starting_z, 0.0]
            obs, reward, done, info = self.env.step(action)
            # print(obs['robot0_eef_pos'], action)        # # randomly sample target ee pos

            action = np.r_[target_ee_pos, 0.0]
            obs, reward, done, info = self.env.step(action)
            # print(obs['robot0_eef_pos'], action)
            
        # reset controller
        robot.controller = curr_controller
        robot.controller.reset_goal()
        self.env._action_dim = curr_action_dim
        
        self.env.timestep = 0
        
        return self.env._get_observations(force_update=True)