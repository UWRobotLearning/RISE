import numpy as np

from sim_pipeline.utils.get_base_env import get_base_env
from sim_pipeline.reward_functions.reward_registry import register_reward_func
from robosuite.environments.manipulation.nut_assembly import NutAssembly


@register_reward_func
def reaching_reward(obs, act, env):
    env = get_base_env(env, NutAssembly)
    if env is None:
        raise ValueError('Invalid env class. Must wrap a NutAssembly.')
    
    nut_name = 'SquareNut'
    
    gripper = env.robots[0].gripper
    gripper_pos: np.ndarray = env.sim.data.get_site_xpos(gripper.important_sites['grip_site'])
    object_loc_center: np.ndarray = env.sim.data.body_xpos[env.obj_body_id[nut_name]]

    dist = np.linalg.norm(gripper_pos - object_loc_center)
    
    return -dist

    
@register_reward_func
def pushing_reward(obs, act, env):
    env = get_base_env(env, NutAssembly)
    if env is None:
        raise ValueError('Invalid env class. Must wrap a NutAssembly.')
    
    nut_name = 'SquareNut'
    
    # shape (3,) -> (x, y, z)
    object_loc: np.ndarray = env.sim.data.body_xpos[env.obj_body_id[nut_name]]
    
    # avg position of ID 
    goal_position = np.array([-0.1125, 0.1675, 0.823])
    
    dist = np.linalg.norm(object_loc - goal_position)
    
    return -dist

@register_reward_func
def pushing_reward_mid(obs, act, env):
    env = get_base_env(env, NutAssembly)
    if env is None:
        raise ValueError('Invalid env class. Must wrap a NutAssembly.')
    
    nut_name = 'SquareNut'
    
    # shape (3,) -> (x, y, z)
    object_loc: np.ndarray = env.sim.data.body_xpos[env.obj_body_id[nut_name]]
    
    # avg position of ID 
    goal_position = np.array([-0.1025, -0.005, 0.823])
    
    dist = np.linalg.norm(object_loc - goal_position)
    
    return -dist

@register_reward_func
def pushing_reward_ll(obs, act, env):
    env = get_base_env(env, NutAssembly)
    if env is None:
        raise ValueError('Invalid env class. Must wrap a NutAssembly.')
    
    nut_name = 'SquareNut'
    
    # shape (3,) -> (x, y, z)
    object_loc: np.ndarray = env.sim.data.body_xpos[env.obj_body_id[nut_name]]
    
    # avg position of ID 
    goal_position = np.array([0.06, -0.06, 0.823])
    
    dist = np.linalg.norm(object_loc - goal_position)
    
    return -dist

@register_reward_func
def reach_and_push_reward(obs, act, env):
    reaching_rew = reaching_reward(obs, act, env)
    pushing_rew = pushing_reward(obs, act, env)
    return reaching_rew + 2 * pushing_rew

@register_reward_func
def reach_and_push_reward_mid(obs, act, env):
    reaching_rew = reaching_reward(obs, act, env)
    pushing_rew = pushing_reward_mid(obs, act, env)
    return reaching_rew + 2 * pushing_rew

@register_reward_func
def reach_and_push_reward_ll(obs, act, env):
    reaching_rew = reaching_reward(obs, act, env)
    pushing_rew = pushing_reward_ll(obs, act, env)
    return reaching_rew + 2 * pushing_rew

@register_reward_func
def stable_pushing_reward(obs, act, env):
    reach_and_push_rew = reach_and_push_reward(obs, act, env)

    def rotate_vector(q, v):
        """
        Rotate a 3D vector using a quaternion.
        
        Args:
        q (numpy.array): Quaternion in the form [w, x, y, z] (scalar first)
        v (numpy.array): 3D vector to rotate
        
        Returns:
        numpy.array: Rotated 3D vector
        """
        # Ensure q is a unit quaternion
        q = q / np.linalg.norm(q)
        
        # Extract components
        w, x, y, z = q
        
        # Compute the rotated vector
        v_rotated = v + 2 * np.cross(q[1:], np.cross(q[1:], v) + w * v)
        
        return v_rotated
    def rotational_distance(v1, v2):
        """
        Calculate the rotational distance (angle) between two 3D vectors.
        
        Args:
        v1 (numpy.array): First 3D vector
        v2 (numpy.array): Second 3D vector
        
        Returns:
        float: Angle between the vectors in radians
        """
        # Normalize the vectors
        v1_normalized = v1 / np.linalg.norm(v1)
        v2_normalized = v2 / np.linalg.norm(v2)
        
        # Calculate the dot product
        dot_product = np.dot(v1_normalized, v2_normalized)
        
        # Clamp the dot product to [-1, 1] to avoid numerical errors
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Calculate the angle using arccos
        angle = np.arccos(dot_product)
        
        return angle

    # in robosuite, yaw is x-axis, we don't care about yaw
    vector = np.array([1, 0, 0])
    
    # make more efficient?
    obs = env._get_observations()
    rotated = rotate_vector(obs['robot0_eef_quat'], vector)
    
    return -rotational_distance(vector, rotated) + reach_and_push_rew


@register_reward_func
def stable_pushing_reward_ll(obs, act, env):
    reach_and_push_rew = reach_and_push_reward_ll(obs, act, env)

    def rotate_vector(q, v):
        """
        Rotate a 3D vector using a quaternion.
        
        Args:
        q (numpy.array): Quaternion in the form [w, x, y, z] (scalar first)
        v (numpy.array): 3D vector to rotate
        
        Returns:
        numpy.array: Rotated 3D vector
        """
        # Ensure q is a unit quaternion
        q = q / np.linalg.norm(q)
        
        # Extract components
        w, x, y, z = q
        
        # Compute the rotated vector
        v_rotated = v + 2 * np.cross(q[1:], np.cross(q[1:], v) + w * v)
        
        return v_rotated
    def rotational_distance(v1, v2):
        """
        Calculate the rotational distance (angle) between two 3D vectors.
        
        Args:
        v1 (numpy.array): First 3D vector
        v2 (numpy.array): Second 3D vector
        
        Returns:
        float: Angle between the vectors in radians
        """
        # Normalize the vectors
        v1_normalized = v1 / np.linalg.norm(v1)
        v2_normalized = v2 / np.linalg.norm(v2)
        
        # Calculate the dot product
        dot_product = np.dot(v1_normalized, v2_normalized)
        
        # Clamp the dot product to [-1, 1] to avoid numerical errors
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Calculate the angle using arccos
        angle = np.arccos(dot_product)
        
        return angle

    # in robosuite, yaw is x-axis, we don't care about yaw
    vector = np.array([1, 0, 0])
    
    # make more efficient?
    obs = env._get_observations()
    rotated = rotate_vector(obs['robot0_eef_quat'], vector)
    
    return -rotational_distance(vector, rotated) + reach_and_push_rew

@register_reward_func
def stable_and_smooth_pushing_reward(obs, act, env, smooth_scale=0.1):
    stable_pushing_rew = stable_pushing_reward(obs, act, env)

    def smoothness_reward(env):
        # penalize large actions
        #return -np.linalg.norm(act)o current_eef_pos = env.sim.data.get_site_xpos(env.robots[0].gripper.important_sites['grip_site'])
        gripper = env.robots[0].gripper
        #eef_velocity : np.ndarray = env.sim.data.get_site_xvel(gripper.important_sites['grip_site'])
        eef_velocity: np.ndarray = env.sim.data.get_body_xvelp('gripper0_eef')

        return -np.linalg.norm(eef_velocity)

    return stable_pushing_rew + smooth_scale * smoothness_reward(env)
