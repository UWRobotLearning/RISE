import numpy as np

from sim_pipeline.utils.get_base_env import get_base_env
from sim_pipeline.success_functions.success_registry import register_success_func
from robosuite.environments.manipulation.nut_assembly import NutAssembly


@register_success_func
def pushing_success(obs, env):
    env = get_base_env(env, NutAssembly)
    if env is None:
        raise ValueError('Invalid env class. Must wrap a NutAssembly.')
    
    nut_name = 'SquareNut'
    # avg position of ID 
    goal_position = np.array([-0.1125, 0.1675, 0.823])

    object_loc_center: np.ndarray = env.sim.data.body_xpos[env.obj_body_id[nut_name]]

    dist = np.linalg.norm(object_loc_center - goal_position)
    return dist < 0.07