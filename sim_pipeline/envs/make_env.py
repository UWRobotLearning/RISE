from functools import partial
from omegaconf import OmegaConf
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv

from sim_pipeline.reward_wrapper import RewardWrapper as FrankaRewardWrapper
from sim_pipeline.configs.env.base import BaseEnvConfig
from sim_pipeline.configs.env.robomimic_base import RobomimicEnvConfig
from sim_pipeline.configs.env.franka_base import EnvConfigFranka
from sim_pipeline.configs.robot.base_franka import RobotConfigFranka
from sim_pipeline.configs.constants import EnvType, RobomimicEnvType
from sim_pipeline.data_manager.dataset_metadata_enums import ObsType
from sim_pipeline.envs.reward_wrapper import RewardWrapper

def make_vec_env(
    env_config: BaseEnvConfig,
    num_workers=1,
    seed=0,
    device_id=0,
    debug=False,
    flatten_obs=True,
    gymnasium_api=True,
    use_gym_vec_env=False,
    mp_context: str | None = None,
    obs_modality_specs=None,
):
    if debug:
        vec_env = SyncVectorEnv if use_gym_vec_env else DummyVecEnv
        num_workers = 1
    else:
        vec_env = AsyncVectorEnv if use_gym_vec_env else SubprocVecEnv
    if env_config.env_type == EnvType.POINT_MAZE:
        return vec_env([make_maze_env(env_config, seed) for _ in range(num_workers)])
    elif env_config.env_type == EnvType.FRANKA:
        from weird_franka.robot.sim.vec_env.vec_wrapper import SubVecEnv

        env_fns = [
            partial(
                make_franka_env,
                env_config.robot_config,
                env_config,
                seed=seed + i,
                device_id=device_id
            )
            for i in range(num_workers)
        ]
        return SubVecEnv(env_fns)
    elif env_config.env_type == EnvType.ROBOMIMIC:
        if vec_env == AsyncVectorEnv:
            return vec_env([
                make_robomimic_env(
                    env_config, 
                    seed + i, 
                    device_id, 
                    flatten_obs=flatten_obs, 
                    gymnasium_api=gymnasium_api,
                    obs_modality_specs=obs_modality_specs,
                ) for i in range(num_workers)
            ],
            context=mp_context
            )
        return vec_env([
            make_robomimic_env(
                env_config, 
                seed + i, 
                device_id, 
                flatten_obs=flatten_obs, 
                gymnasium_api=gymnasium_api,
                obs_modality_specs=obs_modality_specs,
            ) for i in range(num_workers)
        ])
    
def make_env(
    env_config: BaseEnvConfig,
    seed=0,
    device_id=0,
    flatten_obs=True,
    gymnasium_api=True,
):
    if env_config.env_type == EnvType.POINT_MAZE:
        return make_maze_env(env_config, seed)()
    elif env_config.env_type == EnvType.FRANKA:
        return make_franka_env(env_config.robot_config, env_config, seed, device_id)
    elif env_config.env_type == EnvType.ROBOMIMIC:
        if env_config.data_collection:
            return make_robosuite_env(env_config, seed, device_id)()
        return make_robomimic_env(env_config, seed, device_id, flatten_obs=flatten_obs, gymnasium_api=gymnasium_api)()

def make_robomimic_env(
    env_config: RobomimicEnvConfig,
    seed=0,
    device_id=0,
    flatten_obs=True,
    gymnasium_api=True,
    obs_modality_specs=None,
):
    from sim_pipeline.envs.robosuite_wrapper import RobomimicObsWrapper
    from sim_pipeline.envs.initialization_wrapper import InitializationWrapper
        
    obs_type: ObsType = env_config.obs_type
    use_image_obs = obs_type.includes_image_obs()
    use_depth_obs = obs_type.includes_depth_obs()

    env_kwargs = {
        'has_offscreen_renderer': env_config.render_offscreen,
        'use_object_obs': True,
        'use_camera_obs': use_image_obs,
        'controller_configs': {'type': 'OSC_POSE', 'input_max': 1, 'input_min': -1, 'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5], 'output_min': [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], 'kp': 150, 'damping': 1, 'impedance_mode': 'fixed', 'kp_limits': [0, 300], 'damping_limits': [0, 10], 'position_limits': None, 'orientation_limits': None, 'uncouple_pos_ori': True, 'control_delta': True, 'interpolation': None, 'ramp_ratio': 0.2},
        'robots': ['Panda'],
        'camera_heights': 84,
        'camera_widths': 84,
        'reward_shaping': True,
        'render_gpu_device_id': device_id,
        'horizon': env_config.horizon,
        'camera_names': OmegaConf.to_container(env_config.camera_names),
    }
    
    if env_config.custom_obj_init:
        env_kwargs['placement_initializer'] = _get_init_randomizers(env_config)
        env_kwargs['initialization_noise'] = None

    def _init():
        env = RobomimicObsWrapper(
            env_name=env_config.env_name.value,
            render=env_config.render,
            render_offscreen=env_config.render_offscreen,
            use_image_obs=use_image_obs,
            use_depth_obs=use_depth_obs,
            flatten_obs=flatten_obs,
            env_config=env_config,
            gymnasium_api=gymnasium_api,
            obs_modality_specs=obs_modality_specs,
            seed=seed,
            **env_kwargs,
        )
        
        if env_config.custom_ee_init:
            robosuite = env.env
            init_wrapped_env = InitializationWrapper(robosuite, env_config.init_ee_range, seed=seed)
            env.env = init_wrapped_env
            
        if env_config.reward_function is not None:
            env = RewardWrapper(
                env, 
                env_config.reward_function.get_function(), 
                success_func=env_config.success_function.get_function(), 
                gymnasium_api=gymnasium_api
            )
            
        return env
    return _init

def make_robosuite_env(
    env_config: RobomimicEnvConfig,
    seed=0,
    device_id=0
):
    import robosuite as suite
    from robosuite import load_controller_config
    from sim_pipeline.envs.initialization_wrapper import InitializationWrapper

    controller_config = load_controller_config(default_controller='OSC_POSE')
    # Create argument configuration
    config = {
        "env_name": env_config.env_name.value,
        "robots": ['Panda'],
        "controller_configs": controller_config,
    }
    
    if env_config.custom_obj_init:
        config['placement_initializer'] = _get_init_randomizers(env_config)
        config['initialization_noise'] = None
        
    camera_names = OmegaConf.to_container(env_config.camera_names)
    
    def _init():
        # Create environment
        env = suite.make(
            **config,
            has_renderer=env_config.render,
            has_offscreen_renderer=env_config.render_offscreen,
            render_camera='agentview',
            ignore_done=env_config.ignore_done,
            use_camera_obs=env_config.render_offscreen,
            reward_shaping=True,
            control_freq=20,
            camera_names=camera_names,
            camera_heights=84,
            camera_widths=84,
        )
        if env_config.custom_ee_init:
            env = InitializationWrapper(env, env_config.init_ee_range, seed=seed)
            
        if env_config.reward_function is not None:
            env = RewardWrapper(
                env, 
                env_config.reward_function.get_function(), 
                success_func=env_config.success_function.get_function(), 
                gymnasium_api=False
            )    
            
        return env
    return _init

    
def _get_init_randomizers(env_config: RobomimicEnvConfig):
    from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
    
    placement_initializer = SequentialCompositeSampler(name='ObjectSampler')
    obj_names = env_config.env_name.get_object_names()
    
    
    if env_config.env_name in [RobomimicEnvType.NUT_ASSEMBLY_SQUARE, RobomimicEnvType.NUT_ASSEMBLY_ROUND, RobomimicEnvType.TOOL_HANG]:
        ensure_valid_placement = False
    else:
        ensure_valid_placement = True

    # NOTE: Change the rotation_axis to z (to be flat), tip, short_base, or long_base to initialize peg differently
    for obj_name in obj_names:
        
        if env_config.env_name == RobomimicEnvType.TOOL_HANG and obj_name == 'frameObject':
            rotation_axis = 'y'
        else:
            rotation_axis = 'z'
            
        if env_config.env_name == RobomimicEnvType.TOOL_HANG:
            z_offset = 0.001
        else:
            z_offset = 0.02
        
        try:
            init_range = env_config.init_obj_range[obj_name]
        except KeyError:
            init_range = env_config.env_name.get_default_obj_init_ranges()[obj_name]
        try:
            rotation_range = init_range[2]
        except IndexError:  
            rotation_range = None
        sampler = UniformRandomSampler(
            name=f'{obj_name}Sampler',
            x_range=init_range[0],
            y_range=init_range[1],
            ensure_object_boundary_in_range=False,
            rotation=rotation_range,
            rotation_axis=rotation_axis,
            # to avoid overlapping with the round nut (which doesn't exist here)
            ensure_valid_placement=ensure_valid_placement,
            reference_pos=env_config.env_name.get_surface_ref_pos(),
            z_offset=z_offset
        )
        
        placement_initializer.append_sampler(sampler)
        
    return placement_initializer

def make_franka_env(
    robot_config: RobotConfigFranka = None,
    env_config: EnvConfigFranka = None,
    seed=0,
    device_id=0
):
    from weird_franka.robot.robot_env import RobotEnv
    from weird_franka.robot.sim.mujoco.obj_wrapper import ObjWrapper
    
    robot_config.model_name = robot_config.model_name.replace(
            "base", env_config.obj_id
    )
    env = RobotEnv(**robot_config, device_id=device_id)
    env = ObjWrapper(env, **env_config)
    env = FrankaRewardWrapper(env, **env_config)

    env.seed(seed)

    return env

def make_maze_env(
    env_config: EnvConfigFranka,
    seed=0
):
    from .point_maze.point_maze import PointMazeEnv
    from .point_maze.ood_maze_envs import MazeType
    # print(f'Creating environment for {maze_id} maze ...')
    def _init():
        env = PointMazeEnv(
            maze_map=MazeType(env_config.maze_id).to_map(),
            render_mode=env_config.render_mode,
            continuing_task=False,
            max_episode_steps=env_config.max_ep_steps,
            reward_type=env_config.reward_type
        )
        env.reset(seed=seed)
        return env
    return _init