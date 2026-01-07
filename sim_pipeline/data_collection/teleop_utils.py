from collections import deque
import h5py
import datetime
import os
import numpy as np
import robosuite as suite
import uuid
import getpass
import torch
import robomimic.utils.tensor_utils as TensorUtils
import robosuite.utils.transform_utils as T

from scipy.spatial.transform import Rotation as R
from glob import glob
from robosuite.devices import *
from robosuite.models.robots import *
from robosuite.robots import *
from sim_pipeline.data_collection.spacemouse_hybrid import SpaceMouse as SpaceMouseHybrid
from sim_pipeline.data_collection.spacemouse_hybrid import ResetState
from sim_pipeline.data_collection.keyboard import Keyboard
from sim_pipeline.data_manager.dataset_metadata_enums import *

def input2action(device, robot, active_arm="right", env_configuration=None):
    """
    Converts an input from an active device into a valid action sequence that can be fed into an env.step() call

    If a reset is triggered from the device, immediately returns None. Else, returns the appropriate action

    Args:
        device (Device): A device from which user inputs can be converted into actions. Can be either a Spacemouse or
            Keyboard device class

        robot (Robot): Which robot we're controlling

        active_arm (str): Only applicable for multi-armed setups (e.g.: multi-arm environments or bimanual robots).
            Allows inputs to be converted correctly if the control type (e.g.: IK) is dependent on arm choice.
            Choices are {right, left}

        env_configuration (str or None): Only applicable for multi-armed environments. Allows inputs to be converted
            correctly if the control type (e.g.: IK) is dependent on the environment setup. Options are:
            {bimanual, single-arm-parallel, single-arm-opposed}

    Returns:
        2-tuple:

            - (None or np.array): Action interpreted from @device including any gripper action(s). None if we get a
                reset signal from the device
            - (None or int): 1 if desired close, -1 if desired open gripper state. None if get a reset signal from the
                device

    """
    state = device.get_controller_state()
    # Note: Devices output rotation with x and z flipped to account for robots starting with gripper facing down
    #       Also note that the outputted rotation is an absolute rotation, while outputted dpos is delta pos
    #       Raw delta rotations from neutral user input is captured in raw_drotation (roll, pitch, yaw)
    dpos, rotation, raw_drotation, grasp, reset = (
        state["dpos"],
        state["rotation"],
        state["raw_drotation"],
        state["grasp"],
        state["reset"],
    )
        
    # If we're resetting, immediately return None
    if isinstance(reset, ResetState) and reset == ResetState.SAVE or reset == ResetState.DISCARD:
        return reset, None
    elif not isinstance(reset, ResetState) and reset:
        return None, None

    # Get controller reference
    controller = robot.controller if not isinstance(robot, Bimanual) else robot.controller[active_arm]
    gripper_dof = robot.gripper.dof if not isinstance(robot, Bimanual) else robot.gripper[active_arm].dof

    # First process the raw drotation
    drotation = raw_drotation[[1, 0, 2]]
    if controller.name == "IK_POSE":
        # If this is panda, want to swap x and y axis
        if isinstance(robot.robot_model, Panda):
            drotation = drotation[[1, 0, 2]]
        else:
            # Flip x
            drotation[0] = -drotation[0]
        # Scale rotation for teleoperation (tuned for IK)
        drotation *= 10
        dpos *= 5
        # relative rotation of desired from current eef orientation
        # map to quat
        drotation = T.mat2quat(T.euler2mat(drotation))

        # If we're using a non-forward facing configuration, need to adjust relative position / orientation
        if env_configuration == "single-arm-opposed":
            # Swap x and y for pos and flip x,y signs for ori
            dpos = dpos[[1, 0, 2]]
            drotation[0] = -drotation[0]
            drotation[1] = -drotation[1]
            if active_arm == "left":
                # x pos needs to be flipped
                dpos[0] = -dpos[0]
            else:
                # y pos needs to be flipped
                dpos[1] = -dpos[1]

        # Lastly, map to axis angle form
        drotation = T.quat2axisangle(drotation)

    elif controller.name == "OSC_POSE":
        # Flip z
        drotation[2] = -drotation[2]
        # Scale rotation for teleoperation (tuned for OSC) -- gains tuned for each device
        drotation[0] = drotation[0] * 1.5 if isinstance(device, (Keyboard, SpaceMouseHybrid)) else drotation[0] * 50
        drotation[1] = drotation[1] * 1.5 if isinstance(device, (Keyboard, SpaceMouseHybrid)) else drotation[1] * 50
        drotation[2] = drotation[2] * 1.5 if isinstance(device, (Keyboard)) else drotation[2] * 50
        dpos = dpos * 75 if isinstance(device, (Keyboard)) else dpos * 125
    elif controller.name == "OSC_POSITION":
        dpos = dpos * 75 if isinstance(device, (Keyboard)) else dpos * 125
    else:
        # No other controllers currently supported
        print("Error: Unsupported controller specified -- Robot must have either an IK or OSC-based controller!")

    # map 0 to -1 (open) and map 1 to 1 (closed)
    grasp = 1 if grasp else -1

    # Create action based on action space of individual robot
    if controller.name == "OSC_POSITION":
        action = np.concatenate([dpos, [grasp] * gripper_dof])
    else:
        action = np.concatenate([dpos, drotation, [grasp] * gripper_dof])

    # Return the action and grasp
    return action, grasp


def collect_human_trajectory(
    env, 
    device, 
    arm, 
    env_configuration, 
    print_rew=False, 
    eval_model=None, 
    lowdim_keys=None, 
    rgb_keys=None, 
    value_function=None, 
    q_function=None,
    discriminator=False,
    n_obs_steps=2,
    record_image_embedding=False,
    use_dino_features=False,
) -> tuple[bool, bool]:
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
        
    Returns:
        whether or not data collection should continue, and whether the demonstration was successful
            (should be saved)
    """

    obs = env.reset()

    if use_dino_features:
        if 'object' in lowdim_keys or  'object-state' in lowdim_keys:
            # remove object from lowdim_keys
            lowdim_keys = [k for k in lowdim_keys if k != 'object' and k != 'object-state']
        
    # ID = 2 always corresponds to agentview
    env.render()

    is_first = True

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    device.start_control()
    last_action = None
    
    success = False
    num_actions = 0
    
    if eval_model is not None and not discriminator:
        obs_stack = deque(maxlen=n_obs_steps)
        # append the first obs twice to make sure the queue is full
        obs_stack.append(obs)
        obs_stack.append(obs)
    
    if record_image_embedding or use_dino_features:
        from robomimic.models.obs_core import DinoV2Core
        feature_model = DinoV2Core(input_shape=(3, 84, 84), backbone_name='dinov2_vits14', frozen=True, concatenate=True)
        feature_model.eval()
        model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        feature_model.to(model_device)
    if use_dino_features:
        if 'dino_features' not in obs:
            dino_features = run_feature_extractor(obs, feature_model, lowdim_keys, rgb_keys, normalize=False)
            obs['dino_features'] = dino_features

    render_positions = False
    image_embedding = None
    lowdim_embedding = None
    embedding_toggle = False
    action_queue = deque()
    # Loop until we get a reset from the input or the task completes
    while True:
        # Set active robot
        active_robot = env.robots[0] if env_configuration == "bimanual" else env.robots[arm == "left"]

        # Get the newest action
        action, grasp = input2action(
            device=device, robot=active_robot, active_arm=arm, env_configuration=env_configuration
        )
        # If action is none, then this a reset so we should break
        if action is None or (isinstance(action , ResetState) and (action == ResetState.DISCARD or action == ResetState.SAVE)):
            # record one last succesful step to make sure data is saved if required
            if action == ResetState.SAVE and last_action is not None:
                 state = env.env.sim.get_state().flatten()
                 env.states.append(state)
                 info = {}
                 info['actions'] = np.zeros_like(last_action)
                 env.action_infos.append(info)
                 env.successful = True
                 success = True
            break
        
        if device.hold_q:
            if q_function is not None:
                if use_dino_features:
                    q = run_value_function(obs, action, q_function, lowdim_keys + ['dino_features'], [], eval_batch=False)
                else:
                    q = run_value_function(obs, action, q_function, lowdim_keys, rgb_keys, eval_batch=False)
                print('Q: ', q)
            continue
        elif device.hold_value:
            if value_function is not None:
                if use_dino_features:
                    v = run_value_function(obs, None, value_function, lowdim_keys + ['dino_features'], [], eval_batch=False)
                else:
                    v = run_value_function(obs, None, value_function, lowdim_keys, rgb_keys, eval_batch=False)
                print('Value: ', v)
            continue
        elif device.execute_policy or device.eval_policy:
            if eval_model is not None and not discriminator:                    
                stacked_obs = {k: np.stack([obs[k] for obs in obs_stack], axis=0) for k in obs_stack[0].keys()} 
                    
                if use_dino_features:
                    actions = run_policy(stacked_obs, eval_model, lowdim_keys + ['dino_features'], [], num_actions=32)
                else:
                    actions = run_policy(stacked_obs, eval_model, lowdim_keys, rgb_keys, num_actions=64)
                actions = actions.detach().cpu().numpy()
                
                act_xyz = actions[:, :, :3]  # Shape (N, horizon, 3)
                current_eef_pos = obs['robot0_eef_pos'][:3]  # Shape (3,)

                all_positions = []
                # For each trajectory, compute the sequence of eef positions
                for traj_actions in act_xyz:  # traj_actions shape: (horizon, 3)
                    # Compute cumulative positions by adding delta actions
                    positions = np.zeros((len(traj_actions) + 1, 3))
                    positions[0] = current_eef_pos  # First position
                    for i in range(len(traj_actions)):
                        positions[i+1] = positions[i] + traj_actions[i] * 0.2                   
                    all_positions.append(positions)
                if use_dino_features:
                    q_vals = run_value_function(obs, actions[:, 0, :], q_function, lowdim_keys + ['dino_features'], [], eval_batch=True)
                else:
                    q_vals = run_value_function(obs, actions[:, 0, :], q_function, lowdim_keys, rgb_keys, eval_batch=True)

                render_positions = True
                    
                device_action = action
                if device.execute_policy:
                    if len(action_queue) == 0:
                        best_idx = np.argmax(q_vals.detach().cpu().numpy())
                        for action in actions[best_idx]:
                            action_queue.append(action)
                    action = action_queue.popleft()
                    action += device_action
        
        if device.recording_image_embedding:
            if not embedding_toggle:
                if image_embedding is None:
                    print('First embedding')
                    image_embedding = run_feature_extractor(obs, feature_model, lowdim_keys, rgb_keys)
                    print(image_embedding)
                    lowdim_embedding = get_lowdim_embedding(obs, lowdim_keys)
                    embedding_toggle = True
                else:
                    print('Second embedding')
                    second_embedding = run_feature_extractor(obs, feature_model, lowdim_keys, rgb_keys)
                    second_lowdim_embedding = get_lowdim_embedding(obs, lowdim_keys)
                    euclidean_distance = np.linalg.norm(image_embedding - second_embedding)
                    lowdim_euclidean_distance = np.linalg.norm(lowdim_embedding - second_lowdim_embedding)
                    print('Image Embedding distance: ', euclidean_distance)
                    print('Lowdim distance: ', lowdim_euclidean_distance)
                    # # compute cosine similarity
                    # cosine_similarity = np.dot(image_embedding, second_embedding) / (np.linalg.norm(image_embedding) * np.linalg.norm(second_embedding))
                    # print('Cosine similarity: ', cosine_similarity)
                    image_embedding = None
                    embedding_toggle = True
        else:
            if embedding_toggle:
                embedding_toggle = False
            
        # Run environment step
        obs, rew, _, _ = env.step(action)
        if use_dino_features:
            if 'dino_features' not in obs:
                dino_features = run_feature_extractor(obs, feature_model, lowdim_keys, rgb_keys, normalize=False)
                obs['dino_features'] = dino_features

        if eval_model is not None and not discriminator:
            obs_stack.append(obs)
        elif eval_model is not None and discriminator:
            output = run_policy(obs, eval_model, lowdim_keys, rgb_keys)
            print(output)

        last_action = action
        
        # print(obs['robot0_eef_pos'])
        
        if print_rew:
            print(rew)
        if render_positions:
            # Convert q_values to colors using a colormap
            q_values = q_vals.detach().cpu().numpy().flatten()
            # Normalize q_values to [0,1] range
            q_norm = (q_values - q_values.min()) / (q_values.max() - q_values.min())
            # Use blue (low) to red (high) colormap, with highest value black
            colors = np.zeros((len(q_values), 4))
            colors[:, 0] = q_norm  # Red channel
            colors[:, 2] = 1 - q_norm  # Blue channel
            colors[:, 3] = 0.2  # Alpha channel
            
            # Set highest q-value trajectory to green
            max_idx = np.argmax(q_values)
            colors[max_idx] = [0, 1, 0, 0.2]  # Green with same alpha
            
            for positions, color in zip(all_positions, colors):
                # visualize only after stepping to avoid putting the trajectory into the observation
                env.visualize_trajectory(points=positions, rgba=color, radius=0.005)
            render_positions = False
        env.render()
        env.clear_all_markers()

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            success = True
            break
        
        if device.terminate:
            env.close()
            return False, False

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

    # cleanup for end of data collection episodes
    env.close()
    
    return True, success

def get_lowdim_embedding(obs, lowdim_keys):
    lowdim_obs = [obs[k] for k in lowdim_keys]
    return np.concatenate(lowdim_obs)

def run_feature_extractor(obs, model, lowdim_keys, rgb_keys, normalize=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    obs_dict = {k: obs[k] for k in lowdim_keys + rgb_keys}
    features = []
    for k in sorted(rgb_keys):
        if 'image' in k:
            # flip H
            obs_dict[k] = obs_dict[k][::-1]
            obs_dict[k] = obs_dict[k].transpose(2, 0, 1) / 255.0
            ob = torch.from_numpy(obs_dict[k].copy())     
            ob = ob.unsqueeze(0)
            ob = ob.to(device)
            ob = ob.float()
            features.append(model(ob))
            
    # concatenate features
    features = torch.cat(features, dim=1).detach().cpu().numpy().flatten()
    if normalize:
        features = features / np.linalg.norm(features)
    return features

def run_value_function(obs, actions, model, lowdim_keys, rgb_keys, eval_batch=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    obs_dict = {k: obs[k] for k in lowdim_keys + rgb_keys}
    
    # replace 'object-state' with 'object' in obs
    if 'object-state' in obs_dict:
        obs_dict['object'] = obs_dict['object-state']
        del obs_dict['object-state']

    for k in obs_dict.keys():
        if obs_dict[k].ndim == 3:
            obs_dict[k] = obs_dict[k][::-1]
            obs_dict[k] = obs_dict[k].transpose(2, 0, 1)
        elif obs_dict[k].ndim == 4:
            # image is (T, H, W, C)
            # first, flip H
            obs_dict[k] = obs_dict[k][:, ::-1]
            # then, transpose to (T, C, H, W)
            obs_dict[k] = obs_dict[k].transpose(0, 3, 1, 2)
        # scale from (0, 255) to (0, 1)
        obs_dict[k] = obs_dict[k].astype(float) / 255.0
    ob = TensorUtils.to_tensor(obs_dict)
    ob = TensorUtils.to_batch(ob)
    if eval_batch:
        ob = {k: ob[k].repeat(actions.shape[0], *([1] * (len(ob[k].shape) - 1))) for k in ob}
    ob = TensorUtils.to_device(ob, device)
    ob = TensorUtils.to_float(ob)
    
    if actions is not None:
        actions = TensorUtils.to_tensor(actions)
        if not eval_batch:
            actions = TensorUtils.to_batch(actions)
        actions = TensorUtils.to_device(actions, device)
        actions = TensorUtils.to_float(actions)
                    
    if actions is not None:
        predictions = model(obs_dict=ob, acts=actions)
    else:
        predictions = model(obs_dict=ob)
    return predictions

def run_policy(obs, model, lowdim_keys, rgb_keys, num_actions=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    obs_dict = {k: obs[k] for k in lowdim_keys + rgb_keys}

    # replace 'object-state' with 'object' in obs
    if 'object-state' in obs_dict:
        obs_dict['object'] = obs_dict['object-state']
        del obs_dict['object-state']

    for k in obs_dict.keys():
        if 'image' in k:
            # flip H
            if obs_dict[k].ndim == 3:
                obs_dict[k] = obs_dict[k][::-1]
                obs_dict[k] = obs_dict[k].transpose(2, 0, 1)
            elif obs_dict[k].ndim == 4:
                # image is (T, H, W, C)
                # first, flip H
                obs_dict[k] = obs_dict[k][:, ::-1]
                # then, transpose to (T, C, H, W)
                obs_dict[k] = obs_dict[k].transpose(0, 3, 1, 2)
            # scale from (0, 255) to (0, 1)
            obs_dict[k] = obs_dict[k].astype(float) / 255.0
    ob = TensorUtils.to_tensor(obs_dict)
    ob = TensorUtils.to_batch(ob)
    
    # repeat the observation num_actions times
    ob = {k: ob[k].repeat(num_actions, *([1] * (len(ob[k].shape) - 1))) for k in ob}
    
    ob = TensorUtils.to_device(ob, device)
    ob = TensorUtils.to_float(ob)
    
    predictions = model.policy.get_action(ob)
    return predictions


def gather_demonstrations_as_hdf5(directory, out_dir, env_info, env_name, data_name):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, f"{data_name}.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0

    for ep_directory in os.listdir(directory):

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
            success = success or dic["successful"]

        if len(states) == 0:
            continue

        # Add only the successful demonstration to dataset
        if success:
            # print("Demonstration is successful and has been saved")
            # Delete the last state. This is because when the DataCollector wrapper
            # recorded the states and actions, the states were recorded AFTER playing that action,
            # so we end up with an extra state at the end.
            del states[-1]
            assert len(states) == len(actions)

            ep_data_grp = grp.create_group("demo_{}".format(num_eps))
            num_eps += 1

            # store model xml as an attribute
            xml_path = os.path.join(directory, ep_directory, "model.xml")
            with open(xml_path, "r") as f:
                xml_str = f.read()
            ep_data_grp.attrs["model_file"] = xml_str

            # write datasets for states and actions
            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))
        else:
            pass
            # print("Demonstration is unsuccessful and has NOT been saved")

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info
    grp.attrs["creator"] = getpass.getuser()

    grp.attrs['dataset_id'] = str(uuid.uuid4())
    grp.attrs['env_name'] = env_name
    grp.attrs['env_type'] = EnvType.ROBOSUITE.value
    grp.attrs['dataset_type'] = DatasetType.PLAY.value
    grp.attrs['action_type'] = ActionType.RELATIVE.value
    grp.attrs['is_real'] = False
    grp.attrs['is_test'] = data_name == 'test'

    f.close()
    
    return hdf5_path