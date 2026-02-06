"""
A script to collect a batch of human demonstrations.

The demonstrations can be played back using the `playback_demonstrations_from_hdf5.py` script.
"""

import argparse
import datetime
import json
import os
import time
from glob import glob

import h5py
import numpy as np
import robocasa
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.controllers.composite.composite_controller import CompositeController
from robosuite.controllers.composite import REGISTERED_COMPOSITE_CONTROLLERS_DICT

if "COMPOSITE" not in REGISTERED_COMPOSITE_CONTROLLERS_DICT:
    REGISTERED_COMPOSITE_CONTROLLERS_DICT["COMPOSITE"] = CompositeController

def convert_to_osc_config(original_config):
    """
    (T1 强力操控版 - High Power)
    
    解决“按键不动”的核心逻辑：
    1. kp = 500: 提供巨大的驱动力，克服 XML 里的 damping=5.0
    2. output_max = 0.2: 放宽速度限制，让它动得更快
    3. ref_name = "robot0_imu": 确保按 "W" 是向前，而不是向奇怪的方向
    """
    print(">>> 正在注入 T1 暴力配置 (强力操控版)...")
    
    gripper_config = {
        "type": "GRIP",
        "use_action_scaling": False,
        "interpolation": None,
        "input_type": "binary",
    }

    osc_arm_config = {
        "type": "OSC_POSE",
        
        # === 【核心修改 1】马力全开 ===
        # 既然 XML 里有阻尼(刹车)，我们就必须给大油门！
        # 原来 150 推不动，现在给 500
        "kp": 600,  
        
        # 保持适当的控制阻尼，防止超调
        "damping_ratio": 1.0, 
        
        "impedance_mode": "fixed",
        "kp_limits": [0, 1000], "damping_ratio_limits": [0, 10],
        
        "pose_error_type": "vector",
        "input_max": 1, "input_min": -1,
        
        # === 【核心修改 2】放宽限速 ===
        # 原来 0.05 太慢了，改成 0.2，让它能在大阻尼下也能挪动
        "output_max": [0.2, 0.2, 0.2, 1.0, 1.0, 1.0],
        "output_min": [-0.2, -0.2, -0.2, -1.0, -1.0, -1.0],
        
        "control_delta": True, "uncouple_pos_ori": True,
        
        # === 【核心修改 3】使用最直观的胸口参考系 ===
        # mobilebase0_center 可能不存在或方向不对
        # robot0_imu 就在胸口，跟着人走，方向最准
        "ref_name": "robot0_imu", 
        
        "gripper": gripper_config,
        "interpolation": None,
        "ramp_ratio": 0.2
    }

    # 身体锁定参数
    joint_body_config = {def convert_to_joint_action_config(original_config):
    """
    (修改版) 将 BASIC 配置转换为 JOINT_POSITION，并增加 right -> right_arm 的别名映射。
    """
    new_config = {
        "type": "COMPOSITE",
        "interpolation": None,
        "composite_controller_specific_configs": {}
    }

    body_parts = original_config.get('body_parts', {})

    for part_name, part_cfg in body_parts.items():
        
        # === 修改重点在这里 ===
        if part_name in ['right', 'left']:
            print(f"正在转换 {part_name} 为关节角度控制 (并添加 _arm 别名)...")
            
            # 1. 生成配置字典
            joint_config = {
                "controller_type": "JOINT_POSITION",
                "type": "JOINT_POSITION",
                "input_max": 1,
                "input_min": -1,
                'input_type': 'absolute',
                "output_max": 0.5,
                "output_min": -0.5,
                "kp": 200,  # 刚度
                'kv': 200,
                'velocity_limits': [-1, 1],
                'kp_limits': [0, 1000],
                "interpolation": None,
                "ramp_ratio": 0.2,
            }
            if 'gripper' in part_cfg:
                 joint_config['gripper'] = part_cfg['gripper']

            # 2. 【核心修改】写入字典时，同时写入原名和新名
            # 写入 'right' (为了解决 KeyError: 'right')
            new_config["composite_controller_specific_configs"][part_name] = joint_config
            
            # 写入 'right_arm' (满足您的改名需求)
            new_name = f"{part_name}_arm" 
            new_config["composite_controller_specific_configs"][new_name] = joint_config
            
            print(f"  -> 已生成键: '{part_name}' 和 '{new_name}'")

        # 其他部位保持不变
        elif part_cfg.get('type') == 'JOINT_POSITION':
            cfg_copy = part_cfg.copy()
            cfg_copy['controller_type'] = 'JOINT_POSITION'

            new_config["composite_controller_specific_configs"][part_name] = cfg_copy
            
        else:
            cfg_copy = part_cfg.copy()
            cfg_copy['controller_type'] = part_cfg['type']

            new_config["composite_controller_specific_configs"][part_name] = cfg_copy

    return new_config
        "type": "JOINT_POSITION",
        "kp": 500, "damping_ratio": 1,
        "interpolation": None,
        "ramp_ratio": 0.2
    }

    # === 构建配置 ===
    body_parts_dict = {}
    
    body_parts_dict["right"] = osc_arm_config
    
    body_parts_dict["left"] = osc_arm_config
    
    
    body_parts_dict["head"] = joint_body_config
    body_parts_dict["torso"] = joint_body_config
    body_parts_dict["legs"] = joint_body_config

    new_config = {
        "type": "COMPOSITE", 
        "body_parts": body_parts_dict
    }
    
    return new_config

def collect_human_trajectory(env, device, arm, max_fr, goal_update_mode):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        max_fr (int): if specified, pause the simulation whenever simulation runs faster than max_fr
    """

    env.reset()
    env.render()

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    for robot in env.robots:
        robot.print_action_info_dict()

    # Keep track of prev gripper actions when using since they are position-based and must be maintained when arms switched
    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    # Loop until we get a reset from the input or the task completes
    while True:
        start = time.time()

        # Set active robot
        active_robot = env.robots[device.active_robot]

        # Get the newest action
        input_ac_dict = device.input2action(goal_update_mode=goal_update_mode)

        

        # If action is none, then this a reset so we should break
        if input_ac_dict is None:
            break

        # === 【修复核心】手动构建动作向量 ===
        # T1 的动作空间是 维的，我们先创建一个全 0 的空动作
        
        final_action = np.zeros(26)
        # 键盘发 'right_delta'，我们把它填进机器人的 'right' (索引 0:6)
        if 'right_delta' in input_ac_dict:
            # print(f"DEBUG: 收到右手指令: {input_ac_dict['right_delta']}")  # 调试用，通了后可注释
            final_action[0:6] = input_ac_dict['right_delta']

        # --- 2. 接上“左手”的线 (如果需要) ---
        # 键盘发 'left_delta'，填进 'left' (索引 6:12)
        if 'left_delta' in input_ac_dict:
            final_action[6:12] = input_ac_dict['left_delta']

        # === 2. 暴力测试 (调试完请注释掉) ===
        # 强制给右手一个向上的速度！
        # 如果机器人自己往上飘，说明物理没问题，是按键没反应。
        # 如果还是纹丝不动，说明 XML 里的 armature 还是太大！
        #final_action[2] = 0.5  # <--- 取消注释这行来测试
    
        #    === 3. 打印调试 ===
        # 看看按键时有没有数据？
        if np.any(final_action):
            print(f"发送指令right: {final_action[0:6]}")
            print(f"发送指令left: {final_action[6:12]}")

        

        env.step(final_action)
        env.render()
        from copy import deepcopy

        action_dict = deepcopy(input_ac_dict)  # {}
        # set arm actions
        for arm in active_robot.arms:
            if isinstance(active_robot.composite_controller, WholeBody):  # input type passed to joint_action_policy
                controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
            else:
                controller_input_type = active_robot.part_controllers[arm].input_type

            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]
            else:
                raise ValueError

        # Maintain gripper state for each robot but only update the active robot with action
        env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
        env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
        env_action = np.concatenate(env_action)
        for gripper_ac in all_prev_gripper_actions[device.active_robot]:
            all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

        

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

        # limit frame rate if necessary
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    # cleanup for end of data collection episodes
    env.close()


def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
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

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
            success = success or dic["successful"]

        if len(states) == 0:
            continue

        # Add only the successful demonstration to dataset
        if success:
            print("Demonstration is successful and has been saved")
            # Delete the last state. This is because when the DataCollector wrapper
            # recorded the states and actions, the states were recorded AFTER playing that action,
            # so we end up with an extra state at the end.
            del states[-1]
            assert len(states) == len(actions)

            num_eps += 1
            ep_data_grp = grp.create_group("demo_{}".format(num_eps))

            # store model xml as an attribute
            xml_path = os.path.join(directory, ep_directory, "model.xml")
            with open(xml_path, "r") as f:
                xml_str = f.read()
            ep_data_grp.attrs["model_file"] = xml_str

            # write datasets for states and actions
            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))
        else:
            print("Demonstration is unsuccessful and has NOT been saved")

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(suite.models.assets_root, "demonstrations_private"),
    )
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--camera",
        nargs="*",
        type=str,
        default="agentview",
        help="List of camera names to use for collecting demos. Pass multiple names to enable multiple views. Note: the `mujoco` renderer must be enabled when using multiple views; `mjviewer` is not supported.",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be generic (eg. 'BASIC' or 'WHOLE_BODY_MINK_IK') or json file (see robosuite/controllers/config for examples)",
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="mjviewer",
        help="Use Mujoco's builtin interactive viewer (mjviewer) or OpenCV viewer (mujoco)",
    )
    parser.add_argument(
        "--max_fr",
        default=20,
        type=int,
        help="Sleep when simluation runs faster than specified frame rate; 20 fps is real time.",
    )
    parser.add_argument(
        "--reverse_xy",
        type=bool,
        default=False,
        help="(DualSense Only)Reverse the effect of the x and y axes of the joystick.It is used to handle the case that the left/right and front/back sides of the view are opposite to the LX and LY of the joystick(Push LX up but the robot move left in your view)",
    )
    parser.add_argument(
        "--goal_update_mode",
        type=str,
        default="target",
        choices=["target", "achieved"],
        help="Used by the device to get the arm's actions. The mode to update the goal in. Can be 'target' or 'achieved'. If 'target', the goal is updated based on the current target pose. "
        "If 'achieved', the goal is updated based on the current achieved state. "
        "We recommend using 'achieved' (and input_ref_frame='base') if collecting demonstrations with a mobile base robot.",
    )
    args = parser.parse_args()

    # Get controller config
    #controller_config = load_composite_controller_config(
     #   controller=args.controller,
      #  robot=args.robots[0],
    #)
    raw_config = load_composite_controller_config(controller="BASIC", robot=args.robots[0])
    #controller_config = convert_to_osc_config(raw_config)
    raw_config_left  = load_composite_controller_config(controller="BASIC", robot=args.robots[0])
    raw_config_right = load_composite_controller_config(controller="BASIC", robot=args.robots[0])
    # 生成两份完全独立的 OSC 配置
    osc_config_left  = convert_to_osc_config(raw_config_left)
    osc_config_right = convert_to_osc_config(raw_config_right)
    controller_config = {
        "type": "COMPOSITE",
        "body_parts": {
            # 左手拿左手的配置
            "left": osc_config_left["body_parts"]["left"],
            
            # 右手拿右手的配置 
            "right": osc_config_right["body_parts"]["right"],
            
            # 其他部位 (head, torso, legs) 
            "head":  osc_config_left["body_parts"]["head"],
            "torso": osc_config_left["body_parts"]["torso"],
            "legs":  osc_config_left["body_parts"]["legs"],
        }
    }

    if controller_config["type"] == "WHOLE_BODY_MINK_IK":
        # mink-speicific import. requires installing mink
        from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK

    # if WHOLE BODY IK; assert only one robot
    if controller_config["type"] == "WHOLE_BODY_IK":
        assert len(args.robots) == 1, "Whole Body IK only supports one robot"

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    
    print(">>> 当前已注册的复合控制器:", list(REGISTERED_COMPOSITE_CONTROLLERS_DICT.keys()))
    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        renderer=args.renderer,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "dualsense":
        from robosuite.devices import DualSense

        device = DualSense(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
            reverse_xy=args.reverse_xy,
        )
    elif args.device == "mjgui":
        assert args.renderer == "mjviewer", "Mocap is only supported with the mjviewer renderer"
        from robosuite.devices.mjgui import MJGUI

        device = MJGUI(env=env)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    # collect demonstrations
    while True:
        collect_human_trajectory(env, device, args.arm, args.max_fr, args.goal_update_mode)
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)
