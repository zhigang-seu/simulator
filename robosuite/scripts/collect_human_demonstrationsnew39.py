"""
A script to collect a batch of human demonstrations for T1 custom robot.
Architecture: Keyboard -> 6D Delta Pose -> Pose Integrator -> External IK -> JOINT_POSITION
"""

import argparse
import datetime
import json
import os
import time
from glob import glob

import h5py
import numpy as np
import scipy.spatial.transform as st  # 用于严谨的姿态积分计算

import robosuite as suite

import robocasa
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.scripts.t1_7dof_arm_ik2 import T17DofArmIK

# 导入你的真机 IK 求解器
#from t1_7dof_arm_ik2 import T17DofArmIK


def setup_joint_config(original_config):
    """
    踢开难调的 OSC_POSE，将控制器强行转为最底层的 JOINT_POSITION
    """
    import copy
    cfg = copy.deepcopy(original_config)

    for part_name, part_cfg in cfg["body_parts"].items():
        if part_name in ["left", "right"]:
            print(f"Converting {part_name} arm to JOINT_POSITION (True Teleop Mode)")

            saved_gripper = part_cfg.get("gripper", None)
            part_cfg.clear()
            part_cfg.update({
                "type": "JOINT_POSITION",
                "controller_type": "JOINT_POSITION",
                "interpolation": "linear",
                "kp": 150,
                "damping_ratio": 1.0,
                "ramp_ratio": 0.2,
            })

            if saved_gripper:
                part_cfg["gripper"] = saved_gripper
            else:
                part_cfg["gripper"] = {"type": "GRIPPER", "input_type": "delta", "kp": 0}

    return cfg


def collect_human_trajectory(env, device, arm, max_fr, goal_update_mode, args):
    """
    使用外部 IK 的遥操作采集循环
    """
    env.reset()
    env.render()

    task_completion_hold_count = -1
    device.start_control()

    for robot in env.robots:
        robot.print_action_info_dict()

    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    # ================= 真机 IK 求解器初始化 =================
    print("\n>>> 正在初始化 T1 真机 IK 求解器...")
    ik_solver = T17DofArmIK(visualization=False, unit_test=False)
    
    # 1. 直接从 MuJoCo 物理引擎读取当前全部关节的真实角度
    actual_qpos = env.sim.data.qpos.copy()
    
    # 2. 根据你的 T1 物理索引，提取真实的左右臂角度
    q_arm_actual = np.zeros(14)
    q_arm_actual[0:7] = actual_qpos[2:9]   # 物理模型中左臂的索引是 2 到 8
    q_arm_actual[7:14] = actual_qpos[9:16] # 物理模型中右臂的索引是 9 到 15
    
    # 3. 把这个真实姿态喂给 IK，覆盖掉它内置的“往上飘”默认姿态
    current_q = ik_solver.get_initial_joint_positions()
    current_q = ik_solver.set_arm_joints(current_q, q_arm_actual)
    
    # 4. 基于真实起点，计算遥操作的初始目标位姿
    left_pose_matrix = ik_solver.compute_forward_kinematics(current_q, 'left')
    right_pose_matrix = ik_solver.compute_forward_kinematics(current_q, 'right')
    
    left_target_pos = left_pose_matrix[:3, 3].copy()
    left_target_rot = left_pose_matrix[:3, :3].copy()
    right_target_pos = right_pose_matrix[:3, 3].copy()
    right_target_rot = right_pose_matrix[:3, :3].copy()
    print(">>> IK 求解器初始化完成，等待键盘输入...\n")
    # =======================================================

    # 手动定义移动步长（如果键盘移动太快，可以把这里改小）
    pos_step = 0.005 * args.pos_sensitivity
    rot_step = 0.02 * args.rot_sensitivity

    while True:
        start = time.time()
        active_robot = env.robots[device.active_robot]

        # 1. 安全获取底层键盘状态，彻底绕开 input2action
        state = device.get_controller_state()
        if state is None:
            break
            
        if isinstance(state, dict):
            dpos = state.get("dpos", np.zeros(3))
            drot = state.get("raw_drotation", np.zeros(3))
            grasp = state.get("grasp", 0)
            reset = state.get("reset", False)
        else:
            dpos, drot, grasp, reset = state

        if reset:
            break

        active_arm = device.active_arm
        has_ik_update = False

        # 2. 严谨的姿态积分与【方向重映射】
        if np.any(dpos != 0) or np.any(drot != 0):
            has_ik_update = True
            
            # 【修复方向反转】：Robosuite 默认按 W 键 dpos[0] 为 -1，这里翻转它
            dx = -dpos[0]  
            dy = dpos[1]   
            dz = dpos[2]   
            mapped_dpos = np.array([dx, dy, dz])
            
            if active_arm == "right":
                right_target_pos += mapped_dpos * pos_step
                delta_R = st.Rotation.from_euler('xyz', drot * rot_step).as_matrix()
                right_target_rot = delta_R @ right_target_rot
            elif active_arm == "left":
                left_target_pos += mapped_dpos * pos_step
                delta_R = st.Rotation.from_euler('xyz', drot * rot_step).as_matrix()
                left_target_rot = delta_R @ left_target_rot

        # 3. 外部 IK 求解
        if has_ik_update:
            target_left = np.eye(4)
            target_left[:3, :3] = left_target_rot
            target_left[:3, 3] = left_target_pos
            
            target_right = np.eye(4)
            target_right[:3, :3] = right_target_rot
            target_right[:3, 3] = right_target_pos

            q_arm, tau_ff, converged = ik_solver.solve_ik(target_left, target_right, current_q, visualize=False)
            current_q = ik_solver.set_arm_joints(current_q, q_arm)

        # 4. 【防止身体垮塌的终极 Action 映射】
        # 复制 MuJoCo 当前所有的控制指令（包含腿、头、躯干的当前发力状态）
        env_action = env.sim.data.ctrl.copy() 
        
        q_arm_extracted = ik_solver.extract_arm_joints(current_q)
        
        # 严格对齐你的控制通道槽位：左臂在 IK 是 0:7，在 Action 是 7:14
        env_action[7:14] = q_arm_extracted[0:7]   # 赋值左臂指令
        env_action[0:7] = q_arm_extracted[7:14]   # 赋值右臂指令

        # 5. 发送指令
        env.step(env_action)
        env.render()

        # --- 后续状态检测保持不变 ---
        if task_completion_hold_count == 0:
            break
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1
            else:
                task_completion_hold_count = 10
        else:
            task_completion_hold_count = -1

        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    env.close()


def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    # 此处保持原样，没有任何修改
    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")
    grp = f.create_group("data")

    num_eps = 0
    env_name = None

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

        if success:
            print("Demonstration is successful and has been saved")
            del states[-1]
            assert len(states) == len(actions)

            num_eps += 1
            ep_data_grp = grp.create_group("demo_{}".format(num_eps))

            xml_path = os.path.join(directory, ep_directory, "model.xml")
            with open(xml_path, "r") as f_xml:
                xml_str = f_xml.read()
            ep_data_grp.attrs["model_file"] = xml_str

            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))
        else:
            print("Demonstration is unsuccessful and has NOT been saved")

    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str,
                        default=os.path.join(suite.models.assets_root, "demonstrations_private"))
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--config", type=str, default="default",
                        help="Specified environment configuration if necessary")
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--camera", nargs="*", type=str, default="agentview", help="List of camera names")
    parser.add_argument("--controller", type=str, default="BASIC", help="Choice of controller.")
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    parser.add_argument("--renderer", type=str, default="mjviewer")
    parser.add_argument("--max_fr", default=20, type=int)
    parser.add_argument("--reverse_xy", type=bool, default=False)
    parser.add_argument("--goal_update_mode", type=str, default="target", choices=["target", "achieved"])
    args = parser.parse_args()

    # 1. 强制加载基础配置，并转换为关节控制模式
    controller_config = load_composite_controller_config(controller=args.controller)
    final_config = setup_joint_config(controller_config)

    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": final_config,
    }

    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

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


    env = VisualizationWrapper(env)
    env_info = json.dumps(config)

    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception("Invalid device choice: choose 'keyboard'.")

    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    while True:
        # 注意把 args 传进去，以获取敏感度参数
        collect_human_trajectory(env, device, args.arm, args.max_fr, args.goal_update_mode, args)
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)