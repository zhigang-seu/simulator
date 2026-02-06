"""
Bimanual HOLD stability test for T1 in Mujoco.
Zero-action test to verify if arms are stable under gravity.
Does NOT require human input, delta control or goal handling.
"""

import argparse
import time
import numpy as np
import robosuite as suite
import copy
from robosuite.wrappers import VisualizationWrapper
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import CompositeController
from robosuite.controllers.composite import REGISTERED_COMPOSITE_CONTROLLERS_DICT
from robosuite.scripts.collect_human_demonstrations import convert_to_osc_config

# 注册 COMPOSITE
if "COMPOSITE" not in REGISTERED_COMPOSITE_CONTROLLERS_DICT:
    REGISTERED_COMPOSITE_CONTROLLERS_DICT["COMPOSITE"] = CompositeController

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="PnPCupToDrawerClose")
    parser.add_argument("--robots", nargs="+", type=str, default=["T1"])
    parser.add_argument("--device", type=str, default="keyboard")  # 保留参数接口
    parser.add_argument("--camera", nargs="*", type=str, default=["robot0_agentview_center"])
    parser.add_argument("--renderer", type=str, default="mjviewer")
    parser.add_argument("--control_freq", type=int, default=20)
    args = parser.parse_args()

    # ---- Controller config ----
    raw_config = load_composite_controller_config(
        controller="BASIC",
        robot=args.robots[0],
    )
    #controller_config = convert_to_osc_config(raw_config)
    single_arm_cfg = convert_to_osc_config(raw_config)
    single_arm_cfg2 = convert_to_osc_config(raw_config)
    controller_config = {
        "type": "COMPOSITE",
        "body_parts": {
        "left":  single_arm_cfg["body_parts"]["left"],
        "right": single_arm_cfg2["body_parts"]["right"],

        # 这些你之前就有，照抄
        "head":  single_arm_cfg["body_parts"]["head"],
        "torso": single_arm_cfg["body_parts"]["torso"],
        "legs":  single_arm_cfg["body_parts"]["legs"],
    }
    }

    # ---- Environment ----
    env = suite.make(
        env_name=args.environment,
        robots=args.robots,
        controller_configs=controller_config,
        has_renderer=True,
        renderer=args.renderer,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=args.control_freq,
    )
    env = VisualizationWrapper(env)
    env.reset()
    env.render()

    robot = env.robots[0]

    print("=== Robot debug info ===")
    # 打印手臂信息
    print("=== Robot debug info ===")
    print("Arms:", robot.arms)
    
    print("Part controller config keys:", robot.part_controller_config.keys())
    # 打印每个 arm 的控制器 DOF 数量
    for arm in robot.arms:
        if arm in robot.part_controller_config:
            cfg = robot.part_controller_config[arm]
            print(f"{arm} controller config type:", cfg["type"])
            if "gripper" in cfg:
                print(f"{arm} gripper config keys:", cfg["gripper"].keys())

    # ---- Zero-action vector ----
    action_dim = env.action_spec[0].shape[0]
    zero_action = np.zeros(action_dim)

    print(">>> 测试开始：零动作下双臂稳定性")
    print(">>> 按 Ctrl+C 退出测试")

    try:
        while True:
            env.step(zero_action)
            env.render()
            time.sleep(1 / args.control_freq)
    except KeyboardInterrupt:
        print("\n[Exit] 测试结束")
        env.close()


if __name__ == "__main__":
    main()
