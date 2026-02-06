import numpy as np
from robosuite.models.robots.manipulators.legged_manipulator_model import LeggedManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion

# 注意：不需要 register_robot_class，元类会自动注册
class T1(LeggedManipulatorModel):
    """
    T1 Robot Custom Definition
    (Strictly aligned with t1.xml provided by user)
    """
    arms = ["left", "right"]

    def __init__(self, idn=0):
        
        xml_path = xml_path_completion("robots/t1/robot.xml")
        print(f"\n\n======== [T1 Debug] 正在加载 XML: {xml_path} ========")
        # 加载 XML
        super().__init__(xml_path_completion("robots/t1/robot.xml"), idn=idn)
        
        # === 移除 FreeJoint，把机器人钉在空中 ===
        self._remove_free_joint()

    @property
    def default_base(self):
        return "NoActuationBase"

    @property
    def default_gripper(self):
        return {
            "right": None,#"FourierRightHand", 
            "left": None,#"FourierRightHand"
        }

    @property
    def default_controller_config(self):
        return {
            "right": "t1_arm_controller",
            "left": "t1_arm_controller",
            "head": "t1_body_controller",
            "torso": "t1_body_controller",
            "right_leg": "t1_body_controller",
            "left_leg": "t1_body_controller"
        }

    @property
    def init_qpos(self):
        """
        修正版初始姿态：
        让机器人做出“抱拳”或“开车”的姿态（手抬高放在胸前），
        而不是垂在下面（会撞桌子）。
        """
        init_qpos = np.zeros(29)

        # === 手臂初始姿态 (抬起手，避免撞桌子) ===
        # Joint 0 (Shoulder Pitch): -0.6 (向前抬起约35度)
        # Joint 2 (Elbow Pitch): -1.5 (弯曲手肘约90度)
        
        # 左臂配置 (Left)
        # 顺序: Pitch, Roll, Elbow_Pitch, Elbow_Yaw, Wrist_Pitch, Wrist_Yaw, Hand_Roll
        #left_arm_init  = np.array([-1.57, -1.57, 0.0, 0.0, 0.0, 0.0, 0.0])
        left_arm_init  = np.array([0.45, -1.05, 0.5, -1.5, 0.0, 0.0, 0.0])
        # 右臂配置 (Right) - 保持对称
        right_arm_init = np.array([0.45, 1.05, 0.5, 1.5, 0.0, 0.0, 0.0])
        #right_arm_init = np.array([-1.57, 1.57, 0.0, 0.0, 0.0, 0.0, 0.0])
        #right_arm_init = np.array([0, 0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # === 赋值 (严格按 XML 顺序) ===
        # 1. Head (Index 0-1) -> 保持 0
        
        # 2. Left Arm (Index 2-8) -> 填入左臂数据
        init_qpos[2:9] = left_arm_init
        
        # 3. Right Arm (Index 9-15) -> 填入右臂数据
        init_qpos[9:16] = right_arm_init
        
        return init_qpos

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.30, -0.1, 1.15),
            "empty": (-0.29, 0, 1.15),
            
            # === 【关键修改】抬高身体 ===
            # 原来是 1.05，现在改为 1.20
            # 这样躯干更高，手就不容易插进桌子里了
            "table": lambda table_length: (-0.15 - table_length / 2, 0, 1.20),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "bimanual"

    @property
    def _eef_name(self):
        return {
            "right": "right_tip",
            "left": "left_tip"
        }




    # ----------------------------
# Optional Variants (Fixed Legs, Floating, Arms Only)
# ----------------------------

class T1FixedLowerBody(T1):
    def __init__(self, idn=0):
        super().__init__(idn=idn)
        # Fix lower body: remove leg actuation and free joint
        self._remove_joint_actuation("leg")
        self._remove_free_joint()

    @property
    def init_qpos(self):
        # Only upper body DOFs (e.g., 20 = 6 base + 14 arms)
        init_qpos = np.zeros(20)
        right_arm_init = np.array([0.0, -0.1, 0.0, -1.57, 0.0, 0.0, 0.0])
        init_qpos[6:13] = right_arm_init
        left_arm_init = np.array([0.0, 0.1, 0.0, -1.57, 0.0, 0.0, 0.0])
        init_qpos[13:20] = left_arm_init
        return init_qpos

    @property
    def default_base(self):
        return "NoActuationBase"


class T1FloatingBody(T1):
    def __init__(self, idn=0):
        super().__init__(idn=idn)
        self._remove_joint_actuation("leg")
        self._remove_free_joint()

    @property
    def init_qpos(self):
        init_qpos = np.zeros(20)
        right_arm_init = np.array([0.0, -0.1, 0.0, -1.57, 0.0, 0.0, 0.0])
        init_qpos[6:13] = right_arm_init
        left_arm_init = np.array([0.0, 0.1, 0.0, -1.57, 0.0, 0.0, 0.0])
        init_qpos[13:20] = left_arm_init
        return init_qpos

    @property
    def default_base(self):
        return "FloatingLeggedBase"

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.30, -0.1, 0.97),
            "empty": (-0.29, 0, 0.97),
            "table": lambda table_length: (-0.15 - table_length / 2, 0, 0.97),
        }


class T1ArmsOnly(T1):
    def __init__(self, idn=0):
        super().__init__(idn=idn)
        self._remove_joint_actuation("leg")
        self._remove_joint_actuation("head")
        self._remove_joint_actuation("torso")
        self._remove_free_joint()

    @property
    def init_qpos(self):
        # Only arm DOFs (14 = 7 right + 7 left)
        init_qpos = np.zeros(14)
        right_arm_init = np.array([0.0, -0.1, 0.0, -1.57, 0.0, 0.0, 0.0])
        left_arm_init = np.array([0.0, 0.1, 0.0, -1.57, 0.0, 0.0, 0.0])
        init_qpos[0:7] = right_arm_init
        init_qpos[7:14] = left_arm_init
        return init_qpos