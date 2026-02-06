import numpy as np
import robosuite as suite
from robosuite.models import arenas
from robosuite.models.tasks import Task
from robosuite.utils.placement_samplers import ObjectPositionSampler
from robosuite.controllers import load_composite_controller_config


# 1. 创建机器人
robot = suite.robots.T1Robot(idn=0)

# 2. 创建空场地
arena = arenas.EmptyArena()

# 3. 自定义一个极简任务（不放物体）
class EmptyTask(Task):
    def __init__(self, mujoco_arena, mujoco_robots):
        super().__init__(mujoco_arena, mujoco_robots)

    def _setup_references(self):
        super()._setup_references()

    def _reset_internal(self):
        pass

    def reward(self, action):
        return 0

    def _check_success(self):
        return False

    def _get_observations(self):
        # 返回最小观察字典
        return {"robot-state": robot.get_sensor_data()}

# 4. 组装环境
task = EmptyTask(arena, [robot])
env = suite.Environment(
    task,
    robots=[robot],
    controller_configs=load_controller_config(default_controller="JOINT_POSITION"),  # 根据你的控制器调整
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    render_camera=None,
    ignore_done=True,
    control_freq=20,
    horizon=500,
)

# 5. 测试运行
env.reset()
for i in range(100):
    action = np.random.randn(env.action_dim)  # 随机动作
    obs, reward, done, info = env.step(action)
    env.render()  # 显示窗口