import inspect
from robosuite.models.robots import manipulators
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel

# 1. 抓出 T1 类
if hasattr(manipulators, "T1"):
    T1_Class = manipulators.T1
    print(f"找到 T1 类: {T1_Class}")
    
    # 2. 检查是否为抽象类 (这是最可能的凶手)
    is_abstract = inspect.isabstract(T1_Class)
    print(f"检查是否为抽象类 (isabstract): {is_abstract}")
    
    if is_abstract:
        print("!!! 罪魁祸首找到了:T1 被认为是抽象类 !!!")
        print("它还有以下方法没有实现（导致 Demo 不显示它）:")
        print(T1_Class.__abstractmethods__)
    
    # 3. 检查继承关系
    is_subclass = issubclass(T1_Class, ManipulatorModel)
    print(f"检查是否继承自 ManipulatorModel: {is_subclass}")

else:
    print("奇怪.manipulators 里找不到 T1,但这与你刚才的 dir() 矛盾。")