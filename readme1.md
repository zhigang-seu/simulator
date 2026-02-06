这个是在https://github.com/ARISE-Initiative/robosuite和https://github.com/robocasa/robocasa-gr1-tabletop-tasks的基础上做的
第一步：换机器人
首先需要更换机器人模型：
1、下载mujoco3.0.0 地址https://github.com/google-deepmind/mujoco/releases/tag/3.0.0
mkdir ~/.mujoco
tar -zxvf mujoco-3.0.0-linux-x86_64.tar.gz -C ~/.mujoco
2、参考https://blog.csdn.net/baidu_41800370/article/details/139014017?ops_request_misc=%257B%2522request%255Fid%2522%253A%252256ff32fd69e6ffc26bfedd6b13cdc05e%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=56ff32fd69e6ffc26bfedd6b13cdc05e&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-139014017-null-null.142^v102^control&utm_term=urdf%20mujoco&spm=1018.2226.3001.4187
生成xml文件
3、修改STL文件路径 meshdir="meshes"
添加材质 <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0.02" />
添加地板 <geom name="floor" pos="0 0 -0.97" size="1 1 0.02" type="plane" material="grid" />
碰撞体  contype="1" conaffinity="1"
4、其余项：插件转化的xml需检查修改，例如初次转换后Trunck未被放入body项

T1机器人修改相关：
首先修改了xml。利用mujoco把urdf变成xml后
添加了 `<option integrator="implicitfast"/>`。
添加了 `<default>` 段落，设定了 `frictionloss="0.2"`, `armature="0.05"`, `damping="0.5"`。
 在 `<compiler>` 中开启了 `autolimits="true"`，这有助于 MuJoCo 更好地处理关节限位。
新增了一个主躯干 Body 命名为 **`Trunk`** (对应 mesh `Trunk.STL`)。头部、手臂、腰部（连接腿）全部变成了 `Trunk` 的**子节点**。
 在 `Trunk` 中显式定义了整机的惯性参数 (`inertial`)，质量设为 11.7kg。
添加了Sites (标记点)
底部新增 `<actuator>` 模块，定义了所有关节的 `motor`
在 `robot.xml` 的 `<joint>` 定义中，显式添加了 `actuatorfrcrange`

然后关于T1机器人的替换部分，设置比较杂乱。https://github.com/zhigang-seu/simulator.git这个仓库里面已经成功在robosuite里面载入T1机器人。
第二步：具体使用的遥操作脚本是collect_human_demonstrationsnew.py
上传的应该是有四个类似的类似的脚本，有一个是后缀写了配合零动作脚本使用的。零动作是叫test_bimanual_hold.py,目的是为了解决xml抖动问题
剩下两个暂时还不能使用。
对应的操作指令是：
python -m robosuite.scripts.     --environment PnPCupToDrawerClose     --robots T1     --device keyboard     --camera robot0_agentview_center
运行这个指令之前，请先激活对应的环境。