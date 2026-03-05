import example_robot_data
import pinocchio as pin
import numpy as np
from pinocchio.visualize import MeshcatVisualizer

# ====================== 1. 加载机器人模型 ======================
robot = example_robot_data.load('solo12')  # 可替换为 'solo12', 'icub' 等
print("✅ 机器人模型加载成功！")
print("机器人名称:", robot.model.name)
print("关节数量 (nq):", robot.model.nq)
print("速度/力维度 (nv):", robot.model.nv)
print("模型包含的帧数:", robot.model.nframes)
print("-" * 50)

# ====================== 2. 随机生成关节配置 ======================
q0 = pin.randomConfiguration(robot.model)  # 随机关节位置
print("🎲 随机生成的关节配置 (前10个元素):")
print(q0[:10] if len(q0) > 10 else q0)
print("-" * 50)

# ====================== 3. 正向运动学：尝试获取脚底 frame 位姿 ======================
candidate_foot_names = ['left_sole', 'left_sole_link', 'l_sole', 'left_foot', 'l_foot']
foot_frame_id = None
for name in candidate_foot_names:
    if robot.model.existFrame(name):
        foot_frame_id = robot.model.getFrameId(name)
        print(f"🔍 找到脚底 frame: '{name}', ID = {foot_frame_id}")
        break

if foot_frame_id is not None:
    pin.forwardKinematics(robot.model, robot.data, q0)
    pin.updateFramePlacements(robot.model, robot.data)
    foot_placement = robot.data.oMf[foot_frame_id]
    print("🦶 左脚位置 (相对于世界坐标系):")
    print("    平移:", foot_placement.translation.T)
    print("    旋转矩阵:\n", foot_placement.rotation)
else:
    print("⚠️ 未找到预设的脚底 frame 名称，以下为模型中所有 frame：")
    for i, frame in enumerate(robot.model.frames):
        print(f"  {i}: {frame.name}")
    print("请从上述列表中选择正确的 frame 名称，并修改代码中的 `candidate_foot_names`。")
print("-" * 50)

# ====================== 4. 计算动力学量 ======================
M = pin.crba(robot.model, robot.data, q0)
print("📊 质量矩阵形状:", M.shape)
print("质量矩阵范数:", np.linalg.norm(M))

g = pin.computeGeneralizedGravity(robot.model, robot.data, q0)
print("重力项 (前5个元素):", g[:5])
print("-" * 50)

# ====================== 5. meshcat 可视化 ======================
print("🌐 正在初始化 meshcat 可视化器...")
# 创建 Pinocchio 的 MeshcatVisualizer 实例
viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)

# 启动可视化服务器并加载模型
try:
    viz.initViewer(loadModel=True)  # 自动创建 viewer 并加载几何体
    print("✅ 可视化服务器已启动。")
    print("如果浏览器未自动打开，请手动访问 http://127.0.0.1:7000/static/")
except Exception as e:
    print(f"❌ 可视化初始化失败: {e}")
    print("请确保 meshcat 已正确安装，或尝试重新运行。")
    exit(1)

# 将可视化器绑定到机器人对象
robot.setVisualizer(viz)  # 这会设置 robot.viz 属性

# 显示当前随机配置
robot.display(q0)
print("🤖 机器人模型已显示在可视化窗口中。")

# 保持程序运行，直到用户手动中断
print("\n按 Ctrl+C 退出程序...")
try:
    input("按回车键退出...")
except KeyboardInterrupt:
    print("程序已退出。")