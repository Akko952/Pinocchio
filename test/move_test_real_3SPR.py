import pinocchio as pin
import numpy as np
import os
import time
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as mg

def display_frames(viz, model, data):
    """在可视化界面显示所有关节坐标系"""
    frame_group_path = "pinocchio/frames"
    # 第一次运行建议不删除，或者频率降低，否则会闪烁
    
    for i in range(1, model.njoints):
        joint_name = model.names[i]
        joint_placement = data.oMi[i]
        frame_path = f"{frame_group_path}/{joint_name}"
        viz.viewer[frame_path].set_object(mg.triad(0.5)) # 坐标轴大小 0.5
        viz.viewer[frame_path].set_transform(joint_placement.np)

def run_3spr_simulation():
    # 1. 加载模型
    urdf_filename = "real_3SPR.urdf" 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, urdf_filename)

    if not os.path.exists(urdf_path):
        print(f"错误: 找不到文件 {urdf_path}")
        return

    model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path)
    data = model.createData()
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    
    try:
        viz.initViewer(open=True)
        viz.loadViewerModel()
    except Exception as e:
        print(f"Meshcat 初始化失败: {e}")
        return

    # 2. 获取三个 P 副的关节 ID 和在 q 中的索引
    # 在 URDF 中，P 副的名字分别是 leg1_joint4, leg2_joint4, leg3_joint4
    p_joint_names = ["leg1_joint4", "leg2_joint4", "leg3_joint4"]
    p_indices = []
    
    for name in p_joint_names:
        if model.existJointName(name):
            joint_id = model.getJointId(name)
            # q 向量中的起始索引
            idx_q = model.joints[joint_id].idx_q
            p_indices.append(idx_q)
            print(f"找到关节 {name}, q 索引为: {idx_q}")
        else:
            print(f"警告: 未找到关节 {name}")

    # 3. 初始姿态
    q = pin.neutral(model)
    
    print("\n开始演示 3-SPR 协调运动...")
    print("注意：由于是树状 URDF，Leg 2 和 Leg 3 的运动不会直接拉扯平台。")
    
    try:
        t = 0
        while True:
            # 设计三种不同的运动模式
            
            # Leg 1: 基准正弦运动
            # val1 = 0.5 * np.sin(t) + 0.5  # 范围 [0, 1.0]
            
            # # Leg 2: 相位差 120 度 (2π/3)
            # val2 = 0.5 * np.sin(t + 2*np.pi/3) + 0.5
            
            # # Leg 3: 相位差 240 度 (4π/3)
            # val3 = 0.5 * np.sin(t + 4*np.pi/3) + 0.5
            val1=1.5 * np.sin(t) + 0.5
            val2=1.5 * np.sin(t) + 0.5
            val3=1.5 * np.sin(t) + 0.5

            # 更新 q 向量
            if len(p_indices) >= 3:
                q[p_indices[0]] = val1
                q[p_indices[1]] = val2
                q[p_indices[2]] = val3
            
            # 更新动力学并显示
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data) # 确保所有 Frame 也被更新
            
            viz.display(q)
            display_frames(viz, model, data)

            time.sleep(0.02)
            t += 0.05
            
    except KeyboardInterrupt:
        print("演示结束。")

if __name__ == '__main__':
    run_3spr_simulation()