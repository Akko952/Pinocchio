import pinocchio as pin
import numpy as np
import os
import time
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as mg

def display_frames(viz, model, data):
    """
    Displays the coordinate frames of all joints in the visualizer.
    
    Args:
        viz (MeshcatVisualizer): The visualizer instance.
        model (pin.Model): The robot model.
        data (pin.Data): The robot data, after running forward kinematics.
    """
    frame_group_path = "pinocchio/frames"
    viz.viewer[frame_group_path].delete()

    for i in range(1, model.njoints):
        joint_name = model.names[i]
        joint_placement = data.oMi[i]

        frame_path = f"{frame_group_path}/{joint_name}"
        viz.viewer[frame_path].set_object(mg.triad(1))  # 设置坐标轴大小和线宽
        
        viz.viewer[frame_path].set_transform(joint_placement.np)


def run_visualization():
    # 1. 确保加载的是新的 URDF 文件
    urdf_filename = "3SPR.urdf" 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, urdf_filename)

    if not os.path.exists(urdf_path):
        print(f"错误: 找不到文件 {urdf_path}")
        return

    # 2. 加载模型
    model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path)
    data = model.createData()
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    
    try:
        viz.initViewer(open=True)
        viz.loadViewerModel()
    except Exception as e:
        print(f"Meshcat 初始化失败: {e}")
        return

    # 3. 设置初始姿态
    q = pin.neutral(model)

    # 初始计算和显示
    pin.forwardKinematics(model, data, q)
    viz.display(q)
    display_frames(viz, model, data)
    
    print("模型加载成功，关节坐标系已显示。")
    
    # 4. 简单动画 (伸缩 P 副)
    print("开始演示 P 副伸缩运动...")
    try:
        t = 0
        while True:
            motion =  1.5 * ( np.sin(t))+0.5  # 生成一个在 [0.5, 2.0] 范围内的周期性运动
            
            if model.existJointName("leg1_joint4"):
                q[model.joints[model.getJointId("leg1_joint4")].idx_q] = motion
            
            pin.forwardKinematics(model, data, q)
            viz.display(q)
            display_frames(viz, model, data)

            time.sleep(0.02)
            t += 0.05
    except KeyboardInterrupt:
        print("演示结束。")

if __name__ == '__main__':
    run_visualization()