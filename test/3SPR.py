import pinocchio as pin
import numpy as np
import os
import time
from pinocchio.visualize import MeshcatVisualizer

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
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    
    try:
        viz.initViewer(open=True)
        viz.loadViewerModel()
    except Exception as e:
        print(f"Meshcat 初始化失败: {e}")
        return

    # 3. 设置初始姿态
    q = pin.neutral(model)

    # === 关键修正：关节名称映射 ===
    # 在 3SPR_Exact.urdf 中:
    # Joint 1,2,3 = 球铰 (S)
    # Joint 4     = 移动副 (P, Active)
    # Joint 5     = 转动副 (R, Passive)
    
    # 现在的 URDF 是基于精确旋量生成的，意味着：
    # 当所有关节角度 q=0 时，机器人的几何形状就是初始设计状态。
    # 我们 不需要 像之前那样手动旋转 Joint 5 来调平平台，
    # 因为 Leg 1 的 Joint 5 (R副) 的 origin 已经直接指向了 B1 点。
    # 理论上 q=0 时，平台就是平的。
    
    print("模型加载成功。基于精确旋量建模，初始 q=0 应当对应水平平台。")
    
    viz.display(q)

    # 4. 简单动画 (伸缩 P 副)
    print("开始演示 P 副伸缩运动...")
    try:
        t = 0
        while True:
            motion = 0.5 * np.sin(t)
            
            # 使用新的关节名称: joint4 是 P 副
            if model.existJointName("leg1_joint4"):
                q[model.joints[model.getJointId("leg1_joint4")].idx_q] = motion
            if model.existJointName("leg2_joint4"):
                q[model.joints[model.getJointId("leg2_joint4")].idx_q] = motion
            if model.existJointName("leg3_joint4"):
                q[model.joints[model.getJointId("leg3_joint4")].idx_q] = motion
            
            viz.display(q)
            time.sleep(0.02)
            t += 0.05
    except KeyboardInterrupt:
        print("演示结束。")

if __name__ == '__main__':
    run_visualization()