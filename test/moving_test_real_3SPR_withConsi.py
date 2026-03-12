import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
import os
import time
import sys

def create_3spr_simulation():
    print(">>> 步骤 1: 开始初始化仿真...")
    
    # 获取路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_filename = os.path.join(current_dir, "real_3SPR_copy.urdf")
    print(f">>> 检查 URDF 路径: {urdf_filename}")

    if not os.path.exists(urdf_filename):
        print(f"!!! 错误: 找不到 URDF 文件，请确认文件放在: {current_dir}")
        return

    # 加载模型
    print(">>> 步骤 2: 正在加载运动学模型...")
    try:
        model = pin.buildModelFromUrdf(urdf_filename)
        data = model.createData()
        print(f"成功加载模型，关节数量: {model.njoints}, 自由度: {model.nv}")
    except Exception as e:
        print(f"!!! 加载模型失败: {e}")
        return

    # 加载几何模型
    print(">>> 步骤 3: 正在解析视觉几何体 (Visual/Collision)...")
    visual_model = pin.buildGeomFromUrdf(model, urdf_filename, pin.GeometryType.VISUAL)
    collision_model = pin.buildGeomFromUrdf(model, urdf_filename, pin.GeometryType.COLLISION)
    print("成功解析几何模型。")

    # 启动 Meshcat
    print(">>> 步骤 4: 正在尝试启动 Meshcat 可视化器...")
    print("提示: 如果浏览器没有自动打开，请手动访问控制台打印的 URL 地址。")
    try:
        viz = MeshcatVisualizer(model, collision_model, visual_model)
        # 如果 open=True 没反应，可以尝试设为 False，然后手动打开 URL
        viz.initViewer(open=True) 
        viz.loadViewerModel("pinocchio")
        print(f"Meshcat 已启动。")
    except Exception as e:
        print(f"!!! Meshcat 启动失败 (可能是没安装 meshcat 库): {e}")
        return

    # 定义约束
    print(">>> 步骤 5: 正在配置闭环约束...")
    def add_6d_constraint(model, frame1_name, frame2_name):
        id1 = model.getFrameId(frame1_name)
        id2 = model.getFrameId(frame2_name)
        f1 = model.frames[id1]
        f2 = model.frames[id2]
        c_model = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_6D, model,
            f1.parentJoint, f1.placement,
            f2.parentJoint, f2.placement,
            pin.ReferenceFrame.LOCAL
        )
        c_model.corrector.Kp[:] = 1e6   # 大幅提高位置增益（从1000提高到100万）
        c_model.corrector.Kd[:] = 2000.0 # 提高阻尼系数 (通常设为 2 * sqrt(Kp))
        return c_model

    try:
        constraint_models = [
            add_6d_constraint(model, "leg2_tip_frame", "leg2_anchor_frame"),
            add_6d_constraint(model, "leg3_tip_frame", "leg3_anchor_frame")
        ]
        constraint_datas = [c.createData() for c in constraint_models]
        pin.initConstraintDynamics(model, data, constraint_models)
        print("闭环约束配置完成。")
    except Exception as e:
        print(f"!!! 约束配置失败 (请检查 Frame 名称): {e}")
        return

    # 初始状态
    q = np.array([0.0, 0.0, 0.0, 0.0, 0.0,  # Leg 1 的 5 个关节
                  0.0, 0.0, 0.0, 0.0, 0.0,  # Leg 2 的 5 个关节
                  0.0, 0.0, 0.0, 0.0, 0.0]) # Leg 3 的 5 个关节
    v = np.zeros(model.nv)
    tau = np.zeros(model.nv)
    viz.display(q)

    print("\n>>> 正在检查初始位姿的闭环一致性...")
    # 必须先更新运动学，否则获取的坐标是旧的
    pin.framesForwardKinematics(model, data, q)

    def check_closure(leg_num, tip_name, anchor_name):
        id_tip = model.getFrameId(tip_name)
        id_anc = model.getFrameId(anchor_name)
        # 获取两个点在世界坐标系下的位置
        p_tip = data.oMf[id_tip].translation
        p_anc = data.oMf[id_anc].translation
        error = np.linalg.norm(p_tip - p_anc)
        print(f"    Leg {leg_num} 闭环误差: {error:.6f} 米")
        return error

    err2 = check_closure(2, "leg2_tip_frame", "leg2_anchor_frame")
    err3 = check_closure(3, "leg3_tip_frame", "leg3_anchor_frame")

    if err2 > 0.01 or err3 > 0.01:
        print("!!! 警告: 初始闭环误差过大 (>1cm)，仿真可能会瞬间炸开飞走。")
        print("建议: 此时你应该使用位姿投影（IK）来修正 q，或者手动调整 q 的初值。")
    else:
        print(">>> 闭环检查通过（或误差极小），可以安全开始仿真。")
    # --- 检查结束 ---

    # 步骤 6: 进入仿真循环
    print(">>> 步骤 6: 进入仿真循环...")
    # ... 后续循环逻辑 ...
    dt = 0.000001
    prox_settings = pin.ProximalSettings(1e-12, 1e-14, 20)

     # 预先获取 Frame ID，避免在循环中重复查询，提高效率
    tid2 = model.getFrameId("leg2_tip_frame")
    aid2 = model.getFrameId("leg2_anchor_frame")
    tid3 = model.getFrameId("leg3_tip_frame")
    aid3 = model.getFrameId("leg3_anchor_frame")

    try:
        for i in range(10000):  # 运行一百万步
            # 计算动力学
            a = pin.constraintDynamics(
                model, data, q, v, tau, 
                constraint_models, constraint_datas, 
                prox_settings
            )

            # 积分
            v += a * dt
            q = pin.integrate(model, q, v * dt)
# 必须更新前向运动学才能计算最新的 Frame 位置
            pin.framesForwardKinematics(model, data, q)
                
                # 计算 Leg 2 误差
            p_tip3 = data.oMf[tid2].translation # Tip 的 [x, y, z]
            p_anc3 = data.oMf[aid2].translation # Anchor 的 [x, y, z]
    
    # 计算差值向量 (Vector3)
    # diff 是一个包含三个元素的数组: [dx, dy, dz]
            diff2 = p_tip3 - p_anc3 
    
            dx = diff2[0] # X 方向误差
            dy = diff2[1] # Y 方向误差
            dz = diff2[2] # Z 方向误差

    # 打印详细的方向误差
            print(f"步数: {i:6d} | Leg2 Error XYZ: [{dx:.6f}, {dy:.6f}, {dz:.6f}]")    
            # 显示
            viz.display(q)
            
            if i % 100 == 0:
                print(f"仿真进行中... 当前步数: {i}")
                print(np.linalg.norm(a))
            time.sleep(dt)
    except KeyboardInterrupt:
        print("仿真被用户手动停止。")
    except Exception as e:
        print(f"仿真运行出错: {e}")

    print(">>> 仿真任务结束。")

if __name__ == "__main__":
    # 强制刷新输出缓存，确保 print 能实时显示
    print("脚本已启动，正在初始化环境...", flush=True)
    create_3spr_simulation()