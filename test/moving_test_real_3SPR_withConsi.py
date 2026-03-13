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
    try:
        viz = MeshcatVisualizer(model, collision_model, visual_model)
        viz.initViewer(open=True) 
        viz.loadViewerModel("pinocchio")
        print(f"Meshcat 已启动。")
    except Exception as e:
        print(f"!!! Meshcat 启动失败: {e}")
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
        c_model.corrector.Kp[:] = 1e6   
        c_model.corrector.Kd[:] = 2000.0 
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
        print(f"!!! 约束配置失败: {e}")
        return

    # 初始状态定义
    # 假设初始状态下推杆位置为 0.0 (或者根据你的URDF设定)
    q = np.array([0.0, 0.0, 0.0, 0.0, 0.0,  # Leg 1
                  0.0, 0.0, 0.0, 0.0, 0.0,  # Leg 2
                  0.0, 0.0, 0.0, 0.0, 0.0]) # Leg 3
    v = np.zeros(model.nv)
    tau = np.zeros(model.nv)
    v_zero = np.zeros(model.nv)
    a_zero = np.zeros(model.nv)
    viz.display(q)

    # 闭环误差检查
    print("\n>>> 正在检查初始位姿的闭环一致性...")
    pin.framesForwardKinematics(model, data, q)
    tid2 = model.getFrameId("leg2_tip_frame")
    aid2 = model.getFrameId("leg2_anchor_frame")
    tid3 = model.getFrameId("leg3_tip_frame")
    aid3 = model.getFrameId("leg3_anchor_frame")

    def check_closure(leg_num, tid, aid):
        p_tip = data.oMf[tid].translation
        p_anc = data.oMf[aid].translation
        error = np.linalg.norm(p_tip - p_anc)
        print(f"    Leg {leg_num} 初始闭环误差: {error:.6f} 米")

    check_closure(2, tid2, aid2)
    check_closure(3, tid3, aid3)

    # ---------------- PID 控制器参数设置 ----------------
    actuator_indices = [3, 8, 13]  # 三条腿的主动推杆索引
    
    # PID 增益
    Kp = 80000.0   # 比例增益：负责将位置拉向目标
    Ki = 10000.0   # 积分增益：消除稳态误差（如重力导致的下沉）
    Kd = 3000.0    # 微分增益：提供阻尼，抑制抖动
    
    # 目标轨迹参数 (正弦运动)
    Amp =3    # 速度振幅 (m/s)
    freq = 1.0     # 频率 (Hz)
    omega = 2 * np.pi * freq
    
    # 初始参考位置 (作为正弦摆动的中点)
    q_start = q[actuator_indices].copy()
    # 累积积分误差
    error_integral = np.zeros(len(actuator_indices))

    # 仿真参数
    dt = 0.0001  # 1微秒步长
    prox_settings = pin.ProximalSettings(1e-12, 1e-14, 20)

    print("\n>>> 步骤 6: 进入仿真循环 (PID 控制模式)...")
    
    try:
        # 设置运行步数 (例如 100万步)
        for i in range(1000000):
            t = i * dt
            
            # 1. 计算当前时刻的目标位移 q_d 和 目标速度 v_d
            # v_d = Amp * sin(omega * t)
            # q_d = q_0 + Amp/omega * (1 - cos(omega * t))
            v_target = Amp * np.sin(omega * t)
            q_target = q_start + (Amp / omega) * (1.0 - np.cos(omega * t))

            # 2. 计算重力补偿
            tau_gravity = pin.rnea(model, data, q, v_zero, a_zero)
            
            # 3. 计算 PID 力矩
            tau = np.zeros(model.nv)
            for j, idx in enumerate(actuator_indices):
                # 计算位置和速度误差
                err_p = q_target[j] - q[idx]
                err_v = v_target - v[idx]
                
                # 更新积分项
                error_integral[j] += err_p * dt
                
                # PID 公式
                tau[idx] = (Kp * err_p) + (Ki * error_integral[j]) + (Kd * err_v) + tau_gravity[idx]

            # 4. 物理引擎步进
            a = pin.constraintDynamics(
                model, data, q, v, tau, 
                constraint_models, constraint_datas, 
                prox_settings
            )
            
            v += a * dt
            q = pin.integrate(model, q, v * dt)

            # 5. 降低监控和显示的频率 (每 1000 步处理一次，提升性能)
            if i % 1000 == 0:
                pin.framesForwardKinematics(model, data, q)
                
                # 计算 Leg 2 XYZ 方向误差
                diff2 = data.oMf[tid2].translation - data.oMf[aid2].translation 
                
                # 打印监控信息
                print(f"步: {i:6d} | 时间: {t:.3f}s | 推杆1位置: {q[3]:.4f} | Leg2 Error XYZ: [{diff2[0]:.6f}, {diff2[1]:.6f}, {diff2[2]:.6f}]")
                
                # 更新 Meshcat 显示
                viz.display(q)

    except KeyboardInterrupt:
        print("\n仿真被用户手动停止。")
    except Exception as e:
        print(f"\n仿真运行出错: {e}")

    print(">>> 仿真任务结束。")

if __name__ == "__main__":
    print("脚本已启动，正在初始化环境...", flush=True)
    create_3spr_simulation()