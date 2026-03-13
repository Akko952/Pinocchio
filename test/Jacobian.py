import pinocchio as pin
import numpy as np
import os

def calculate_active_jacobian():
    # --- 1. 加载模型 ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_filename = os.path.join(current_dir, "real_3SPR_copy.urdf")
    model = pin.buildModelFromUrdf(urdf_filename)
    data = model.createData()

    # --- 2. 设定初始位姿 q ---
    # 提示：此处的 q 必须是满足闭环几何条件的，否则结果无物理意义
    q = pin.neutral(model) 
    q[3] = 0.0   # Leg 1 主动关节
    q[8] = 0.0   # Leg 2 主动关节   
    q[13] = 0.0  # Leg 3 主动关节
    # 其他关节保持在中立位置 (0.0)，确保满足闭环约束的几何条件
    # --- 3. 基础计算 ---
    pin.computeJointJacobians(model, data, q)
    pin.framesForwardKinematics(model, data, q)
    err2 = np.linalg.norm(data.oMf[model.getFrameId("leg2_tip_frame")].translation - 
                      data.oMf[model.getFrameId("leg2_anchor_frame")].translation)
    print(f"闭环误差: {err2}")
    # --- 4. 提取全雅可比 (J_ee: 6 x 15) ---
    frame_id = model.getFrameId("motion_link")
    J_ee = pin.getFrameJacobian(model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    # --- 5. 构造约束雅可比 (J_c: 12 x 15) ---
    # 定义闭环点
    constraints = [
        ("leg2_tip_frame", "leg2_anchor_frame"),
        ("leg3_tip_frame", "leg3_anchor_frame")
    ]
    Jc_list = []
    for tip, anc in constraints:
        # 相对雅可比 = Tip雅可比 - Anchor雅可比
        J_tip = pin.getFrameJacobian(model, data, model.getFrameId(tip), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_anc = pin.getFrameJacobian(model, data, model.getFrameId(anc), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        Jc_list.append(J_tip - J_anc)
    Jc = np.vstack(Jc_list) # 12 x 15

    # --- 6. 矩阵拆分 ---
    # 主动副索引 (根据你的 Joint List: 3, 8, 13)
    active_idx = [3, 8, 13]
    # 从动副索引 (剩下的 12 个)
    passive_idx = [i for i in range(model.nv) if i not in active_idx]

    # 拆分 J_ee
    J_ea = J_ee[:, active_idx]  # 6 x 3
    J_ep = J_ee[:, passive_idx]  # 6 x 12

    # 拆分 J_c
    J_ca = Jc[:, active_idx]    # 12 x 3
    J_cp = Jc[:, passive_idx]    # 12 x 12

    # --- 7. 消元计算 J_active ---
    # 公式: J_active = J_ea - J_ep * (J_cp^-1 * J_ca)
    # 使用最小二乘解 (pinv) 以提高数值稳定性
    try:
        J_active = J_ea - J_ep @ np.linalg.solve(J_cp, J_ca)
    except np.linalg.LinAlgError:
        # 如果 J_cp 奇异，说明处于机构奇异位形
        J_active = J_ea - J_ep @ np.linalg.pinv(J_cp) @ J_ca

    # --- 8. 输出结果 ---
    np.set_printoptions(precision=4, suppress=True)
    print("主动关节雅可比矩阵 (6x3):")
    print(J_active)
    
    print("\n[验证]")
    print(f"该矩阵描述了: [vx, vy, vz, wx, wy, wz]^T = J_active * [d1_dot, d2_dot, d3_dot]^T")

if __name__ == "__main__":
    calculate_active_jacobian()