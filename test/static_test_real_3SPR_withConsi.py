import pinocchio as pin
import numpy as np
import os
import time
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as mg 

def display_all_frames(viz, model, data, axis_scale=1.0):
    """
    遍历并显示模型中所有的坐标系
    """
    for i, frame in enumerate(model.frames):
        # 排除掉自带的默认无意义 frame (如 OP_FRAME)
        if "universe" in frame.name:
            continue
            
        frame_path = f"visualizer/frames/{frame.name}"
        
        # 设置巨大的坐标轴 (triad)
        # axis_scale=1.0 已经非常大，可以根据需要调至 1.5 或 2.0
        viz.viewer[frame_path].set_object(mg.triad(axis_scale))
        
        # 更新位置
        viz.viewer[frame_path].set_transform(data.oMf[i].np)

def run_3spr_static_visual():
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
    viz.initViewer(open=True)
    viz.loadViewerModel()

    # 2. 获取 IK 所需 ID
    tip_ids = [model.getFrameId(f"leg{i}_tip") for i in [2, 3]]
    anchor_ids = [model.getFrameId(f"leg{i}_anchor") for i in [2, 3]]
    p_jids = [model.getJointId(f"leg{i}_joint4") for i in [1, 2, 3]]
    q_p_idx = [model.joints[jid].idx_q for jid in p_jids]
    v_p_idx = [model.joints[jid].idx_v for jid in p_jids]

    # 3. IK 解算器 (确保初始位置完美闭合)
    def solve_ik(q_prev, target_lengths):
        q_curr = q_prev.copy()
        for i in range(3):
            q_curr[q_p_idx[i]] = target_lengths[i]

        for i in range(100):
            pin.forwardKinematics(model, data, q_curr)
            pin.updateFramePlacements(model, data)

            errs = []
            for tip_id, anchor_id in zip(tip_ids, anchor_ids):
                p_err = data.oMf[tip_id].translation - data.oMf[anchor_id].translation
                errs.append(p_err)
            
            error = np.concatenate(errs)
            if np.linalg.norm(error) < 1e-8: break

            J2 = pin.getFrameJacobian(model, data, tip_ids[0], pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
            J3 = pin.getFrameJacobian(model, data, tip_ids[1], pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
            J = np.vstack([J2, J3])

            mask = np.ones(model.nv)
            for v_idx in v_p_idx: mask[v_idx] = 0
            
            damp = 1e-4
            J_masked = J[:, mask == 1]
            v_reduced = - J_masked.T @ np.linalg.solve(J_masked @ J_masked.T + damp * np.eye(6), error)
            
            v_full = np.zeros(model.nv)
            v_full[mask == 1] = v_reduced
            q_curr = pin.integrate(model, q_curr, v_full * 0.5)

        return q_curr

    # 4. 设定初始 0 位 (P副位移为 0)
    # 此时腿长等于 URDF 里定义的 7.7735
    q_initial = pin.neutral(model)
    q_final = solve_ik(q_initial, [0.0, 0.0, 0.0])

    print("--- 0位静态显示 ---")
    print("坐标轴比例: 1.0 (巨大)")
    
    try:
        while True:
            # 持续更新运动学状态
            pin.forwardKinematics(model, data, q_final)
            pin.updateFramePlacements(model, data)

            # 显示机器人
            viz.display(q_final)
            
            # 显示所有坐标系 (Scale 设为 1.0, 轴长1米)
            display_all_frames(viz, model, data, axis_scale=1.0)

            time.sleep(0.5) # 静态显示，降低 CPU 占用
            
    except KeyboardInterrupt:
        print("停止。")

if __name__ == "__main__":
    run_3spr_static_visual()