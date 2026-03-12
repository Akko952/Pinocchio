import pinocchio as pin
import numpy as np

urdf_path = "test/real_3SPR_copy.urdf"

model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()

print("=== Model Summary ===")
print("nq =", model.nq)
print("nv =", model.nv)
print("njoints =", model.njoints)
print()

print("=== Joint List ===")
for i in range(model.njoints):
    j = model.joints[i]
    print(f"[{i}] {model.names[i]}")
    print(f"  parent = {model.parents[i]}")
    print(f"  type   = {j.shortname()}")
    print(f"  idx_q  = {j.idx_q}, nq = {j.nq}")
    print(f"  idx_v  = {j.idx_v}, nv = {j.nv}")
print()

q = pin.neutral(model)
pin.forwardKinematics(model, data, q)

print("=== Joint Placements at q=0 ===")
for i in range(1, model.njoints):
    M = data.oMi[i]
    print(f"{model.names[i]}")
    print(f"  pos = {M.translation}")
    print(f"  rot =\n{M.rotation}")
print()

print("=== Single Joint Motion Test ===")
for joint_name in model.names[1:]:
    jid = model.getJointId(joint_name)
    joint = model.joints[jid]
    if joint.nq == 1:
        q_test = q.copy()
        q_test[joint.idx_q] = 0.2
        pin.forwardKinematics(model, data, q_test)
        M = data.oMi[jid]
        print(f"{joint_name} moved:")
        print(f"  pos = {M.translation}")