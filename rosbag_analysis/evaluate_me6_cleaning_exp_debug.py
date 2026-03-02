#!/usr/bin/env python3
"""
rosbag 解析（軸入れ替え + FK オフセット適用 + quaternion ベース誤差）
- target (mocap) の軸置換を適用: target.x -> robot.z, target.y -> robot.x, target.z -> robot.y
- joint_state を基準に target_pos を nearest-neighbor 同期
- FK に teleop と同じオフセットを適用（OFFSET_RPY）
- 姿勢誤差は相対 quaternion -> scalar angle と RPY（最短経路）で評価
- 最初の 10 サンプルを詳細にデバッグ出力
"""

import rosbag
import numpy as np
import PyKDL as kdl
from kdl_parser_py.urdf import treeFromFile
from tf.transformations import (
    euler_from_quaternion,
    quaternion_from_euler,
    quaternion_from_matrix,
)
import math

# -----------------------------
# ユーザが編集する設定
# -----------------------------
ROSBAG_PATH = "/home/kaneko/mthesis/rosbag/me6_cleaning/2025-12-10-15-02-02_me6_brush_with_feedback_without_energy_succeeded.bag"
URDF_PATH = "/home/kaneko/ros/jsk_aerial_robot_ws/src/jsk_aerial_robot/robots/twin_hammer/scripts/me6_robot_converted.urdf"

BASE_POS = np.array([0.0, 0.0, 0.0])   # world 上の base の位置補正 [m]
BASE_RPY = np.array([0.0, 0.0, 0.0])   # world 上の base の姿勢補正 [rad]

BASE_LINK = "base_link"
EE_LINK = "Link6"

ANALYSIS_START = 17.0   # bag 開始からの相対秒
ANALYSIS_END   = 58.0

# teleop と同じオフセット (me6_teleop.py に合わせる)
OFFSET_RPY = np.array([0.0, np.pi/2.0, 0.0])
OFFSET_QUAT = quaternion_from_euler(*OFFSET_RPY)  # [x,y,z,w]

# target -> robot 軸入れ替え行列（target の座標軸を robot_EE の軸に写す）
# mapping: target.x -> robot.z, target.y -> robot.x, target.z -> robot.y
R_permute = np.array([
    [0.0, 0.0, 1.0],  # robot.x = target.z? (but mapping given as target.x -> robot.z etc.)
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0]
])
# NOTE: The above matrix maps target vector to robot vector: v_robot = R_permute @ v_target
# Convert to quaternion (4x4 matrix required)
R4 = np.eye(4)
R4[:3, :3] = R_permute
Q_PERMUTE = quaternion_from_matrix(R4)  # quaternion [x,y,z,w]

# -----------------------------
# ヘルパ関数
# -----------------------------
def clamp(x, a=-1.0, b=1.0):
    return max(a, min(b, x))

def wrap_pi(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

def quat_conjugate(q):
    # q: [x,y,z,w]
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)

def quat_mul(a, b):
    # Hamilton product, a and b are [x,y,z,w]
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    x = aw*bx + ax*bw + ay*bz - az*by
    y = aw*by - ax*bz + ay*bw + az*bx
    z = aw*bz + ax*by - ay*bx + az*bw
    w = aw*bw - ax*bx - ay*by - az*bz
    return np.array([x, y, z, w], dtype=float)

def norm_quat(q):
    q = np.array(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n

# -----------------------------
# KDL FK 構築
# -----------------------------
def build_fk_solver():
    ok, tree = treeFromFile(URDF_PATH)
    if not ok:
        raise RuntimeError("URDF の読み込み/パースに失敗しました: {}".format(URDF_PATH))
    chain = tree.getChain(BASE_LINK, EE_LINK)
    solver = kdl.ChainFkSolverPos_recursive(chain)
    return chain, solver

def compute_fk_apply_offset(chain, solver, joint_angles):
    """
    joint_angles: iterable (length = chain.getNrOfJoints())
    returns:
      pos_world: np.array([x,y,z])
      quat_actual_offset: quaternion after applying teleop OFFSET [x,y,z,w]
      rpy_actual_offset: np.array([r,p,y]) in radians
      quat_actual_raw: quaternion before OFFSET (for debugging)
      rpy_actual_raw: rpy before OFFSET
    """
    n = chain.getNrOfJoints()
    arr = kdl.JntArray(n)
    for i in range(n):
        arr[i] = float(joint_angles[i])

    frame = kdl.Frame()
    solver.JntToCart(arr, frame)

    # position
    pos = np.array([frame.p[0], frame.p[1], frame.p[2]], dtype=float)
    pos_world = pos + BASE_POS

    # rotation -> quaternion from PyKDL
    rot = frame.M
    try:
        qx, qy, qz, qw = rot.GetQuaternion()
        quat_raw = np.array([qx, qy, qz, qw], dtype=float)
    except Exception:
        # fallback via RPY
        rpy_raw = np.array(rot.GetRPY())
        quat_raw = quaternion_from_euler(*rpy_raw)

    quat_raw = norm_quat(quat_raw)
    rpy_raw = np.array(euler_from_quaternion(quat_raw))

    # apply teleop OFFSET to FK result (same order as teleop: quat_actual * offset)
    quat_actual_offset = quat_mul(quat_raw, OFFSET_QUAT)
    quat_actual_offset = norm_quat(quat_actual_offset)
    rpy_actual_offset = np.array(euler_from_quaternion(quat_actual_offset))

    return pos_world, quat_actual_offset, rpy_actual_offset, quat_raw, rpy_raw

# -----------------------------
# Main
# -----------------------------
def main():
    chain, solver = build_fk_solver()

    # collect data from bag
    js_time = []
    js_vals = []

    tgt_time = []
    tgt_xyz = []
    tgt_quat = []
    tgt_rpy = []

    with rosbag.Bag(ROSBAG_PATH, "r") as bag:
        bag_start = bag.get_start_time()

        for topic, msg, t in bag.read_messages():
            t_rel = t.to_sec() - bag_start
            if not (ANALYSIS_START <= t_rel <= ANALYSIS_END):
                continue

            if topic == "/me6_robot/joint_states":
                js_time.append(t_rel)
                js_vals.append(np.array(msg.position, dtype=float))

            elif topic == "/debug/target_pos":
                tgt_time.append(t_rel)
                tgt_xyz.append(np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float))
                q = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w], dtype=float)
                q = norm_quat(q)
                tgt_quat.append(q)
                tgt_rpy.append(np.array(euler_from_quaternion(q), dtype=float))

    if len(js_time) == 0:
        print("解析区間内に /me6_robot/joint_state が見つかりませんでした。")
        return
    if len(tgt_time) == 0:
        print("解析区間内に /debug/target_pos が見つかりませんでした。")
        return

    js_time = np.array(js_time)
    js_vals = np.array(js_vals)
    tgt_time = np.array(tgt_time)
    tgt_xyz = np.array(tgt_xyz)
    tgt_quat = np.array(tgt_quat)
    tgt_rpy = np.array(tgt_rpy)

    # sync: for each joint_state timestamp, find nearest target_pos
    synced_tgt_xyz = []
    synced_tgt_quat = []
    synced_tgt_rpy = []

    synced_act_pos = []
    synced_act_quat = []
    synced_act_rpy = []
    synced_act_quat_raw = []
    synced_act_rpy_raw = []

    # We'll also keep the matched target timestamp for debug output
    matched_tgt_time = []

    for i, (t_js, js_q) in enumerate(zip(js_time, js_vals)):
        idx = int(np.argmin(np.abs(tgt_time - t_js)))
        matched_tgt_time.append(tgt_time[idx])

        # get target and apply axis permutation (frame mapping)
        q_t = tgt_quat[idx]  # [x,y,z,w] in target (mocap) frame
        # convert target quaternion to robot EE frame via permutation quaternion:
        # q_t_robot = Q_PERMUTE * q_t
        q_t_robot = quat_mul(Q_PERMUTE, q_t)
        q_t_robot = norm_quat(q_t_robot)
        rpy_t_robot = np.array(euler_from_quaternion(q_t_robot))

        # FK and apply teleop offset to actual
        pos_act, quat_act_off, rpy_act_off, quat_act_raw, rpy_act_raw = compute_fk_apply_offset(chain, solver, js_q)

        # store
        synced_tgt_xyz.append(tgt_xyz[idx])
        synced_tgt_quat.append(q_t_robot)
        synced_tgt_rpy.append(rpy_t_robot)

        synced_act_pos.append(pos_act)
        synced_act_quat.append(quat_act_off)
        synced_act_rpy.append(rpy_act_off)
        synced_act_quat_raw.append(quat_act_raw)
        synced_act_rpy_raw.append(rpy_act_raw)

    synced_tgt_xyz = np.array(synced_tgt_xyz)
    synced_tgt_quat = np.array(synced_tgt_quat)
    synced_tgt_rpy = np.array(synced_tgt_rpy)

    synced_act_pos = np.array(synced_act_pos)
    synced_act_quat = np.array(synced_act_quat)
    synced_act_rpy = np.array(synced_act_rpy)
    synced_act_quat_raw = np.array(synced_act_quat_raw)
    synced_act_rpy_raw = np.array(synced_act_rpy_raw)
    matched_tgt_time = np.array(matched_tgt_time)

    # -------------------------
    # Position RMSE
    # -------------------------
    pos_err = synced_tgt_xyz - synced_act_pos
    pos_rmse = np.sqrt(np.mean(pos_err**2, axis=0))

    # -------------------------
    # Orientation: relative quaternion + RPY shortest-path
    # -------------------------
    rpy_err_list = []
    scalar_angle_list = []

    for q_t_r, q_a in zip(synced_tgt_quat, synced_act_quat):
        qt = norm_quat(q_t_r)
        qa = norm_quat(q_a)

        # relative quaternion: q_err = q_target_corrected * conj(q_actual)
        q_err = quat_mul(qt, quat_conjugate(qa))
        q_err = norm_quat(q_err)

        # scalar rotation angle
        w = clamp(q_err[3], -1.0, 1.0)
        angle = 2.0 * math.acos(w)  # [0, pi]
        scalar_angle_list.append(angle)

        # RPY of q_err (then wrap to [-pi,pi])
        rpy_err = np.array(euler_from_quaternion(q_err))
        rpy_err = np.array([wrap_pi(v) for v in rpy_err])
        rpy_err_list.append(rpy_err)

    rpy_err_arr = np.array(rpy_err_list)
    rpy_rmse = np.sqrt(np.mean(rpy_err_arr**2, axis=0))
    scalar_rmse = math.sqrt(np.mean(np.array(scalar_angle_list)**2))

    # -------------------------
    # Debug: first 10 samples detailed print
    # -------------------------
    N = min(10, len(synced_tgt_quat))
    print("\n===== DEBUG: First {} samples =====".format(N))
    for i in range(N):
        print("\n--- Sample {} ---".format(i))
        print(f"joint_state time: {js_time[i]:.4f}   matched target time: {matched_tgt_time[i]:.4f}")
        print("raw target quat (mocap) [x,y,z,w]:", tgt_quat[np.argmin(np.abs(tgt_time - js_time[i]))])
        print("target quat -> robot frame (after permute) [x,y,z,w]:", synced_tgt_quat[i])
        print("target rpy (deg) in robot frame:", np.degrees(synced_tgt_rpy[i]))

        print("FK actual quat (raw) [x,y,z,w]:", synced_act_quat_raw[i])
        print("FK actual rpy (deg) (raw):", np.degrees(synced_act_rpy_raw[i]))

        print("FK actual quat (offset applied) [x,y,z,w]:", synced_act_quat[i])
        print("FK actual rpy (deg) (offset applied):", np.degrees(synced_act_rpy[i]))

        # relative
        q_err = quat_mul(synced_tgt_quat[i], quat_conjugate(synced_act_quat[i]))
        q_err = norm_quat(q_err)
        ang_deg = np.degrees(2.0 * math.acos(clamp(q_err[3], -1.0, 1.0)))
        print("relative quat (target_corrected * conj(actual)):", q_err)
        print("relative rotation angle (deg): {:.3f}".format(ang_deg))
        re_rpy = np.array(euler_from_quaternion(q_err))
        re_rpy = np.array([wrap_pi(v) for v in re_rpy])
        print("relative RPY (deg):", np.degrees(re_rpy))

    # -------------------------
    # Results
    # -------------------------
    print("\n========== RMSE (Position) [m] ==========")
    print(f"X: {pos_rmse[0]:.6f}")
    print(f"Y: {pos_rmse[1]:.6f}")
    print(f"Z: {pos_rmse[2]:.6f}")

    print("\n========== RMSE (Orientation, RPY) [rad] ==========")
    print(f"Roll : {rpy_rmse[0]:.6f}")
    print(f"Pitch: {rpy_rmse[1]:.6f}")
    print(f"Yaw  : {rpy_rmse[2]:.6f}")

    print("\n========== RMSE (Orientation, scalar rotation angle) ==========")
    print(f"Angle [rad]: {scalar_rmse:.6f}")

    print("\n解析完了")

if __name__ == "__main__":
    main()
