#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import bisect
import rosbag
import numpy as np
import matplotlib.pyplot as plt
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_from_matrix
import PyKDL as kdl
from kdl_parser_py.urdf import treeFromFile
from tf.transformations import euler_from_quaternion as tf_euler_from_quaternion
from tf.transformations import quaternion_from_euler as tf_quaternion_from_euler

# -----------------------------
# ユーザ設定（ここを直接編集）
# -----------------------------
ROSBAG_PATH = "/home/kaneko/mthesis/rosbag/me6_cleaning/fpv/20251229_154243_without_feedback.bag"
OUTPUT_DIR = "/home/kaneko/AIS/MSP-Latex-Template/rosbag_analysis/analyzed_images/me6_painting/20251229_154243_without_feedback"  # 手入力、絶対パス推奨
use_feedback_flag = False
use_energy_flag = False
# lpf_alpha = 0.1
LPF_CUTOFF_HZ = 3.0

ANALYSIS_START = 18   # bag 開始からの相対秒（開始）
ANALYSIS_END   = 44   # bag 開始からの相対秒（終了）

# Fx linear analysis
FX_THRESHOLD = 1.0   # [N] 押し込み判定しきい値

# FK 用設定（元スクリプトにならう）
BASE_POS = np.array([0.0, 0.0, 0.0])
BASE_RPY = np.array([0.0, 0.0, 0.0])
BASE_LINK = "base_link"
EE_LINK = "Link6"
URDF_PATH = "/home/kaneko/AIS/MSP-Latex-Template/rosbag_analysis/me6_robot_converted.urdf"

OFFSET_RPY = np.array([0.0, np.pi/2.0, 0.0])  # teleop と同じオフセット
OFFSET_QUAT = tf_quaternion_from_euler(*OFFSET_RPY)  # [x,y,z,w]

# target -> robot 軸入れ替え行列（元スクリプトと同様）
R_permute = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0]
])
R4 = np.eye(4)
R4[:3, :3] = R_permute
Q_PERMUTE = quaternion_from_matrix(R4)  # [x,y,z,w]

# 出力先準備
os.makedirs(OUTPUT_DIR, exist_ok=True)
BAG_FILENAME = os.path.splitext(os.path.basename(ROSBAG_PATH))[0]

# -----------------------------
# ヘルパ関数群（元スクリプトに合わせた実装）
# -----------------------------
def clamp(x, a=-1.0, b=1.0):
    return max(a, min(b, x))

def wrap_pi(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

def quat_conjugate(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)

def quat_mul(a, b):
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

def compute_rmse_vector(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.sqrt(np.mean((a - b)**2, axis=0))

def nearest_interp(base_times, base_data, target_time):
    # base_times はソート済みであることを仮定
    idx = bisect.bisect_left(base_times, target_time)
    if idx == 0:
        return base_data[0]
    if idx >= len(base_times):
        return base_data[-1]
    before = idx - 1
    after = idx
    if abs(base_times[before] - target_time) <= abs(base_times[after] - target_time):
        return base_data[before]
    else:
        return base_data[after]

def first_order_lpf(data, alpha):
    """
    data : (N, dim) numpy array
    alpha: delay_param (0 < alpha <= 1)
    """
    y = np.zeros_like(data)
    y[0] = data[0]  # 初期値

    for k in range(1, len(data)):
        y[k] = (1 - alpha) * y[k-1] + alpha * data[k]

    return y

def cutoff_to_alpha(fc, dt):
    """
    fc : cutoff frequency [Hz]
    dt : sampling period [s]
    """
    return (2.0 * np.pi * fc * dt) / (1.0 + 2.0 * np.pi * fc * dt)

def fit_fx_linear(time, fx, threshold):
    """
    threshold を超えた区間のみで一次近似
    """
    mask = np.abs(fx) > threshold
    if np.sum(mask) < 2:
        return None

    t_sel = time[mask]
    fx_sel = fx[mask]

    # 最小二乗直線
    coef = np.polyfit(t_sel, fx_sel, 1)
    fx_hat = np.polyval(coef, t_sel)

    rmse = np.sqrt(np.mean((fx_sel - fx_hat)**2))

    # 勾配符号反転回数
    diff = np.diff(fx_sel)
    sign = np.sign(diff)
    sign_changes = np.sum(sign[1:] * sign[:-1] < 0)

    return {
        "slope": coef[0],
        "intercept": coef[1],
        "rmse": rmse,
        "sign_flip_count": int(sign_changes),
        "num_samples": int(np.sum(mask))
    }

# -----------------------------
# KDL FK 構築（元スクリプトロジックを流用）
# -----------------------------
def build_fk_solver():
    ok, tree = treeFromFile(URDF_PATH)
    if not ok:
        raise RuntimeError("URDF の読み込み/パースに失敗しました: {}".format(URDF_PATH))
    chain = tree.getChain(BASE_LINK, EE_LINK)
    solver = kdl.ChainFkSolverPos_recursive(chain)
    return chain, solver

def compute_fk_apply_offset(chain, solver, joint_angles):
    n = chain.getNrOfJoints()
    arr = kdl.JntArray(n)
    for i in range(n):
        arr[i] = float(joint_angles[i])
    frame = kdl.Frame()
    solver.JntToCart(arr, frame)
    pos = np.array([frame.p[0], frame.p[1], frame.p[2]], dtype=float)
    pos_world = pos + BASE_POS
    rot = frame.M
    try:
        qx, qy, qz, qw = rot.GetQuaternion()
        quat_raw = np.array([qx, qy, qz, qw], dtype=float)
    except Exception:
        rpy_raw = np.array(rot.GetRPY())
        quat_raw = tf_quaternion_from_euler(*rpy_raw)
    quat_raw = norm_quat(quat_raw)
    rpy_raw = np.array(euler_from_quaternion(quat_raw))
    quat_actual_offset = quat_mul(quat_raw, OFFSET_QUAT)
    quat_actual_offset = norm_quat(quat_actual_offset)
    rpy_actual_offset = np.array(euler_from_quaternion(quat_actual_offset))
    return pos_world, quat_actual_offset, rpy_actual_offset, quat_raw, rpy_raw

# -----------------------------
# メイン処理
# -----------------------------
def main():
    chain, solver = build_fk_solver()

    # データ格納領域（時間は bag_start 基準の相対秒で格納）
    js_time = []
    js_vals = []

    tgt_time = []
    tgt_xyz = []
    tgt_quat = []
    tgt_rpy = []

    cfs_times = []
    cfs_force = []
    cfs_torque = []

    if use_feedback_flag:
        haptics_force = []
        haptics_torque = []
        haptics_times = []

    if use_energy_flag:
        debug_times = []
        energy = []
        dadd = []
        lam = []

    print("Loading rosbag:", ROSBAG_PATH)
    bag = rosbag.Bag(ROSBAG_PATH, "r")
    bag_start = bag.get_start_time()

    for topic, msg, t in bag.read_messages():
        t_rel = t.to_sec() - bag_start
        # 時間ウィンドウ外はスキップ
        if not (ANALYSIS_START <= t_rel <= ANALYSIS_END):
            continue

        # joint_states (me6 用)
        if topic == "/me6_robot/joint_states":
            js_time.append(t_rel)
            js_vals.append(np.array(msg.position, dtype=float))

        # target_pos (debug/mocap target)
        if topic == "/debug/target_pos":
            tgt_time.append(t_rel)
            tgt_xyz.append(np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float))
            q = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w], dtype=float)
            q = norm_quat(q)
            tgt_quat.append(q)
            tgt_rpy.append(np.array(euler_from_quaternion(q), dtype=float))

        # /cfs/data 相当 (WrenchStamped)
        if topic in "/cfs/data":
            cfs_times.append(t_rel)
            cfs_force.append([msg.wrench.force.z, msg.wrench.force.x, msg.wrench.force.y])
            cfs_torque.append([msg.wrench.torque.z, msg.wrench.torque.x, msg.wrench.torque.y])

        # /twin_hammer/haptics_wrench (WrenchStamped)
        if use_feedback_flag:
            if topic == "/twin_hammer/haptics_wrench":
                haptics_times.append(t_rel)
                haptics_force.append([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
                haptics_torque.append([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])

        if use_energy_flag:
            # ---------------------------------------------------------
            if topic == "/debug/energy":
                debug_times.append(t_rel)
                ene = msg.data[0]
                energy.append(ene)

            # ---------------------------------------------------------
            if topic == "/debug/dadd":
                dx = msg.data[0]
                dy = msg.data[1]
                dz = msg.data[2]
                droll = msg.data[3]
                dpitch = msg.data[4]
                dyaw = msg.data[5]
                dadd.append([dx, dy, dz, droll, dpitch, dyaw])

            # ---------------------------------------------------------
            if topic == "/debug/lambda":
                l = msg.data[0]
                lam.append([l])

    bag.close()

    # numpy 化（存在チェック）
    js_time = np.array(js_time)
    js_vals = np.array(js_vals) if len(js_vals) > 0 else np.zeros((0,0))

    tgt_time = np.array(tgt_time)
    tgt_xyz = np.array(tgt_xyz) if len(tgt_xyz) > 0 else np.zeros((0,3))
    tgt_quat = np.array(tgt_quat) if len(tgt_quat) > 0 else np.zeros((0,4))
    tgt_rpy = np.array(tgt_rpy) if len(tgt_rpy) > 0 else np.zeros((0,3))

    cfs_times = np.array(cfs_times)
    cfs_force = np.array(cfs_force) if len(cfs_force) > 0 else np.zeros((0,3))
    cfs_torque = np.array(cfs_torque) if len(cfs_torque) > 0 else np.zeros((0,3))

    if use_feedback_flag:
        haptics_times = np.array(haptics_times)
        haptics_force = np.array(haptics_force) if len(haptics_force) > 0 else np.zeros((0,3))
        haptics_torque = np.array(haptics_torque) if len(haptics_torque) > 0 else np.zeros((0,3))

    if use_energy_flag:
        dadd = np.array(dadd)

    ###########################################################
    # low pass filter
    ###########################################################
    cfs_wrench = np.hstack([cfs_force, cfs_torque])  # (N,6)
    dt_cfs = np.mean(np.diff(cfs_times))
    lpf_alpha = cutoff_to_alpha(LPF_CUTOFF_HZ, dt_cfs)
    print(f"LPF cutoff = {LPF_CUTOFF_HZ:.2f} Hz, dt = {dt_cfs:.4f} s, alpha = {lpf_alpha:.4f}")
    cfs_wrench_lpf = first_order_lpf(cfs_wrench, lpf_alpha)

    cfs_force  = cfs_wrench_lpf[:, 0:3]
    cfs_torque = cfs_wrench_lpf[:, 3:6]

    ###########################################################
    # Fx linear approximation & evaluation
    ###########################################################
    fx = cfs_force[:, 0]

    result = fit_fx_linear(cfs_times, fx, FX_THRESHOLD)

    if result is None:
        print("\n=== Fx Linear Evaluation ===")
        print("Not enough samples above threshold.")
    else:
        print("\n=== Fx Linear Evaluation ===")
        print(f"Threshold            : {FX_THRESHOLD:.2f} N")
        print(f"Samples used         : {result['num_samples']}")
        print(f"Slope (dF/dt)        : {result['slope']:.4f} [N/s]")
        print(f"Intercept            : {result['intercept']:.4f} [N]")
        print(f"RMSE to fitted line  : {result['rmse']:.4f} [N]")
        print(f"Sign flip count      : {result['sign_flip_count']}")
        print("=====================================")

    # -------------------------
    # /cfs/data の平均と軸ごとの RMSE を計算・出力
    # -------------------------
    if cfs_force.shape[0] > 0:
        avg_force = np.mean(cfs_force, axis=0)
        avg_torque = np.mean(cfs_torque, axis=0)
        rmse_force_axis = np.sqrt(np.mean((cfs_force - avg_force)**2, axis=0))
        rmse_torque_axis = np.sqrt(np.mean((cfs_torque - avg_torque)**2, axis=0))

        print("=== /cfs/data Statistics ===")

        print("Force Average:")
        print(f"  Fx: {avg_force[0]:.4f}")
        print(f"  Fy: {avg_force[1]:.4f}")
        print(f"  Fz: {avg_force[2]:.4f}")

        print("Force RMSE:")
        print(f"  Fx: {rmse_force_axis[0]:.4f}")
        print(f"  Fy: {rmse_force_axis[1]:.4f}")
        print(f"  Fz: {rmse_force_axis[2]:.4f}")

        print("Torque Average:")
        print(f"  Tx: {avg_torque[0]:.4f}")
        print(f"  Ty: {avg_torque[1]:.4f}")
        print(f"  Tz: {avg_torque[2]:.4f}")

        print("Torque RMSE:")
        print(f"  Tx: {rmse_torque_axis[0]:.4f}")
        print(f"  Ty: {rmse_torque_axis[1]:.4f}")
        print(f"  Tz: {rmse_torque_axis[2]:.4f}")

        print("=====================================\n")


    ###########################################################
    # 2. cfs force/torque プロット
    ###########################################################
    """
    # ----- Force -----
    plt.figure(figsize=(10,6))
    plt.title("CFS Wrench - Force")
    plt.plot(cfs_times, cfs_force[:,0], label="Fx")
    plt.plot(cfs_times, cfs_force[:,1], label="Fy")
    plt.plot(cfs_times, cfs_force[:,2], label="Fz")
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"cfs_force.png"))

    # ----- Torque -----
    plt.figure(figsize=(10,6))
    plt.title("CFS Wrench - Torque")
    plt.plot(cfs_times, cfs_torque[:,0], label="Tx")
    plt.plot(cfs_times, cfs_torque[:,1], label="Ty")
    plt.plot(cfs_times, cfs_torque[:,2], label="Tz")
    plt.xlabel("Time [s]")
    plt.ylabel("Torque [Nm]")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"cfs_torque.png"))

    ###########################################################
    # 3. twin_hammer force/torque プロット
    ###########################################################
    if use_feedback_flag:
        # ----- Force -----
        plt.figure(figsize=(10,6))
        plt.title("Haptics Wrench - Force")
        plt.plot(haptics_times, haptics_force[:,0], label="Fx")
        plt.plot(haptics_times, haptics_force[:,1], label="Fy")
        plt.plot(haptics_times, haptics_force[:,2], label="Fz")
        plt.xlabel("Time [s]")
        plt.ylabel("Force [N]")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"haptics_force.png"))

        # ----- Torque -----
        plt.figure(figsize=(10,6))
        plt.title("Haptics Wrench - Torque")
        plt.plot(haptics_times, haptics_torque[:,0], label="Tx")
        plt.plot(haptics_times, haptics_torque[:,1], label="Ty")
        plt.plot(haptics_times, haptics_torque[:,2], label="Tz")
        plt.xlabel("Time [s]")
        plt.ylabel("Torque [Nm]")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"haptics_torque.png"))
    """

    # fig, (ax_cfs_force, ax_haptics_force, ax_cfs_torque, ax_haptics_torque, ax_energy) = plt.subplots(
    #     5, 1, sharex=True, figsize=(6,18)
    # )

    fig, (ax_cfs_force, ax_haptics_force, ax_cfs_torque, ax_haptics_torque) = plt.subplots(
        4, 1, sharex=True, figsize=(6,14)
    )

    ax_cfs_force.plot(cfs_times, cfs_force[:, 0], label="Fx")
    ax_cfs_force.plot(cfs_times, cfs_force[:, 1], label="Fy")
    ax_cfs_force.plot(cfs_times, cfs_force[:, 2], label="Fz")
    ax_cfs_force.set_ylim(-11.5, 6.5)
    ax_cfs_force.set_ylabel("Measured Force [N]")
    ax_cfs_force.yaxis.set_label_coords(-0.12, 0.5)
    ax_cfs_force.grid(True)
    ax_cfs_force.legend()

    if use_feedback_flag:
        ax_haptics_force.plot(haptics_times, haptics_force[:, 0], label="Fx")
        ax_haptics_force.plot(haptics_times, haptics_force[:, 1], label="Fy")
        ax_haptics_force.plot(haptics_times, haptics_force[:, 2], label="Fz")
        ax_haptics_force.set_ylim(-11.5, 6.5)
        ax_haptics_force.set_ylabel("Feedback Force [N]")
        ax_haptics_force.yaxis.set_label_coords(-0.12, 0.5)
        ax_haptics_force.grid(True)
        ax_haptics_force.legend()
    else:
        ax_haptics_force.plot(cfs_times, cfs_force[:, 0], label="Fx")
        ax_haptics_force.plot(cfs_times, cfs_force[:, 1], label="Fy")
        ax_haptics_force.plot(cfs_times, cfs_force[:, 2], label="Fz")
        ax_haptics_force.set_ylim(-11.5, 6.5)
        ax_haptics_force.set_ylabel("Measured Force [N]")
        ax_haptics_force.yaxis.set_label_coords(-0.12, 0.5)
        ax_haptics_force.grid(True)
        ax_haptics_force.legend()

    ax_cfs_torque.plot(cfs_times, cfs_torque[:, 0], label="Tx")
    ax_cfs_torque.plot(cfs_times, cfs_torque[:, 1], label="Ty")
    ax_cfs_torque.plot(cfs_times, cfs_torque[:, 2], label="Tz")
    ax_cfs_torque.set_ylim(-1.2, 0.6)
    ax_cfs_torque.set_ylabel("Measured Torque [Nm]")
    ax_cfs_torque.yaxis.set_label_coords(-0.12, 0.5)
    ax_cfs_torque.grid(True)
    ax_cfs_torque.legend()

    if use_feedback_flag:
        ax_haptics_torque.plot(haptics_times, haptics_torque[:, 0], label="Tx")
        ax_haptics_torque.plot(haptics_times, haptics_torque[:, 1], label="Ty")
        ax_haptics_torque.plot(haptics_times, haptics_torque[:, 2], label="Tz")
        ax_haptics_torque.set_ylim(-1.2, 0.6)
        ax_haptics_torque.set_xlabel("time [s]")
        ax_haptics_torque.set_ylabel("Feedback Torque [Nm]")
        ax_haptics_torque.yaxis.set_label_coords(-0.12, 0.5)
        ax_haptics_torque.grid(True)
        ax_haptics_torque.legend()
    else:
        ax_haptics_torque.plot(cfs_times, cfs_torque[:, 0], label="Tx")
        ax_haptics_torque.plot(cfs_times, cfs_torque[:, 1], label="Ty")
        ax_haptics_torque.plot(cfs_times, cfs_torque[:, 2], label="Tz")
        ax_haptics_torque.set_ylim(-1.2, 0.6)
        ax_haptics_torque.set_xlabel("time [s]")
        ax_haptics_torque.set_ylabel("Measured Torque [Nm]")
        ax_haptics_torque.yaxis.set_label_coords(-0.12, 0.5)
        ax_haptics_torque.grid(True)
        ax_haptics_torque.legend()

    # if use_energy_flag:
    #     ax_energy.plot(debug_times, energy, label="Energy")
    #     ax_energy.set_xlabel("time [s]")
    #     ax_energy.set_ylabel("Energy [J]")
    #     ax_energy.yaxis.set_label_coords(-0.12, 0.5)
    #     ax_energy.grid(True)
    #     ax_energy.legend()
    # else:
    #     ax_energy.plot(cfs_times, cfs_torque[:, 0], label="Energy")
    #     ax_energy.set_xlabel("time [s]")
    #     ax_energy.set_ylabel("Energy [J]")
    #     ax_energy.yaxis.set_label_coords(-0.12, 0.5)
    #     ax_energy.grid(True)
    #     ax_energy.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "wrench_all_plot.png"))


    ###########################################################
    # 4. debug energy プロット
    ###########################################################

    if use_energy_flag:

        # ----- Energy-----
        plt.figure(figsize=(10,6))
        plt.title("Energy")
        plt.plot(debug_times, energy, label="Energy")
        plt.xlabel("Time [s]")
        plt.ylabel("Energy [J]")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"energy.png"))

        # ----- Dadd-----
        plt.figure(figsize=(10,6))
        plt.title("Added Damping Coefficient")
        plt.plot(debug_times, dadd[:,0], label="Dx")
        plt.plot(debug_times, dadd[:,1], label="Dy")
        plt.plot(debug_times, dadd[:,2], label="Dz")
        plt.plot(debug_times, dadd[:,3], label="Droll")
        plt.plot(debug_times, dadd[:,4], label="Dpitch")
        plt.plot(debug_times, dadd[:,5], label="Dyaw")
        plt.xlabel("Time [s]")
        plt.ylabel("Damping Coefficient [Ns/m]")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"dadd.png"))

        # ----- lambda-----
        # plt.figure(figsize=(10,6))
        # plt.title("Haptics Wrench Scale - Lambda")
        # plt.plot(debug_times, lam, label="Lambda")
        # plt.xlabel("Time [s]")
        # plt.ylabel("Scale Parameter")
        # plt.legend()
        # plt.grid()
        # plt.tight_layout()
        # plt.savefig(os.path.join(OUTPUT_DIR, f"lambda.png"))


    # -------------------------
    # target_pos と joint_states(FK) の同期 -> 実測と目標を XY にプロット
    # 元スクリプトの nearest-neighbor 同期ロジックを使用
    # -------------------------
    if js_time.shape[0] == 0:
        print("No joint_states found in analysis window; skipping FK/target comparison.")
    elif tgt_time.shape[0] == 0:
        print("No target_pos found in analysis window; skipping FK/target comparison.")
    else:
        # for each joint_state time, find nearest target
        synced_tgt_xyz = []
        synced_tgt_quat = []
        synced_tgt_rpy = []
        synced_act_pos = []
        synced_act_quat = []
        synced_act_rpy = []
        matched_tgt_time = []

        for t_js, js_q in zip(js_time, js_vals):
            idx = int(np.argmin(np.abs(tgt_time - t_js)))
            matched_tgt_time.append(tgt_time[idx])

            # target を robot フレームに変換（クォータニオンで軸入れ替え）
            q_t = tgt_quat[idx]
            q_t_robot = quat_mul(Q_PERMUTE, q_t)
            q_t_robot = norm_quat(q_t_robot)
            rpy_t_robot = np.array(euler_from_quaternion(q_t_robot))

            pos_act, quat_act_off, rpy_act_off, quat_act_raw, rpy_act_raw = compute_fk_apply_offset(chain, solver, js_q)

            synced_tgt_xyz.append(tgt_xyz[idx])
            synced_tgt_quat.append(q_t_robot)
            synced_tgt_rpy.append(rpy_t_robot)

            synced_act_pos.append(pos_act)
            synced_act_quat.append(quat_act_off)
            synced_act_rpy.append(rpy_act_off)

        synced_tgt_xyz = np.array(synced_tgt_xyz)
        synced_act_pos = np.array(synced_act_pos)

        # XY プロット
        plt.figure(figsize=(8,8))
        plt.title("XY: Target vs FK Actual (me6)")
        plt.plot(synced_tgt_xyz[:,0], synced_tgt_xyz[:,1], label="Target (mocap)", linewidth=2)
        plt.plot(synced_act_pos[:,0], synced_act_pos[:,1], label="FK Actual", linewidth=2)
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.legend()
        plt.grid()
        plt.axis("equal")
        plt.tight_layout()
        fname = os.path.join(OUTPUT_DIR, f"target_vs_fk_xy.png")
        plt.savefig(fname)
        plt.close()
        # print("Saved:", fname)

        # 位置 RMSE（軸ごと）
        pos_err = synced_tgt_xyz - synced_act_pos
        pos_rmse_axis = np.sqrt(np.mean(pos_err**2, axis=0))
        print("\n========== RMSE (Position) [m] ==========")
        print(f"X: {pos_rmse_axis[0]:.4f}")
        print(f"Y: {pos_rmse_axis[1]:.4f}")
        print(f"Z: {pos_rmse_axis[2]:.4f}")

        # orientation error: relative quaternion と RPY
        rpy_err_list = []
        scalar_angle_list = []
        for q_t_r, q_a in zip(synced_tgt_quat, synced_act_quat):
            qt = norm_quat(q_t_r)
            qa = norm_quat(q_a)
            q_err = quat_mul(qt, quat_conjugate(qa))
            q_err = norm_quat(q_err)
            w = clamp(q_err[3], -1.0, 1.0)
            angle = 2.0 * math.acos(w)
            scalar_angle_list.append(angle)
            rpy_err = np.array(euler_from_quaternion(q_err))
            rpy_err = np.array([wrap_pi(v) for v in rpy_err])
            rpy_err_list.append(rpy_err)

        rpy_err_arr = np.array(rpy_err_list)
        rpy_rmse = np.sqrt(np.mean(rpy_err_arr**2, axis=0))
        scalar_rmse = math.sqrt(np.mean(np.array(scalar_angle_list)**2))

        print("\n========== RMSE (Orientation, RPY) [rad] ==========")
        print(f"Roll : {rpy_rmse[0]:.4f}")
        print(f"Pitch: {rpy_rmse[1]:.4f}")
        print(f"Yaw  : {rpy_rmse[2]:.4f}")
        # print("\n========== RMSE (Orientation, scalar rotation angle) ==========")
        # print(f"Angle [rad]: {scalar_rmse:.6f}")

    print(f"\nAll images saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
