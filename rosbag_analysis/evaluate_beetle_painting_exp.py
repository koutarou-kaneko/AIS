#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rosbag
import numpy as np
import matplotlib.pyplot as plt
from tf.transformations import euler_from_quaternion
import math
import bisect
import os

###########################################################
#  ユーザー設定
###########################################################
rosbag_path = "/home/kaneko/mthesis/rosbag/beetle_cleaning/fpv/20251222_201757_without_feedback.bag"   # 手入力
output_dir  = "/home/kaneko/AIS/MSP-Latex-Template/rosbag_analysis/analyzed_images/beetle_painting/20251222_201757_without_feedback"     # 手入力
use_feedback_flag = False
use_energy_flag = False
# lpf_alpha = 0.05
lpf_alpha_haptics = 0.1

start_time_sec = 61     # 評価開始時間 [s]
end_time_sec   = 101    # 評価終了時間 [s]

contact_force_threshold = 0.5   # [N] 接触判定しきい値（Fx）
qs_epsilon = 0.2 # 準定常帯域の許容幅 [N]（平均値 ± epsilon）

lpf_cutoff_hz = 3.0          # [Hz] 評価用 LPF（操縦者帯域）
lpf_cutoff_haptics_hz = 3.0  # [Hz] 触覚提示用（必要なら）

# rosbag のファイル名（拡張子除く）
bag_filename = os.path.splitext(os.path.basename(rosbag_path))[0]

# 出力先ディレクトリ作成
os.makedirs(output_dir, exist_ok=True)

###########################################################
#  便利関数
###########################################################
def in_time_window(t, t0):
    return start_time_sec <= (t - t0) <= end_time_sec

def wrap_to_pi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def compute_rmse(a, b):
    a = np.array(a)
    b = np.array(b)
    return math.sqrt(np.mean((a - b) ** 2))

def compute_angle_rmse(a, b):
    a = np.array(a)
    b = np.array(b)
    diff = wrap_to_pi(a - b)
    return math.sqrt(np.mean(diff ** 2))

def nearest_interp(base_times, base_data, target_time):
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

def compute_lpf_alpha(fc, dt):
    """
    fc : cutoff frequency [Hz]
    dt : sampling period [s]
    """
    return (2.0 * np.pi * fc * dt) / (1.0 + 2.0 * np.pi * fc * dt)

def quasi_static_index(Fx, eps=1e-9):
    """
    準定常性指標 Q_qs
    Fx : 接触区間の力データ（numpy array）
    eps: ゼロ割防止用の微小値
    """
    Fx = np.asarray(Fx)

    mean_fx = np.mean(Fx)
    std_fx  = np.std(Fx)

    Q_qs = np.abs(mean_fx) / (std_fx + eps)

    return Q_qs, mean_fx, std_fx

def quasi_static_ratio(Fx, epsilon):
    """
    準定常帯域滞在率 R_qs
    epsilon : 許容偏差 [N]
    """
    Fx = np.asarray(Fx)
    mean_fx = np.mean(Fx)

    inside = np.abs(Fx - mean_fx) <= epsilon
    R_qs = np.sum(inside) / len(Fx)

    return R_qs

def linear_fit_metrics(time, Fx):
    """
    time : 時刻 [s]
    Fx   : 力 [N]
    """
    # 一次近似
    coeff = np.polyfit(time, Fx, 1)
    Fx_fit = np.polyval(coeff, time)

    # RMSE
    rmse = np.sqrt(np.mean((Fx - Fx_fit) ** 2))

    # 勾配符号反転回数
    dFx = np.diff(Fx)
    sign_change = np.sum(np.diff(np.sign(dFx)) != 0)

    return coeff, rmse, sign_change



###########################################################
#  データ格納領域
###########################################################
cfs_wrench_force = []
cfs_wrench_torque = []
cfs_times = []

if use_feedback_flag:
    haptics_force = []
    haptics_torque = []
    haptics_times = []

nav_pos = []
nav_rpy = []
nav_times = []

mocap_pos = []
mocap_rpy = []
mocap_times = []

if use_energy_flag:
    debug_times = []
    energy = []
    dadd = []
    lam = []

###########################################################
#  rosbag 読み込み
###########################################################
print("Loading rosbag:", rosbag_path)
bag = rosbag.Bag(rosbag_path, "r")
bag_start = bag.get_start_time()

for topic, msg, t in bag.read_messages():
    t_sec = t.to_sec()

    # ---------------------------------------------------------
    if topic == "/filtered_ftsensor" and in_time_window(t_sec, bag_start):
        cfs_times.append(t_sec - bag_start)
        cfs_wrench_force.append([
            msg.wrench.force.z,
            msg.wrench.force.x,
            msg.wrench.force.y
        ])
        cfs_wrench_torque.append([
            msg.wrench.torque.z,
            msg.wrench.torque.x,
            msg.wrench.torque.y
        ])

    # ---------------------------------------------------------
    if not use_feedback_flag:
        if topic == "/cfs/data" and in_time_window(t_sec, bag_start):
            cfs_times.append(t_sec - bag_start)
            cfs_wrench_force.append([
                msg.wrench.force.z,
                msg.wrench.force.x,
                msg.wrench.force.y
            ])
            cfs_wrench_torque.append([
                msg.wrench.torque.z,
                msg.wrench.torque.x,
                msg.wrench.torque.y
            ])

    # ---------------------------------------------------------
    if use_feedback_flag:
        # ---------------------------------------------------------
        if topic == "/twin_hammer/haptics_wrench" and in_time_window(t_sec, bag_start):
            haptics_times.append(t_sec - bag_start)
            haptics_force.append([
                msg.wrench.force.x,
                msg.wrench.force.y,
                msg.wrench.force.z
            ])
            haptics_torque.append([
                msg.wrench.torque.x,
                msg.wrench.torque.y,
                msg.wrench.torque.z
            ])

    # ---------------------------------------------------------
    if topic == "/gimbalrotor/uav/nav" and in_time_window(t_sec, bag_start):
        nav_times.append(t_sec - bag_start)
        nav_pos.append([
            msg.target_pos_x,
            msg.target_pos_y,
            msg.target_pos_z
        ])
        nav_rpy.append([
            msg.target_roll,
            msg.target_pitch,
            msg.target_yaw
        ])

    # ---------------------------------------------------------
    if topic == "/gimbalrotor/mocap/pose" and in_time_window(t_sec, bag_start):
        mocap_times.append(t_sec - bag_start)
        px = msg.pose.position.x
        py = msg.pose.position.y
        pz = msg.pose.position.z
        q = msg.pose.orientation
        roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        mocap_pos.append([px, py, pz])
        mocap_rpy.append([roll, pitch, yaw])

    if use_energy_flag:
        # ---------------------------------------------------------
        if topic == "/debug/energy" and in_time_window(t_sec, bag_start):
            debug_times.append(t_sec - bag_start)
            ene = msg.data[0]
            energy.append(ene)

        # ---------------------------------------------------------
        if topic == "/debug/dadd" and in_time_window(t_sec, bag_start):
            dx = msg.data[0]
            dy = msg.data[1]
            dz = msg.data[2]
            droll = msg.data[3]
            dpitch = msg.data[4]
            dyaw = msg.data[5]
            dadd.append([dx, dy, dz, droll, dpitch, dyaw])

        # ---------------------------------------------------------
        if topic == "/debug/lambda" and in_time_window(t_sec, bag_start):
            l = msg.data[0]
            lam.append([l])

bag.close()

###########################################################
# numpy 配列化
###########################################################
cfs_wrench_force = np.array(cfs_wrench_force)
cfs_wrench_torque = np.array(cfs_wrench_torque)

if use_feedback_flag:
    haptics_force = np.array(haptics_force)
    haptics_torque = np.array(haptics_torque)

nav_pos = np.array(nav_pos)
nav_rpy = np.array(nav_rpy)
# yaw（index=2）のみ unwrap
nav_rpy[:, 2] = np.unwrap(nav_rpy[:, 2])

mocap_pos = np.array(mocap_pos)
mocap_rpy = np.array(mocap_rpy)
mocap_rpy[:, 2] = np.unwrap(mocap_rpy[:, 2])

if use_energy_flag:
    dadd = np.array(dadd)

###########################################################
# low pass filter
###########################################################
# サンプリング周期推定（中央値を使用）
dt_cfs = np.median(np.diff(cfs_times))

lpf_alpha = compute_lpf_alpha(lpf_cutoff_hz, dt_cfs)
print(f"LPF cutoff = {lpf_cutoff_hz:.2f} Hz  -> alpha = {lpf_alpha:.4f}")
# lpf_alpha = 0.03203

if use_energy_flag:
    dt_haptics = np.median(np.diff(haptics_times))

    lpf_alpha_haptics = compute_lpf_alpha(lpf_cutoff_haptics_hz, dt_haptics)
    print(f"LPF cutoff = {lpf_cutoff_haptics_hz:.2f} Hz  -> alpha = {lpf_alpha_haptics:.4f}")

cfs_wrench = np.hstack([cfs_wrench_force, cfs_wrench_torque])  # (N,6)

cfs_wrench_lpf = first_order_lpf(cfs_wrench, lpf_alpha)

cfs_wrench_force  = cfs_wrench_lpf[:, 0:3]
cfs_wrench_torque = cfs_wrench_lpf[:, 3:6]

if use_energy_flag:
    haptics_wrench = np.hstack([haptics_force, haptics_torque])  # (N,6)

    haptics_wrench_lpf = first_order_lpf(haptics_wrench, lpf_alpha_haptics)

    haptics_force  = haptics_wrench_lpf[:, 0:3]
    haptics_torque = haptics_wrench_lpf[:, 3:6]

###########################################################
# 接触判定マスク（Fx しきい値）
###########################################################
# Fx = cfs_wrench_force[:, 0]

contact_mask = np.abs(cfs_wrench_force[:, 0]) > contact_force_threshold

num_contact_samples = np.sum(contact_mask)

# print(f"Contact samples: {num_contact_samples} / {len(cfs_wrench_force)}")

# if num_contact_samples == 0:
#     print("WARNING: No contact samples detected. Statistics will be skipped.")

###########################################################
# 1. /cfs/data 接触区間のみの平均・RMSE
###########################################################
if len(cfs_wrench_force) > 0 and num_contact_samples > 0:

    contact_force  = cfs_wrench_force[contact_mask]
    contact_torque = cfs_wrench_torque[contact_mask]

    avg_force = np.mean(contact_force, axis=0)
    avg_torque = np.mean(contact_torque, axis=0)

    rmse_force_axis = np.sqrt(
        np.mean((contact_force - avg_force) ** 2, axis=0)
    )
    rmse_torque_axis = np.sqrt(
        np.mean((contact_torque - avg_torque) ** 2, axis=0)
    )

    # print("=== /cfs/data Contact Statistics (Fx threshold based) ===")
    # print(f"Fx threshold: {contact_force_threshold:.3f} [N]")
    # print(f"Contact ratio: {num_contact_samples/len(cfs_wrench_force)*100:.1f} [%]\n")

    # print("Force Average (Contact Only):")
    # print(f"  Fx: {avg_force[0]:.4f}")
    # print(f"  Fy: {avg_force[1]:.4f}")
    # print(f"  Fz: {avg_force[2]:.4f}")

    # print("Force RMSE (Contact Only):")
    # print(f"  Fx: {rmse_force_axis[0]:.4f}")
    # print(f"  Fy: {rmse_force_axis[1]:.4f}")
    # print(f"  Fz: {rmse_force_axis[2]:.4f}")

    # print("Torque Average (Contact Only):")
    # print(f"  Tx: {avg_torque[0]:.4f}")
    # print(f"  Ty: {avg_torque[1]:.4f}")
    # print(f"  Tz: {avg_torque[2]:.4f}")

    # print("Torque RMSE (Contact Only):")
    # print(f"  Tx: {rmse_torque_axis[0]:.4f}")
    # print(f"  Ty: {rmse_torque_axis[1]:.4f}")
    # print(f"  Tz: {rmse_torque_axis[2]:.4f}")

    # print("=====================================\n")

    ###########################################################
    # 準定常性評価（Quasi-static metrics）
    ###########################################################

    # # Fx のみを対象（法線力）
    # Fx_contact = contact_force[:, 0]

    # Q_qs, mean_fx, std_fx = quasi_static_index(Fx_contact)
    # R_qs = quasi_static_ratio(Fx_contact, qs_epsilon)

    # print("=== Quasi-static Evaluation (Contact Fx) ===")
    # print(f"Mean Fx        : {mean_fx:.4f} [N]")
    # print(f"Std  Fx        : {std_fx:.4f} [N]")
    # print(f"Q_qs (|mean|/std) : {Q_qs:.3f}")
    # print(f"R_qs (|Fx-mean|<= {qs_epsilon:.2f} N): {R_qs*100:.1f} [%]")
    # print("===========================================\n")


Fx = cfs_wrench_force[:, 0]
t  = np.array(cfs_times)

mask_fx = np.abs(Fx) > contact_force_threshold

Fx_contact = Fx[mask_fx]
t_contact  = t[mask_fx]

if len(Fx_contact) > 10:
    coeff, rmse_lin, n_flip = linear_fit_metrics(t_contact, Fx_contact)

    print("=== Operator Oscillation Metrics (Fx) ===")
    print(f"Linear slope      : {coeff[0]:.4f} [N/s]")
    print(f"Linear RMSE       : {rmse_lin:.4f} [N]")
    print(f"Gradient flips    : {n_flip} [-]")
    print("=========================================\n")
else:
    print("Not enough contact samples for linear analysis.")


###########################################################
# 2. cfs force/torque プロット
###########################################################

# ----- Force -----
plt.figure(figsize=(10,6))
# plt.title("CFS Wrench - Force")
plt.plot(cfs_times, cfs_wrench_force[:,0], label="Fx")
plt.plot(cfs_times, cfs_wrench_force[:,1], label="Fy")
plt.plot(cfs_times, cfs_wrench_force[:,2], label="Fz")
plt.ylim(-4.0, 2.0)
plt.xlabel("Time [s]")
plt.ylabel("Force [N]")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"cfs_force.png"))

# ----- Torque -----
plt.figure(figsize=(10,6))
# plt.title("CFS Wrench - Torque")
plt.plot(cfs_times, cfs_wrench_torque[:,0], label="Tx")
plt.plot(cfs_times, cfs_wrench_torque[:,1], label="Ty")
plt.plot(cfs_times, cfs_wrench_torque[:,2], label="Tz")
plt.ylim(-0.25,0.10)
plt.xlabel("Time [s]")
plt.ylabel("Torque [Nm]")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"cfs_torque.png"))

###########################################################
# 3. twin_hammer force/torque プロット
###########################################################
"""
if use_feedback_flag:
    # ----- Force -----
    plt.figure(figsize=(10,6))
    plt.title("Haptics Wrench - Force")
    plt.plot(haptics_times, -haptics_force[:,0], label="Fx")
    plt.plot(haptics_times, -haptics_force[:,1], label="Fy")
    plt.plot(haptics_times, -haptics_force[:,2], label="Fz")
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"haptics_force.png"))

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
    plt.savefig(os.path.join(output_dir, f"haptics_torque.png"))
"""

fig, (ax_cfs_force, ax_haptics_force, ax_cfs_torque, ax_haptics_torque, ax_energy) = plt.subplots(
    5, 1, sharex=True, figsize=(6,18)
)

ax_cfs_force.plot(cfs_times, cfs_wrench_force[:, 0], label="Fx")
ax_cfs_force.plot(cfs_times, cfs_wrench_force[:, 1], label="Fy")
ax_cfs_force.plot(cfs_times, cfs_wrench_force[:, 2], label="Fz")
ax_cfs_force.set_ylim(-4.0, 2.0)
ax_cfs_force.set_ylabel("Measured Force [N]")
ax_cfs_force.yaxis.set_label_coords(-0.12, 0.5)
ax_cfs_force.grid(True)
ax_cfs_force.legend()

if use_feedback_flag:
    ax_haptics_force.plot(haptics_times, -haptics_force[:, 0], label="Fx")
    ax_haptics_force.plot(haptics_times, -haptics_force[:, 1], label="Fy")
    ax_haptics_force.plot(haptics_times, haptics_force[:, 2], label="Fz")
    ax_haptics_force.set_ylim(-8.0, 3.0)
    ax_haptics_force.set_ylabel("Feedback Force [N]")
    ax_haptics_force.yaxis.set_label_coords(-0.12, 0.5)
    ax_haptics_force.grid(True)
    ax_haptics_force.legend()
else:
    ax_haptics_force.plot(cfs_times, cfs_wrench_force[:, 0], label="Fx")
    ax_haptics_force.plot(cfs_times, cfs_wrench_force[:, 1], label="Fy")
    ax_haptics_force.plot(cfs_times, cfs_wrench_force[:, 2], label="Fz")
    ax_haptics_force.set_ylim(-4.0, 2.0)
    ax_haptics_force.set_ylabel("Measured Force [N]")
    ax_haptics_force.yaxis.set_label_coords(-0.12, 0.5)
    ax_haptics_force.grid(True)
    ax_haptics_force.legend()

ax_cfs_torque.plot(cfs_times, cfs_wrench_torque[:, 0], label="Tx")
ax_cfs_torque.plot(cfs_times, cfs_wrench_torque[:, 1], label="Ty")
ax_cfs_torque.plot(cfs_times, cfs_wrench_torque[:, 2], label="Tz")
ax_cfs_torque.set_ylim(-0.25, 0.10)
ax_cfs_torque.set_ylabel("Measured Torque [Nm]")
ax_cfs_torque.yaxis.set_label_coords(-0.12, 0.5)
ax_cfs_torque.grid(True)
ax_cfs_torque.legend()

if use_feedback_flag:
    ax_haptics_torque.plot(haptics_times, haptics_torque[:, 0], label="Tx")
    ax_haptics_torque.plot(haptics_times, haptics_torque[:, 1], label="Ty")
    ax_haptics_torque.plot(haptics_times, haptics_torque[:, 2], label="Tz")
    ax_haptics_torque.set_ylim(-1.2, 1.2)
    # ax_haptics_torque.set_xlabel("time [s]")
    ax_haptics_torque.set_ylabel("Feedback Torque [Nm]")
    ax_haptics_torque.yaxis.set_label_coords(-0.12, 0.5)
    ax_haptics_torque.grid(True)
    ax_haptics_torque.legend()
else:
    ax_haptics_torque.plot(cfs_times, cfs_wrench_torque[:, 0], label="Tx")
    ax_haptics_torque.plot(cfs_times, cfs_wrench_torque[:, 1], label="Ty")
    ax_haptics_torque.plot(cfs_times, cfs_wrench_torque[:, 2], label="Tz")
    ax_haptics_torque.set_ylim(-0.25, 0.10)
    # ax_haptics_torque.set_xlabel("time [s]")
    ax_haptics_torque.set_ylabel("Measured Torque [Nm]")
    ax_haptics_torque.yaxis.set_label_coords(-0.12, 0.5)
    ax_haptics_torque.grid(True)
    ax_haptics_torque.legend()

if use_energy_flag:
    ax_energy.plot(debug_times, energy, label="Energy")
    ax_energy.set_xlabel("time [s]")
    ax_energy.set_ylabel("Energy [J]")
    ax_energy.yaxis.set_label_coords(-0.12, 0.5)
    ax_energy.grid(True)
    ax_energy.legend()
else:
    ax_energy.plot(cfs_times, cfs_wrench_torque[:, 0], label="Energy")
    ax_energy.set_xlabel("time [s]")
    ax_energy.set_ylabel("Energy [J]")
    ax_energy.yaxis.set_label_coords(-0.12, 0.5)
    ax_energy.grid(True)
    ax_energy.legend()
  


plt.tight_layout()
plt.savefig(os.path.join(output_dir, "wrench_all_plot.png"))

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
    plt.savefig(os.path.join(output_dir, f"energy.png"))

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
    plt.savefig(os.path.join(output_dir, f"dadd.png"))

    # ----- lambda-----
    plt.figure(figsize=(10,6))
    plt.title("Haptics Wrench Scale - Lambda")
    plt.plot(debug_times, lam, label="Lambda")
    plt.xlabel("Time [s]")
    plt.ylabel("Scale Parameter")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"lambda.png"))

###########################################################
# 5. Nav – Mocap 時間同期
###########################################################
sync_nav_pos = []
sync_nav_rpy = []

for t in mocap_times:
    sync_nav_pos.append(nearest_interp(nav_times, nav_pos, t))
    sync_nav_rpy.append(nearest_interp(nav_times, nav_rpy, t))

sync_nav_pos = np.array(sync_nav_pos)
sync_nav_rpy = np.array(sync_nav_rpy)

###########################################################
# 6. RMSE 計算
###########################################################
print("=== Nav vs Mocap RMSE (Time-Synchronized) ===")

labels_pos = ["x", "y", "z"]
for i, name in enumerate(labels_pos):
    rmse = compute_rmse(sync_nav_pos[:, i], mocap_pos[:, i])
    print(f"pos_{name}: RMSE = {rmse:.4f}")

labels_rpy = ["roll", "pitch", "yaw"]
for i, name in enumerate(labels_rpy):
    if name == "yaw":
        rmse = compute_angle_rmse(sync_nav_rpy[:, i], mocap_rpy[:, i])
    else:
        rmse = compute_rmse(sync_nav_rpy[:, i], mocap_rpy[:, i])
    print(f"{name}: RMSE = {rmse:.4f}")

print("=====================================\n")

###########################################################
# 7. XY plot (Nav vs Mocap)
###########################################################
plt.figure(figsize=(8,8))
plt.title("XY Trajectory Comparison (Nav vs Mocap)")
plt.plot(mocap_pos[:,0], mocap_pos[:,1], label="Mocap", linewidth=2)
plt.plot(sync_nav_pos[:,0], sync_nav_pos[:,1], label="Nav (synced)", linewidth=2)
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.legend()
plt.grid()
plt.axis("equal")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"beetle_xy_trajectory.png"))

print(f"All images saved to: {output_dir}")
