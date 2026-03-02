#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rosbag
import numpy as np
import matplotlib.pyplot as plt
import math
import bisect
import os

###########################################################
#  ユーザー設定
###########################################################
rosbag_path = "/home/kaneko/mthesis/rosbag/device_wrench/2025-12-18-16-08-27_keyboard_command_random.bag"
output_dir  = "/home/kaneko/mthesis/rosbag/analysis_scripts/analized_images/device_wrench/2025-12-18-16-08-27_keyboard_command_random"

lpf_alpha = 0.1

start_time_sec = 8     # 評価開始時間 [s]
end_time_sec   = 135    # 評価終了時間 [s]

# --- Force sensor calibration (from calibration script) ---
# SENSOR_X_OFFSET [m] and SENSOR_Y_OFFSET [m] and FZ_OFFSET [N]
SENSOR_X_OFFSET = 0.053595
SENSOR_Y_OFFSET = -0.001366
FZ_OFFSET = 16.656539

###########################################################
#  出力先ディレクトリ
###########################################################
os.makedirs(output_dir, exist_ok=True)

###########################################################
#  便利関数
###########################################################
def in_time_window(t, t0):
    return start_time_sec <= (t - t0) <= end_time_sec

def compute_rmse(a, b):
    a = np.array(a)
    b = np.array(b)
    return math.sqrt(np.mean((a - b) ** 2))

def nearest_interp(base_times, base_data, target_time):
    """
    base_times : 昇順の時刻リスト
    base_data  : base_times に対応するデータ (N, dim)
    target_time: 補間したい時刻
    """
    idx = bisect.bisect_left(base_times, target_time)
    if idx == 0:
        return base_data[0]
    if idx >= len(base_times):
        return base_data[-1]

    before = idx - 1
    after  = idx
    if abs(base_times[before] - target_time) <= abs(base_times[after] - target_time):
        return base_data[before]
    else:
        return base_data[after]

def zoh_interp(base_times, base_data, target_times):
    """
    Zero-Order Hold interpolation

    base_times : 昇順の時刻リスト（haptics）
    base_data  : base_times に対応するデータ (N, dim)
    target_times : 補間したい時刻列（cfs_times）

    return:
        (len(target_times), dim)
    """
    result = np.zeros((len(target_times), base_data.shape[1]))

    idx = 0
    last_value = base_data[0]

    for i, t in enumerate(target_times):
        while idx + 1 < len(base_times) and base_times[idx + 1] <= t:
            idx += 1
            last_value = base_data[idx]

        result[i] = last_value

    return result

def first_order_lpf(data, alpha):
    y = np.zeros_like(data)
    y[0] = data[0]
    for k in range(1, len(data)):
        y[k] = (1 - alpha) * y[k-1] + alpha * data[k]
    return y

def apply_position_offset_correction(wrench, x_offset, y_offset, fz_offset):
    corrected = wrench.copy()
    Fz = wrench[:, 2]

    corrected[:, 2] -= fz_offset      # Fz
    corrected[:, 3] -= y_offset * Fz  # Tx
    corrected[:, 4] += x_offset * Fz  # Ty

    return corrected

###########################################################
#  データ格納
###########################################################
cfs_times = []
cfs_wrench = []

haptics_times = []
haptics_wrench = []

###########################################################
#  rosbag 読み込み
###########################################################
print("Loading rosbag:", rosbag_path)
bag = rosbag.Bag(rosbag_path, "r")
bag_start = bag.get_start_time()

for topic, msg, t in bag.read_messages():
    t_sec = t.to_sec()

    # 実測（CFS）
    if topic == "/cfs/data" and in_time_window(t_sec, bag_start):
        cfs_times.append(t_sec - bag_start)
        cfs_wrench.append([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z
        ])

    # 目標（Haptics）
    if topic == "/twin_hammer/haptics_wrench" and in_time_window(t_sec, bag_start):
        haptics_times.append(t_sec - bag_start)
        haptics_wrench.append([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z
        ])

bag.close()

###########################################################
# numpy 化
###########################################################
cfs_times = np.array(cfs_times)
cfs_wrench = np.array(cfs_wrench)

haptics_times = np.array(haptics_times)
haptics_wrench = np.array(haptics_wrench)

###########################################################
# LPF（CFS のみ）
###########################################################
cfs_wrench_lpf = first_order_lpf(cfs_wrench, lpf_alpha)

cfs_wrench_lpf = apply_position_offset_correction(
    cfs_wrench_lpf,
    SENSOR_X_OFFSET,
    SENSOR_Y_OFFSET,
    FZ_OFFSET
)

###########################################################
# CFS 時刻に合わせて目標値を補完
###########################################################
# haptics_interp = np.zeros_like(cfs_wrench_lpf)

# for i, t in enumerate(cfs_times):
#     haptics_interp[i] = nearest_interp(
#         haptics_times.tolist(),
#         haptics_wrench,
#         t
#     )

haptics_interp = zoh_interp(
    haptics_times,
    haptics_wrench,
    cfs_times
)

###########################################################
# プロット & RMSE 計算
###########################################################
labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]

print("===== RMSE (Target vs Measured) =====")
for i, label in enumerate(labels):
    rmse = compute_rmse(haptics_interp[:, i], cfs_wrench_lpf[:, i])
    print(f"{label}: RMSE = {rmse:.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(cfs_times, haptics_interp[:, i], label="Target", linestyle="--")
    plt.plot(cfs_times, cfs_wrench_lpf[:, i], label="Measured (LPF)")
    plt.xlabel("Time [s]")
    plt.ylabel(label)
    plt.title(f"{label} Target vs Measured")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(output_dir, f"{label}_target_vs_measured.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

###########################################################
# 縦6成分まとめプロット
###########################################################
fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)

for i, label in enumerate(labels):
    axes[i].plot(
        cfs_times,
        haptics_interp[:, i],
        linestyle="--",
        label="Target"
    )
    axes[i].plot(
        cfs_times,
        cfs_wrench_lpf[:, i],
        label="Measured (LPF)"
    )
    if i < 3:
        axes[i].set_ylabel(f"{label} [N]")
    else:
        axes[i].set_ylabel(f"{label} [Nm]")
    axes[i].yaxis.set_label_coords(-0.05, 0.5)
    axes[i].grid(True)

    if i == 0:
        axes[i].legend(loc="upper right")

axes[-1].set_xlabel("Time [s]")
fig.suptitle("Wrench Target vs Measured (All Components)", fontsize=14)

fig.tight_layout(rect=[0, 0.03, 1, 0.95])

save_path = os.path.join(
    output_dir,
    "wrench_all.png"
)
plt.savefig(save_path, dpi=150)
plt.close()

print(f"All images saved to: {output_dir}")
