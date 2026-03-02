#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rosbag
import numpy as np
import math

###########################################################
# ユーザー設定（必須）
###########################################################
ROSBAG_PATH = "/home/kaneko/mthesis/rosbag/device_wrench/2025-12-18-15-00-10_hovering_state.bag"
CFS_TOPIC   = "/cfs/data"

TARGET_FZ = 14.6   # [N] キャリブレーション時の既知鉛直力

START_TIME_SEC = 7.0    # bag 開始から [s]
END_TIME_SEC   = 58.0   # bag 開始から [s]

LPF_ALPHA = 0.1

###########################################################
# 便利関数
###########################################################
def first_order_lpf(data, alpha):
    y = np.zeros_like(data)
    y[0] = data[0]
    for k in range(1, len(data)):
        y[k] = (1 - alpha) * y[k-1] + alpha * data[k]
    return y

def in_time_window(t, t0):
    return START_TIME_SEC <= (t - t0) <= END_TIME_SEC

###########################################################
# データ読み込み
###########################################################
times = []
wrench = []

bag = rosbag.Bag(ROSABAG_PATH := ROSBAG_PATH, "r")
bag_start = bag.get_start_time()

for topic, msg, t in bag.read_messages(topics=[CFS_TOPIC]):
    t_sec = t.to_sec()
    if in_time_window(t_sec, bag_start):
        times.append(t_sec - bag_start)
        wrench.append([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z
        ])

bag.close()

if len(wrench) == 0:
    raise RuntimeError("指定時間範囲にデータがありません")

wrench = np.array(wrench)

###########################################################
# LPF
###########################################################
wrench_lpf = first_order_lpf(wrench, LPF_ALPHA)

Fx = wrench_lpf[:, 0]
Fy = wrench_lpf[:, 1]
Fz = wrench_lpf[:, 2]
Tx = wrench_lpf[:, 3]
Ty = wrench_lpf[:, 4]

###########################################################
# 推定
###########################################################
Fx_m = np.mean(Fx)
Fy_m = np.mean(Fy)
Fz_m = np.mean(Fz)
Tx_m = np.mean(Tx)
Ty_m = np.mean(Ty)

# 位置オフセット
x_offset = - Ty_m / TARGET_FZ
y_offset =   Tx_m / TARGET_FZ

# 姿勢（小角近似）
pitch = - Fx_m / TARGET_FZ
roll  =   Fy_m / TARGET_FZ
yaw   = 0.0

###########################################################
# 出力
###########################################################
print("========== Calibration Result ==========")
print(f"x_offset [m] : {x_offset:.6f}")
print(f"y_offset [m] : {y_offset:.6f}")
print(f"roll  [rad]  : {roll:.6f}")
print(f"pitch [rad]  : {pitch:.6f}")
print(f"yaw   [rad]  : {yaw:.6f}")
print(f"Hovering_thrust_offset [N]  : {Fz_m:.6f}")
print("========================================")

print("\n--- Paste the following into evaluation script ---")
print(f"SENSOR_X_OFFSET = {x_offset:.6f}")
print(f"SENSOR_Y_OFFSET = {y_offset:.6f}")
print(f"FZ_OFFSET = {Fz_m:.6f}")
