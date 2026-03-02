#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rosbag から実験データを読み込み，timestamp に基づく線形補間による時間同期を行った上で
評価指標（RMSE）の算出およびプロットを行うスクリプト

- rosbag ファイルパス，開始・終了時間はユーザが直接編集
- プロット保存先ディレクトリをユーザが直接編集
- matplotlib により各内容を 1 枚ずつ PNG 出力
- /gimbalrotor/uav/nav (aerial_robot_msgs/FlightNav): 目標値
- /gimbalrotor/mocap/pose (geometry_msgs/PoseStamped): 計測値
  -> 位置(x,y,z)および姿勢(roll,pitch,yaw)の RMSE を算出（timestamp 同期）
- 目標値と計測値を x-y 平面で重ね描き
- /twin_hammer/mocap/pose: x-y 平面プロット
- /twin_hammer/haptics_wrench: 力・トルクをそれぞれプロット
"""

import os
import rosbag
import numpy as np
import math
import matplotlib.pyplot as plt
from tf.transformations import euler_from_quaternion

# ==============================
# ユーザ設定項目
# ==============================
BAG_PATH = "/home/kaneko/mthesis/rosbag/beetle_obstacle/fpv/20251227_142854_without_feedback.bag"   # rosbag への絶対パス
OUTPUT_DIR = "/home/kaneko/AIS/MSP-Latex-Template/rosbag_analysis/analyzed_images/beetle_obstacle/20251227_142854_without_feedback"
USE_FEEDBACK_FLAG = False

START_TIME = 55  # [s]
END_TIME   = 123  # [s]

NAV_TOPIC = "/gimbalrotor/uav/nav"
MOCAP_TOPIC = "/gimbalrotor/mocap/pose"
TWIN_MOCAP_TOPIC = "/twin_hammer/mocap/pose"
WRENCH_TOPIC = "/twin_hammer/haptics_wrench"

ideal_path_xy = np.array([
    [-1.5, -0.75],
    [ 0.75, -0.75],
    [ 0.75,  1.5]
])

# ==============================
# ユーティリティ関数
# ==============================

def quat_to_rpy(q):
    """geometry_msgs/Quaternion -> roll, pitch, yaw [rad]"""
    return euler_from_quaternion([q.x, q.y, q.z, q.w])

def rmse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return math.sqrt(np.mean((a - b) ** 2))

def interp_time_series(t_src, x_src, t_dst):
    """
    timestamp に基づく線形補間
    t_src: 元データの time [N]
    x_src: 元データの値 [N, dim]
    t_dst: 補間先 time [M]
    """
    t_src = np.asarray(t_src)
    x_src = np.asarray(x_src)
    t_dst = np.asarray(t_dst)

    x_dst = np.zeros((len(t_dst), x_src.shape[1]))
    for i in range(x_src.shape[1]):
        x_dst[:, i] = np.interp(t_dst, t_src, x_src[:, i])
    return x_dst

def point_to_segment_distance(p, a, b):
    """
    点 p と 線分 a-b の最短距離
    p, a, b: np.array([x, y])
    """
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0.0, 1.0)
    closest = a + t * ab
    return np.linalg.norm(p - closest)

def path_rmse(measured_xy, path_points):
    """
    理想軌跡（折れ線）と実測軌跡の RMSE
    measured_xy: [N, 2]
    path_points: [M, 2] (M>=2)
    """
    errors = []
    for p in measured_xy:
        d_min = np.inf
        for i in range(len(path_points) - 1):
            d = point_to_segment_distance(
                p,
                path_points[i],
                path_points[i + 1]
            )
            d_min = min(d_min, d)
        errors.append(d_min)
    errors = np.array(errors)
    return np.sqrt(np.mean(errors ** 2))

def project_point_to_path(p, path):
    """
    点 p を path（折れ線）に射影し，
    最も近い点の弧長位置 s を返す
    """
    s_total = 0.0
    s_proj_best = 0.0
    d_min = np.inf

    for i in range(len(path) - 1):
        a = path[i]
        b = path[i + 1]
        ab = b - a
        ap = p - a

        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0.0, 1.0)
        proj = a + t * ab
        d = np.linalg.norm(p - proj)

        if d < d_min:
            d_min = d
            s_proj_best = s_total + t * np.linalg.norm(ab)

        s_total += np.linalg.norm(ab)

    return s_proj_best, s_total

def path_coverage_ratio(measured_xy, path, threshold=0.2):
    """
    threshold: パス到達とみなす距離 [m]
    """
    s_max = 0.0
    total_length = None

    for p in measured_xy:
        s_proj, L = project_point_to_path(p, path)
        if total_length is None:
            total_length = L
        if np.linalg.norm(p - path[-1]) < threshold or s_proj > s_max:
            s_max = max(s_max, s_proj)

    return s_max / total_length



# ==============================
# データ格納用
# ==============================
nav_t = []
nav_pos = []
nav_rpy = []

mocap_t = []
mocap_pos = []
mocap_rpy = []

twin_pos = []

if USE_FEEDBACK_FLAG:
  wrench_t = []
  force = []
  torque = []

# ==============================
# rosbag 読み込み
# ==============================
bag = rosbag.Bag(BAG_PATH)
start_ros_time = None

for topic, msg, t in bag.read_messages(topics=[NAV_TOPIC, MOCAP_TOPIC, TWIN_MOCAP_TOPIC, WRENCH_TOPIC]):
    if start_ros_time is None:
        start_ros_time = t.to_sec()

    rel_t = t.to_sec() - start_ros_time
    if rel_t < START_TIME or rel_t > END_TIME:
        continue

    if topic == NAV_TOPIC:
        nav_t.append(rel_t)
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

    if topic == MOCAP_TOPIC:
        mocap_t.append(rel_t)
        p = msg.pose.position
        mocap_pos.append([p.x, p.y, p.z])
        mocap_rpy.append(list(quat_to_rpy(msg.pose.orientation)))

    if topic == TWIN_MOCAP_TOPIC:
        p = msg.pose.position
        twin_pos.append([p.x, p.y])

    if USE_FEEDBACK_FLAG:
      if topic == WRENCH_TOPIC:
          wrench_t.append(rel_t)
          f = msg.wrench.force
          tau = msg.wrench.torque
          force.append([f.x, f.y, f.z])
          torque.append([tau.x, tau.y, tau.z])

bag.close()

nav_t = np.array(nav_t)
mocap_t = np.array(mocap_t)
nav_pos = np.array(nav_pos)
nav_rpy = np.array(nav_rpy)
mocap_pos = np.array(mocap_pos)
mocap_rpy = np.array(mocap_rpy)
twin_pos = np.array(twin_pos)
if USE_FEEDBACK_FLAG:
  force = np.array(force)
  torque = np.array(torque)

# ==============================
# timestamp に基づく時間同期
# ==============================
# mocap の timestamp に nav を線形補間
nav_pos_sync = interp_time_series(nav_t, nav_pos, mocap_t)
nav_rpy_sync = interp_time_series(nav_t, nav_rpy, mocap_t)

# ==============================
# RMSE 計算
# ==============================
pos_labels = ["x", "y", "z"]
rpy_labels = ["roll", "pitch", "yaw"]

for i, l in enumerate(pos_labels):
    v = rmse(nav_pos_sync[:, i], mocap_pos[:, i])
    print(f"Position {l} RMSE: {v:.4f}")

for i, l in enumerate(rpy_labels):
    v = rmse(nav_rpy_sync[:, i], mocap_rpy[:, i])
    print(f"Attitude {l} RMSE: {v:.4f}")

# ==============================
# 理想軌跡 (x,y) と実測値の RMSE
# ==============================
measured_xy = mocap_pos[:, :2]

path_rmse_val = path_rmse(measured_xy, ideal_path_xy)
print(f"Ideal path (x,y) RMSE: {path_rmse_val:.4f}")
coverage = path_coverage_ratio(measured_xy, ideal_path_xy)
print(f"Path coverage ratio: {coverage:.4f}")


# ==============================
# プロット（PNG 保存）
# ==============================

# 目標値 vs 計測値 (x-y)
plt.figure()
plt.plot(nav_pos_sync[:, 0], nav_pos_sync[:, 1], label="Target")
plt.plot(mocap_pos[:, 0], mocap_pos[:, 1], label="Measured")
# wall plot
plt.plot([-1.5,1.5], [-1.5, -1.5], color="red", linewidth=3)
plt.plot([1.5,1.5], [-1.5, 1.5], color="red", linewidth=3)
plt.plot([-1.5,0.0], [0.0, 0.0], color="red", linewidth=3)
plt.plot([0.0,0.0], [0.0, 1.5], color="red", linewidth=3)
# ideal path
plt.plot(
    ideal_path_xy[:, 0],
    ideal_path_xy[:, 1],
    "--k",
    linewidth=1.5,
    label="Ideal path"
)
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.legend()
# plt.title("Gimbalrotor Target vs Measured (x-y)")
# plt.show()
plt.savefig(os.path.join(OUTPUT_DIR, "gimbalrotor_xy_target_vs_measured.png"))

# twin hammer mocap (x-y)
plt.figure()
plt.plot(twin_pos[:, 0], twin_pos[:, 1])
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
# plt.title("Twin Hammer Mocap Trajectory (x-y)")
# plt.savefig(os.path.join(OUTPUT_DIR, "twin_hammer_xy.png"))

if USE_FEEDBACK_FLAG:
  # wrench force
  plt.figure()
  plt.plot(wrench_t, force[:, 0], label="Fx")
  plt.plot(wrench_t, force[:, 1], label="Fy")
  plt.plot(wrench_t, force[:, 2], label="Fz")
  plt.xlabel("time [s]")
  plt.ylabel("Force")
  plt.grid(True)
  plt.legend()
#   plt.title("Haptics Wrench Force")
#   plt.savefig(os.path.join(OUTPUT_DIR, "haptics_wrench_force.png"))

  # wrench torque
  plt.figure()
  plt.plot(wrench_t, torque[:, 0], label="Tx")
  plt.plot(wrench_t, torque[:, 1], label="Ty")
  plt.plot(wrench_t, torque[:, 2], label="Tz")
  plt.xlabel("time [s]")
  plt.ylabel("Torque")
  plt.grid(True)
  plt.legend()
#   plt.title("Haptics Wrench Torque")
#   plt.savefig(os.path.join(OUTPUT_DIR, "haptics_wrench_torque.png"))

  # ==============================
  # wrench force & torque (1 figure, shared x-axis)
  # ==============================
  fig, (ax_force, ax_torque) = plt.subplots(
      2, 1, sharex=True, figsize=(8, 6)
  )

  # force (upper)
  ax_force.plot(wrench_t, force[:, 0], label="Fx")
  ax_force.plot(wrench_t, force[:, 1], label="Fy")
  ax_force.plot(wrench_t, force[:, 2], label="Fz")
  ax_force.set_ylabel("Force [N]")
  ax_force.grid(True)
  ax_force.legend()
  # torque (lower)
  ax_torque.plot(wrench_t, torque[:, 0], label="Tx")
  ax_torque.plot(wrench_t, torque[:, 1], label="Ty")
  ax_torque.plot(wrench_t, torque[:, 2], label="Tz")
  ax_torque.set_xlabel("time [s]")
  ax_torque.set_ylabel("Torque [Nm]")
  ax_torque.grid(True)
  ax_torque.legend()

  plt.tight_layout()
  plt.savefig(os.path.join(OUTPUT_DIR, "haptics_wrench_force_torque.png"))


print(f"All images saved to: {OUTPUT_DIR}")

plt.close('all')
