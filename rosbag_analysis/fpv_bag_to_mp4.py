#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import rosbag
import numpy as np
import cv2

# ==========================
# ユーザー設定
# ==========================
# rosbag の絶対パス
BAG_PATH = "/home/kaneko/mthesis/rosbag/me6_cleaning/fpv/20251229_162957_without_feedback.bag"

# 映像トピック名（sensor_msgs/Image）
IMAGE_TOPIC = "/usb_cam/image_raw"

# 出力ファイル名（拡張子 .mp4 を含む）
OUTPUT_FILENAME = "operater_camera_view.mp4"

# 出力ファイル保存先の絶対パス
OUTPUT_DIR = "/home/kaneko/mthesis/rosbag/analysis_scripts/analized_images/me6_cleaning/fpv/20251229_162957_without_feedback"

DEFAULT_FPS = 10.0
# ==========================


def estimate_fps(ts):
    if len(ts) < 2:
        return DEFAULT_FPS
    dt = np.median(np.diff(ts))
    return 1.0 / dt if dt > 0 else DEFAULT_FPS


def decode_image(msg):
    h, w = msg.height, msg.width
    enc = msg.encoding.lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ("yuyv", "yuy2"):
        img = buf.reshape((h, w, 2))
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUY2)

    elif enc == "uyvy":
        img = buf.reshape((h, w, 2))
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_UYVY)

    elif enc == "rgb8":
        return buf.reshape((h, w, 3))[:, :, ::-1]

    elif enc == "bgr8":
        return buf.reshape((h, w, 3))

    else:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    print("Opening rosbag...")
    bag = rosbag.Bag(BAG_PATH)

    frames = []
    timestamps = []

    print("Decoding sensor_msgs/Image frames...")
    for _, msg, t in bag.read_messages(topics=[IMAGE_TOPIC]):
        try:
            frame = decode_image(msg)
        except Exception:
            continue

        frames.append(frame)
        timestamps.append(t.to_sec())

    bag.close()

    if not frames:
        raise RuntimeError("No frames decoded (check encoding)")

    fps = estimate_fps(np.array(timestamps))
    print(f"Estimated FPS: {fps:.2f}")

    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print("Writing mp4...")
    for f in frames:
        writer.write(f)

    writer.release()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
