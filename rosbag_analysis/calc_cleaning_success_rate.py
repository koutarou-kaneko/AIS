import cv2
import numpy as np
import os

# ------------------------------
# 設定：ユーザーが指定する4点（0〜1 の比率）
# 順番：左上 → 右上 → 右下 → 左下
# ------------------------------
REGION_POINTS_RATIO = np.array([
    [0.29, 0.50],  # 左上（10%, 15%）
    [0.62, 0.50],  # 右上
    [0.62, 0.70],  # 右下
    [0.29, 0.70],  # 左下
], dtype=np.float32)
# ------------------------------
# うねり解析用パラメータ
# ------------------------------
BIN_RATIO = 0.01      # 領域長に対するビン幅割合（例：5%）
SMOOTH_WINDOW = 1    # 移動平均窓（奇数）
# ------------------------------
# 局所平行度（うねり）評価用
# ------------------------------
NUM_WINDOWS = 5        # 評価領域の分割数（縦長）
MIN_PIXELS_PER_WIN = 50 # PCAを計算する最小画素数


def ratio_to_pixels(ratio_points, width, height):
    """
    0〜1 の比率 → ピクセル座標へ変換
    """
    pts = []
    for rx, ry in ratio_points:
        x = int(rx * width)
        y = int(ry * height)
        pts.append([x, y])
    return np.array(pts, dtype=np.int32)


def resize_keep_ratio(img, long_side=800):
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale))), scale


def align_images_ecc_fast(before_gray, after_gray):
    before_small, _ = resize_keep_ratio(before_gray)
    after_small, _ = resize_keep_ratio(after_gray)

    h, w = before_small.shape
    after_small = cv2.resize(after_small, (w, h))

    before_f = before_small.astype(np.float32) / 255.0
    after_f  = after_small.astype(np.float32) / 255.0

    warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        300,
        1e-6
    )

    try:
        cc, warp_matrix = cv2.findTransformECC(
            before_f, after_f, warp_matrix,
            motionType=cv2.MOTION_AFFINE,
            criteria=criteria
        )
    except cv2.error:
        print("[WARN] ECC alignment failed → identity")
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    H, W = before_gray.shape
    aligned_after = cv2.warpAffine(after_gray, warp_matrix, (W, H),
                                   flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return aligned_after, warp_matrix


def create_polygon_mask(shape, polygon_points):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_points], 255)
    return mask


def extract_red_line_mask(img_bgr):
    """
    赤色テープ（目標線）をHSVで抽出
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 赤は Hue が 0 付近と 180 付近に分かれる
    lower1 = np.array([0, 80, 80])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 80, 80])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    red_mask = cv2.bitwise_or(mask1, mask2)

    return red_mask

def skeletonize(mask):
    """
    Zhang-Suen thinning (OpenCV版簡易)
    """
    skel = np.zeros(mask.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    img = mask.copy()

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        done = cv2.countNonZero(img) == 0

    return skel

def principal_direction(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return None
    pts = np.column_stack((xs, ys)).astype(np.float32)
    mean, eigvecs = cv2.PCACompute(pts, mean=None)
    v = eigvecs[0]
    return v / np.linalg.norm(v)

def ordered_skeleton_points(skel):
    """
    スケルトン画素を y→x の順で並べる（単調曲線を想定）
    """
    ys, xs = np.where(skel > 0)
    pts = np.column_stack((xs, ys))

    # 主方向でソート（PCA 第1主成分方向）
    mean, eigvecs = cv2.PCACompute(pts.astype(np.float32), mean=None)
    v = eigvecs[0]
    proj = pts @ v
    order = np.argsort(proj)

    return pts[order]

def moving_average(x, window):
    window = max(3, window | 1)  # 奇数化
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")

def project_points(xs, ys, origin, v_axis):
    """
    点群を指定軸に射影
    """
    pts = np.column_stack((xs, ys)) - origin
    return pts @ v_axis

def extract_upper_lower_edges(diff_bin, v_red, mask):
    ys, xs = np.where((diff_bin > 0) & (mask > 0))
    if len(xs) < 50:
        return None

    v_red = v_red / np.linalg.norm(v_red)
    n_red = np.array([-v_red[1], v_red[0]])

    origin = np.array([xs.mean(), ys.mean()])

    s = project_points(xs, ys, origin, v_red)
    d = project_points(xs, ys, origin, n_red)

    # ビニング
    s_min, s_max = s.min(), s.max()
    bin_width = BIN_RATIO * (s_max - s_min)
    bins = np.arange(s_min, s_max, bin_width)

    s_centers = []
    d_upper = []
    d_lower = []

    for b0, b1 in zip(bins[:-1], bins[1:]):
        idx = (s >= b0) & (s < b1)
        if np.sum(idx) < 5:
            continue
        s_centers.append((b0 + b1) / 2)
        d_upper.append(np.max(d[idx]))
        d_lower.append(np.min(d[idx]))

    return (
        np.array(s_centers),
        np.array(d_upper),
        np.array(d_lower)
    )

def smooth_and_count_turns(d, window):
    d_s = moving_average(d, window)
    sign = np.sign(d_s)
    sign[sign == 0] = np.nan
    turns = np.nansum(np.diff(np.sign(sign)) != 0)
    return d_s, int(turns)

def local_parallelism_variation(
        paint_bin,
        mask,
        v_red,
        num_windows,
        min_pixels=50
    ):
    """
    赤線主方向 v_red を基準に、評価領域を縦長ウィンドウに分割し、
    各ウィンドウでの局所平行度誤差を算出する
    """

    if v_red is None:
        return None, None, None

    v_red = v_red / np.linalg.norm(v_red)
    n_red = np.array([-v_red[1], v_red[0]])

    # 塗装画素（評価対象）
    ys, xs = np.where((paint_bin > 0) & (mask > 0))
    if len(xs) < min_pixels:
        return None, None, None

    pts = np.column_stack((xs, ys)).astype(np.float32)
    origin = pts.mean(axis=0)

    # 赤線方向に射影
    s = (pts - origin) @ v_red

    s_min, s_max = s.min(), s.max()
    bins = np.linspace(s_min, s_max, num_windows + 1)

    local_angles = []
    window_centers = []

    for b0, b1 in zip(bins[:-1], bins[1:]):
        idx = (s >= b0) & (s < b1)
        if np.sum(idx) < min_pixels:
            continue

        pts_w = pts[idx]

        # PCAで局所主方向
        mean, eigvecs = cv2.PCACompute(pts_w, mean=None)
        v_local = eigvecs[0]
        v_local /= np.linalg.norm(v_local)

        # 赤線との角度
        cos_angle = np.clip(np.dot(v_local, v_red), -1.0, 1.0)
        theta = np.degrees(np.arccos(cos_angle))

        local_angles.append(theta)
        window_centers.append((b0 + b1) / 2)

    if len(local_angles) < 3:
        return None, None, None

    local_angles = np.array(local_angles)

    # --- うねり指標 ---
    mean_angle = np.mean(local_angles)
    std_angle  = np.std(local_angles)
    diff_angle = np.mean(np.abs(np.diff(local_angles)))

    return mean_angle, std_angle, diff_angle


def calc_cleaned_ratio(
        before_path,
        after_path,
        thresh=25,
        save_dir="output",
        save_images=True
    ):

    if save_images:
        os.makedirs(save_dir, exist_ok=True)

    # ------------------------------
    # 画像読み込み
    # ------------------------------
    before = cv2.imread(before_path)
    after  = cv2.imread(after_path)

    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray  = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    H, W = before_gray.shape

    # ------------------------------
    # 💡 比率 → ピクセル座標へ変換
    # ------------------------------
    region_points = ratio_to_pixels(REGION_POINTS_RATIO, W, H)

    # ------------------------------
    # 位置合わせ
    # ------------------------------
    aligned_after, warp_matrix = align_images_ecc_fast(before_gray, after_gray)

    # ★ カラー画像にも同じ warp を適用
    aligned_after_color = cv2.warpAffine(
        after,
        warp_matrix,
        (W, H),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    )

    # ------------------------------
    # 差分
    # ------------------------------
    diff = cv2.absdiff(before_gray, aligned_after)

    # ------------------------------
    # マスク適用
    # ------------------------------
    mask = create_polygon_mask(before_gray.shape, region_points)
    diff_masked = cv2.bitwise_and(diff, diff, mask=mask)
    red_line_mask = extract_red_line_mask(aligned_after_color)
    red_line_mask = cv2.bitwise_and(red_line_mask, red_line_mask, mask=mask)

    # ------------------------------
    # 二値化
    # ------------------------------
    _, diff_bin = cv2.threshold(diff_masked, thresh, 255, cv2.THRESH_BINARY)

    # ------------------------------
    # 成功率（領域内のみ）
    # ------------------------------
    area_pixels = np.sum(mask > 0)
    cleaned_pixels = np.sum(diff_bin > 0)
    ratio = cleaned_pixels / area_pixels * 100.0

    # 赤線上の画素数
    red_pixels = np.sum(red_line_mask > 0)

    # 赤線かつ塗装検出された画素
    covered_pixels = np.sum(
        (red_line_mask > 0) & (diff_bin > 0)
    )

    line_coverage = covered_pixels / red_pixels * 100.0 if red_pixels > 0 else 0.0

    red_skel = skeletonize(red_line_mask)

    # 距離変換（赤線からの距離）
    # dist_map = cv2.distanceTransform(
    #     cv2.bitwise_not(red_skel),
    #     cv2.DIST_L2,
    #     5
    # )
    paint_mask = diff_bin > 0
    paint_mask_u8 = paint_mask.astype(np.uint8) * 255

    dist_map = cv2.distanceTransform(
        cv2.bitwise_not(paint_mask_u8),
        cv2.DIST_L2,
        5
    )

    paint_pixels = diff_bin > 0
    lateral_distances = dist_map[paint_pixels]

    mean_lateral_distance = (
        np.mean(lateral_distances)
        if lateral_distances.size > 0 else np.nan
    )

    v_red = principal_direction(red_skel)
    v_paint = principal_direction(diff_bin)

    if v_red is not None and v_paint is not None:
        cos_angle = np.clip(np.dot(v_red, v_paint), -1.0, 1.0)
        parallel_angle_deg = np.degrees(np.arccos(cos_angle))
    else:
        parallel_angle_deg = np.nan


    # --- うねり解析 ---
    edges = extract_upper_lower_edges(diff_bin, v_red, mask)

    if edges is not None:
        s_axis, d_top, d_bottom = edges

        d_top_s, top_turns = smooth_and_count_turns(d_top, SMOOTH_WINDOW)
        d_bot_s, bot_turns = smooth_and_count_turns(d_bottom, SMOOTH_WINDOW)
    else:
        top_turns = bot_turns = np.nan

    # ------------------------------
    # 局所平行度変動（うねり評価）
    # ------------------------------
    mean_local_angle, std_local_angle, diff_local_angle = (
        local_parallelism_variation(
            diff_bin,
            mask,
            v_red,
            NUM_WINDOWS,
            MIN_PIXELS_PER_WIN
        )
    )



    # ------------------------------
    # 保存処理
    # ------------------------------
    if save_images:
        cv2.imwrite(os.path.join(save_dir, "before.png"), before)
        cv2.imwrite(os.path.join(save_dir, "after.png"), after)
        cv2.imwrite(os.path.join(save_dir, "aligned_after.png"), aligned_after)

        cv2.imwrite(os.path.join(save_dir, "mask_region.png"), mask)
        cv2.imwrite(os.path.join(save_dir, "diff_masked.png"), diff_masked)
        cv2.imwrite(os.path.join(save_dir, "red_line_mask.png"), red_line_mask)
        cv2.imwrite(os.path.join(save_dir, "diff_bin_masked.png"), diff_bin)

        # before に領域枠を描画
        before_vis = before.copy()
        cv2.polylines(before_vis, [region_points], True, (0, 255, 0), 3)
        cv2.imwrite(os.path.join(save_dir, "before_region.png"), before_vis)

        print(f"[保存完了] 画像は {save_dir}/ に保存されました")

    return (
        ratio,
        line_coverage,
        mean_lateral_distance,
        parallel_angle_deg,
        top_turns,
        bot_turns,
        mean_local_angle,
        std_local_angle,
        diff_local_angle,
        diff_masked,
        diff_bin,
        aligned_after,
        mask
    )


# ----------------------------------------
# 使用例
# ----------------------------------------
before_img = "/home/kaneko/mthesis/rosbag/me6_cleaning/figs/fpv/wall_before_20251229.jpg"
after_img  = "/home/kaneko/mthesis/rosbag/me6_cleaning/figs/fpv/wall_after_20251229_154243_without_feedback.jpg"

ratio, line_cov, mean_lateral_distance, parallel_angle_deg, top_turns, bot_turns, mean_local_angle, std_local_angle, diff_local_angle, diff_masked, diff_bin, aligned, mask = calc_cleaned_ratio(
    before_img,
    after_img,
    thresh=60,
    save_dir="/home/kaneko/mthesis/rosbag/analysis_scripts/analized_images/beetle_cleaning/fpv/20251222_194249_without_energy",
    save_images=False
)

print(f"拭き取り成功率（指定領域のみ）: {ratio:.2f}%")
print(f"赤線追従率: {line_cov:.2f}%")
print(f"平均横方向距離: {mean_lateral_distance:.2f} px")
print(f"平行度誤差: {parallel_angle_deg:.2f} deg")
print(f"上端うねり回数: {top_turns}")
print(f"下端うねり回数: {bot_turns}")

print(f"局所平行度平均: {mean_local_angle:.2f} deg")
print(f"局所平行度変動(STD): {std_local_angle:.2f} deg")
print(f"局所平行度差分平均: {diff_local_angle:.2f} deg")


