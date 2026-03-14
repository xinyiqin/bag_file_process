import os
import cv2
import math
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
import matplotlib.pyplot as plt


# -----------------------------
# 工具函数
# -----------------------------
def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def angle(p1, p2, p3):

    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)

    ba = a - b
    bc = c - b

    cos = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)

    return np.degrees(np.arccos(cos))


def _draw_rounded_rect(img, x1, y1, x2, y2, color, thickness=2, radius=6):
    """绘制圆角矩形边框（用于标签背景或框）。"""
    h, w = img.shape[:2]
    radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2, max(1, radius))
    if thickness <= 0:
        return
    # 四角圆弧 + 四条边
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def _draw_detection_box_and_label(img, x1, y1, x2, y2, tid_int, feeding_events_count, is_feeding, color_bgr):
    """在画面上绘制检测框与标签：ID: x Feed: n（纯 ASCII 避免 OpenCV 中文乱码）。"""
    accent = color_bgr
    outline = (40, 40, 40)
    label_bg = (28, 28, 28)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    thickness = 1
    text = f"ID: {tid_int}  Feed: {feeding_events_count}"
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    label_w = tw + 12
    label_h = th + 8
    gap = 6
    if y1 - label_h - gap >= 0:
        ly1 = y1 - label_h - gap
    else:
        ly1 = min(y2 + gap, img.shape[0] - label_h - 2)
    ly1 = max(0, ly1)
    ly2 = ly1 + label_h
    lx1 = max(0, min(x1, img.shape[1] - label_w - 2))
    lx2 = lx1 + label_w
    overlay = img.copy()
    cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), label_bg, -1)
    cv2.addWeighted(overlay, 0.88, img, 0.12, 0, img)
    _draw_rounded_rect(img, lx1, ly1, lx2, ly2, accent, thickness=1, radius=3)
    tx, ty = lx1 + 6, ly1 + th + 4
    for (dx, dy) in [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)]:
        cv2.putText(img, text, (tx + dx, ty + dy), font, font_scale, outline, thickness)
    cv2.putText(img, text, (tx, ty), font, font_scale, accent, thickness)
    cv2.rectangle(img, (x1, y1), (x2, y2), outline, 4)
    cv2.rectangle(img, (x1, y1), (x2, y2), accent, 2)


# -----------------------------
# 绘制轨迹图
# -----------------------------
def draw_tracks(track_history, save_dir):

    plt.figure()

    for tid in track_history:

        pts = np.array(track_history[tid])

        if len(pts) > 1:
            plt.plot(pts[:,0], pts[:,1], linewidth=1)

    plt.gca().invert_yaxis()

    plt.title("Fish Movement Trajectories")

    plt.savefig(os.path.join(save_dir,"fish_tracks.png"),dpi=300)

    plt.close()


# -----------------------------
# 绘制热力图
# -----------------------------
def draw_heatmap(track_history, shape, save_dir):

    h,w = shape

    heat = np.zeros((h,w))

    for tid in track_history:

        for x,y in track_history[tid]:

            x=int(x)
            y=int(y)

            if 0<=x<w and 0<=y<h:
                heat[y,x]+=1

    plt.figure()

    plt.imshow(heat,cmap="jet")

    plt.colorbar()

    plt.title("Fish Activity Heatmap")

    plt.savefig(os.path.join(save_dir,"heatmap.png"),dpi=300)

    plt.close()


# -----------------------------
# 速度曲线
# -----------------------------
def draw_speed_curve(speed_df, save_dir):

    plt.figure()

    plt.plot(speed_df["frame"], speed_df["speed_cm_s"])

    plt.xlabel("Frame")

    plt.ylabel("Speed (cm/s)")

    plt.title("Speed Curve")

    plt.savefig(os.path.join(save_dir,"speed_curve.png"),dpi=300)

    plt.close()


# -----------------------------
# 区域统计图
# -----------------------------
def draw_zone_chart(zone_df, save_dir):

    plt.figure()

    plt.bar(zone_df["zone"], zone_df["feeding_count"])

    plt.xlabel("Zone")

    plt.ylabel("Feeding Count")

    plt.title("Zone Feeding Statistics")

    plt.savefig(os.path.join(save_dir,"zone_feeding.png"),dpi=300)

    plt.close()


# -----------------------------
# 分析流程（供 API 调用，返回可序列化数据）
# -----------------------------
def run_analysis(
    source,
    model_path,
    tracker="botsort.yaml",
    conf=0.25,
    iou=0.45,
    fps=30.0,
    scale=0.1,
    save_output_video=True,
    min_consecutive_frames=3,
    progress_callback=None,
    partial_callback=None,
    partial_interval=30,
):
    """
    对视频做轨迹跟踪与摄食分类，返回前端所需的全部数据（不写文件）。
    source: 视频路径；model_path: best.pt 路径。
    save_output_video: 是否生成带 ID 与摄食状态的预测视频（base64 放入返回）。
    min_consecutive_frames: 连续多少帧为摄食状态才计为一次摄食事件（默认 3）。
    progress_callback: 可选，每帧调用 callback(current_frame, total_frames) 用于进度条。
    partial_callback: 可选，每 partial_interval 帧调用 callback(partial_data) 用于流式推送图表数据。
    partial_interval: 每多少帧推送一次部分数据（默认 30）。
    返回 dict: coordinates, trajectory, feeding_stats, speed, ..., output_video_base64(可选)
    """

    def _py(obj):
        """Convert numpy scalars to native Python for JSON serialization."""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    import base64
    import tempfile

    model = YOLO(model_path)
    track_history = defaultdict(list)
    realtime_data = []
    speed_data = []
    accel_data = []
    angle_data = []
    feeding_count = defaultdict(int)  # 每帧若为摄食则+1（原始摄食帧数）
    consecutive_feeding = defaultdict(int)  # 当前连续摄食帧数
    feeding_events = defaultdict(int)  # 摄食次数：连续 min_consecutive_frames 帧摄食计为 1 次
    counted_this_run = defaultdict(bool)  # 当前这一段连续摄食是否已计过 1 次
    total_frames = defaultdict(int)
    zone_count = defaultdict(int)
    zone_feeding = defaultdict(int)
    zone_feeding_events = defaultdict(int)  # 各区域摄食次数（与总计摄食次数同口径：每完成 1 次事件 +1）
    CLASS_FEEDING = 0
    frame_shape = None

    # 获取视频 fps、尺寸与总帧数（用于进度回调）
    cap = cv2.VideoCapture(source)
    out_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_video = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()

    vid_writer = None
    out_video_path = None
    if save_output_video and out_w > 0 and out_h > 0:
        fd, out_video_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vid_writer = cv2.VideoWriter(out_video_path, fourcc, out_fps, (out_w, out_h))

    results = model.track(
        source=source,
        tracker=tracker,
        conf=conf,
        iou=iou,
        persist=True,
        stream=True,
    )
    frame_id = 0
    h = w = 0
    feeding_frequency_list = []  # 每帧处于摄食状态的鱼数 [{time, frequency}, ...]
    zone_over_time = []          # 每帧各区域鱼体数量 [{time, center, middle, edge}, ...]
    cumulative_feeding_events = 0  # 到当前帧为止的累计摄食次数（所有鱼、所有已完成事件之和）
    cumulative_feeding_events_over_time = []  # [{time, cumulative}, ...] 用于「总计摄食次数随时间变化」
    last_partial_at = 0  # 上次推送部分数据的帧号
    last_partial_speed_len = 0
    last_partial_accel_len = 0
    last_partial_angle_len = 0

    for r in results:

        frame_id += 1
        if progress_callback and total_frames_video > 0:
            progress_callback(frame_id, total_frames_video)

        frame = r.orig_img
        if frame is None:
            continue

        if frame_shape is None:
            frame_shape = frame.shape[:2]

        h, w = frame.shape[:2]
        cx0 = w / 2
        cy0 = h / 2
        r1 = min(w, h) * 0.2
        r2 = min(w, h) * 0.4

        # 每帧都画图并写入输出视频：无框时只写原图，有框时叠加 ID 与摄食状态
        if len(frame.shape) == 2:
            img_draw = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            img_draw = frame.copy()

        if r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy()
            clss = r.boxes.cls.int().cpu().numpy()

            # 本帧各区域鱼数、本帧摄食鱼数（用于实时图与区域变化图）
            zone_this_frame = {"center": 0, "middle": 0, "edge": 0}
            feeding_in_frame = int((clss == CLASS_FEEDING).sum())

            for tid, box, cls in zip(ids, boxes, clss):
                tid_int = int(tid)
                x1, y1, x2, y2 = map(int, box)
                is_feeding = int(cls) == CLASS_FEEDING
                total_frames[tid_int] = total_frames.get(tid_int, 0) + 1
                if is_feeding:
                    feeding_count[tid_int] = feeding_count.get(tid_int, 0) + 1
                    consecutive_feeding[tid_int] = consecutive_feeding.get(tid_int, 0) + 1
                    if consecutive_feeding[tid_int] >= min_consecutive_frames and not counted_this_run[tid_int]:
                        feeding_events[tid_int] = feeding_events.get(tid_int, 0) + 1
                        counted_this_run[tid_int] = True
                        cumulative_feeding_events += 1
                        cx_ev = (x1 + x2) / 2
                        cy_ev = (y1 + y2) / 2
                        d_zone_ev = distance((cx_ev, cy_ev), (cx0, cy0))
                        zone_ev = "center" if d_zone_ev < r1 else ("middle" if d_zone_ev < r2 else "edge")
                        zone_feeding_events[zone_ev] = zone_feeding_events.get(zone_ev, 0) + 1
                else:
                    consecutive_feeding[tid_int] = 0
                    counted_this_run[tid_int] = False

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                realtime_data.append([frame_id, tid, cx, cy, x2 - x1, y2 - y1])
                track_history[tid_int].append((cx, cy))

                if len(track_history[tid_int]) >= 2:
                    p1 = track_history[tid_int][-2]
                    p2 = track_history[tid_int][-1]
                    d = distance(p1, p2)
                    speed_data.append([frame_id, tid, d * scale * fps])
                if len(track_history[tid_int]) >= 3:
                    p0, p1, p2 = track_history[tid_int][-3], track_history[tid_int][-2], track_history[tid_int][-1]
                    angle_data.append([frame_id, tid, angle(p0, p1, p2)])
                    v1, v2 = distance(p0, p1), distance(p1, p2)
                    accel_data.append([frame_id, tid, (v2 - v1) * scale * fps])

                d_zone = distance((cx, cy), (cx0, cy0))
                zone = "center" if d_zone < r1 else ("middle" if d_zone < r2 else "edge")
                zone_count[zone] = zone_count.get(zone, 0) + 1
                zone_this_frame[zone] = zone_this_frame.get(zone, 0) + 1
                if is_feeding:
                    zone_feeding[zone] = zone_feeding.get(zone, 0) + 1

            feeding_frequency_list.append({"time": frame_id, "frequency": feeding_in_frame})
            cumulative_feeding_events_over_time.append({"time": frame_id, "cumulative": cumulative_feeding_events})
            zone_over_time.append({
                "time": frame_id,
                "center": zone_this_frame["center"],
                "middle": zone_this_frame["middle"],
                "edge": zone_this_frame["edge"],
            })

            # 绘制框与标签（美观样式）：摄食=红，普通=绿；Feed 为摄食次数
            for tid, box, cls in zip(ids, boxes, clss):
                tid_int = int(tid)
                x1, y1, x2, y2 = map(int, box)
                is_feeding = int(cls) == CLASS_FEEDING
                color_bgr = (0, 80, 255) if is_feeding else (0, 255, 100)  # BGR 稍柔和
                n = feeding_events.get(tid_int, 0)
                _draw_detection_box_and_label(img_draw, x1, y1, x2, y2, tid_int, n, is_feeding, color_bgr)

        else:
            feeding_frequency_list.append({"time": frame_id, "frequency": 0})
            cumulative_feeding_events_over_time.append({"time": frame_id, "cumulative": cumulative_feeding_events})
            zone_over_time.append({"time": frame_id, "center": 0, "middle": 0, "edge": 0})

        if vid_writer is not None:
            vid_writer.write(img_draw)

        # 流式推送部分数据（供前端实时更新图表）
        if partial_callback and frame_id - last_partial_at >= partial_interval:
            last_partial_at = frame_id
            # 各区域统计（当前累计）
            zone_stats_partial = [
                {"zone": z, "fish_count": int(zone_count[z]), "feeding_count": int(zone_feeding[z]), "feeding_events": int(zone_feeding_events.get(z, 0))}
                for z in ["center", "middle", "edge"]
            ]
            # 热力图（当前轨迹累计）
            heatmap_partial = []
            if frame_shape is not None:
                gh, gw = max(1, int(frame_shape[0]) // 20), max(1, int(frame_shape[1]) // 20)
                grid = np.zeros((gh, gw))
                for tid in track_history:
                    for x, y in track_history[tid]:
                        gi = min(int(y) * gh // max(1, int(frame_shape[0])), gh - 1)
                        gj = min(int(x) * gw // max(1, int(frame_shape[1])), gw - 1)
                        if 0 <= gi < gh and 0 <= gj < gw:
                            grid[gi, gj] += 1
                for i in range(gh):
                    for j in range(gw):
                        if grid[i, j] > 0:
                            heatmap_partial.append({"x": round(j * 100.0 / gw, 2), "y": round(i * 100.0 / gh, 2), "value": int(grid[i, j])})
            # 速度/加速度/转角：与其余时序一致，只推送增量，前端 append
            speed_slice = speed_data[last_partial_speed_len:]
            accel_slice = accel_data[last_partial_accel_len:]
            angle_slice = angle_data[last_partial_angle_len:]
            last_partial_speed_len = len(speed_data)
            last_partial_accel_len = len(accel_data)
            last_partial_angle_len = len(angle_data)
            speed_partial = [{"frame": _py(row[0]), "id": _py(row[1]), "speed_cm_s": round(float(row[2]), 4)} for row in speed_slice]
            accel_partial = [{"frame": _py(row[0]), "id": _py(row[1]), "acceleration_cm_s2": round(float(row[2]), 4)} for row in accel_slice]
            angle_partial = [{"frame": _py(row[0]), "id": _py(row[1]), "turn_angle": round(float(row[2]), 2)} for row in angle_slice]
            # 轨迹（当前全量）
            trajectory_partial = [{"id": _py(tid), "points": [{"x": _py(p[0]), "y": _py(p[1])} for p in track_history[tid]]} for tid in track_history]
            # 坐标与摄食统计（当前每 ID 最新）
            by_id = {}
            for row in realtime_data:
                by_id[_py(row[1])] = row
            coordinates_partial = [{"id": tid, "x": f"{r[2]:.2f}", "y": f"{r[3]:.2f}", "width": f"{r[4]:.2f}", "height": f"{r[5]:.2f}"} for tid, r in by_id.items()]
            feeding_stats_partial = [
                {"id": _py(tid), "total_frames": int(total_frames[tid]), "feeding_frames": int(feeding_count[tid]), "feeding_events": int(feeding_events.get(tid, 0))}
                for tid in total_frames
            ]
            # 时序三条推送增量（前端 append），与之前可正确流式显示区域鱼体数量变化的逻辑一致
            partial_data = {
                "zone_over_time": zone_over_time[-partial_interval:],
                "cumulative_feeding_events_over_time": cumulative_feeding_events_over_time[-partial_interval:],
                "feeding_frequency": feeding_frequency_list[-partial_interval:],
                "zone_stats": zone_stats_partial,
                "heatmap": heatmap_partial[:500] if len(heatmap_partial) > 500 else heatmap_partial,
                "speed": speed_partial,
                "acceleration": accel_partial,
                "angle": angle_partial,
                "trajectory": trajectory_partial,
                "coordinates": coordinates_partial,
                "feeding_stats": feeding_stats_partial,
            }
            partial_callback(partial_data)

    def _py(obj):
        """Convert numpy scalars to native Python for JSON serialization."""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # 每个 ID 只保留最新一帧的坐标（供前端“实时坐标”表显示）
    by_id = {}
    for row in realtime_data:
        by_id[_py(row[1])] = row
    coordinates = [{"id": tid, "x": f"{r[2]:.2f}", "y": f"{r[3]:.2f}", "width": f"{r[4]:.2f}", "height": f"{r[5]:.2f}"} for tid, r in by_id.items()]
    trajectory = [{"id": _py(tid), "points": [{"x": _py(p[0]), "y": _py(p[1])} for p in track_history[tid]]} for tid in track_history]
    feeding_stats = [
        {"id": _py(tid), "total_frames": int(total_frames[tid]), "feeding_frames": int(feeding_count[tid]), "feeding_events": int(feeding_events.get(tid, 0))}
        for tid in total_frames
    ]
    speed_list = [{"frame": _py(row[0]), "id": _py(row[1]), "speed_cm_s": round(float(row[2]), 4)} for row in speed_data]
    accel_list = [{"frame": _py(row[0]), "id": _py(row[1]), "acceleration_cm_s2": round(float(row[2]), 4)} for row in accel_data]
    angle_list = [{"frame": _py(row[0]), "id": _py(row[1]), "turn_angle": round(float(row[2]), 2)} for row in angle_data]
    zone_stats = [
        {"zone": z, "fish_count": int(zone_count[z]), "feeding_count": int(zone_feeding[z]), "feeding_events": int(zone_feeding_events.get(z, 0))}
        for z in ["center", "middle", "edge"]
    ]
    heatmap = []
    if frame_shape is not None:
        gh, gw = max(1, int(frame_shape[0]) // 20), max(1, int(frame_shape[1]) // 20)
        grid = np.zeros((gh, gw))
        for tid in track_history:
            for x, y in track_history[tid]:
                gi = min(int(y) * gh // max(1, int(frame_shape[0])), gh - 1)
                gj = min(int(x) * gw // max(1, int(frame_shape[1])), gw - 1)
                if 0 <= gi < gh and 0 <= gj < gw:
                    grid[gi, gj] += 1
        for i in range(gh):
            for j in range(gw):
                if grid[i, j] > 0:
                    heatmap.append({"x": round(j * 100.0 / gw, 2), "y": round(i * 100.0 / gh, 2), "value": int(grid[i, j])})
    # 摄食次数实时：每帧处于摄食状态的鱼数；限制长度便于前端
    feeding_frequency = feeding_frequency_list[:2000]
    cumulative_feeding_events_over_time_trim = cumulative_feeding_events_over_time[:2000]

    output_video_base64 = None
    if vid_writer is not None and out_video_path and os.path.isfile(out_video_path):
        try:
            vid_writer.release()
            with open(out_video_path, "rb") as f:
                output_video_base64 = base64.b64encode(f.read()).decode("ascii")
        finally:
            try:
                os.unlink(out_video_path)
            except Exception:
                pass

    zone_over_time_trim = zone_over_time[:2000]

    out = {
        "coordinates": coordinates,
        "trajectory": trajectory,
        "feeding_stats": feeding_stats,
        "speed": speed_list,
        "acceleration": accel_list,
        "angle": angle_list,
        "zone_stats": zone_stats,
        "zone_over_time": zone_over_time_trim,
        "cumulative_feeding_events_over_time": cumulative_feeding_events_over_time_trim,
        "heatmap": heatmap[:500] if len(heatmap) > 500 else heatmap,
        "feeding_frequency": feeding_frequency,
        "frame_shape": [int(frame_shape[0]), int(frame_shape[1])] if frame_shape is not None else None,
        "realtime_data": realtime_data,
        "track_history": {str(k): [[float(x), float(y)] for x, y in v] for k, v in track_history.items()},
        "speed_data": speed_data,
        "accel_data": accel_data,
        "angle_data": angle_data,
        "zone_count": dict(zone_count),
        "zone_feeding": dict(zone_feeding),
    }
    if output_video_base64 is not None:
        out["output_video_base64"] = output_video_base64
    return out


# -----------------------------
# 主程序（CLI）
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "..", "EAW-Yolo11", "weights", "best.pt"))
    parser.add_argument("--source", default=os.path.join(os.path.dirname(__file__), "..", "EAW-Yolo11", "output3min10_video.mp4"))
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--output", default="output")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    print("加载模型...")
    result = run_analysis(args.source, args.model, tracker=args.tracker, conf=args.conf, iou=args.iou, fps=args.fps, scale=args.scale)
    realtime_data = result["realtime_data"]
    track_history = {int(k): v for k, v in result["track_history"].items()}
    speed_data = result["speed_data"]
    frame_shape = tuple(result["frame_shape"]) if result.get("frame_shape") else None
    zone_count, zone_feeding = result["zone_count"], result["zone_feeding"]
    realtime_df = pd.DataFrame(realtime_data, columns=["frame", "id", "x", "y", "width", "height"])
    realtime_df.to_csv(f"{args.output}/realtime_coordinates.csv", index=False)
    speed_df = pd.DataFrame(speed_data, columns=["frame", "id", "speed_cm_s"])
    speed_df.to_csv(f"{args.output}/speed.csv", index=False)
    accel_df = pd.DataFrame(result["acceleration"], columns=["frame", "id", "acceleration_cm_s2"])
    accel_df.to_csv(f"{args.output}/acceleration.csv", index=False)
    angle_df = pd.DataFrame(result["angle"], columns=["frame", "id", "turn_angle"])
    angle_df.to_csv(f"{args.output}/turn_angle.csv", index=False)
    feeding_df = pd.DataFrame(result["feeding_stats"], columns=["id", "total_frames", "feeding_frames"])
    feeding_df.to_csv(f"{args.output}/feeding_statistics.csv", index=False)
    zone_df = pd.DataFrame(result["zone_stats"], columns=["zone", "fish_count", "feeding_count"])
    zone_df.to_csv(f"{args.output}/zone_statistics.csv", index=False)
    print("生成论文图...")
    draw_tracks(track_history, args.output)
    draw_heatmap(track_history, frame_shape, args.output)
    draw_speed_curve(speed_df, args.output)
    draw_zone_chart(zone_df, args.output)
    print("分析完成，输出目录:", args.output)


if __name__ == "__main__":
    main()