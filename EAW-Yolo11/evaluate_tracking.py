"""
用「人工修正后的预标注」作为 GT，评估当前模型的跟踪与识别指标。

使用流程:
  1) 导出预标注: python3 run_feeding_track.py --source fish_video.mp4 --export-pre-annotation output/pre_gt_fish_video.csv
  2) 人工修正 pre_gt_fish_video.csv（改错的 track_id、class 等），另存为 output/gt_fish_video.csv
  3) 评估: python3 evaluate_tracking.py --gt output/gt_fish_video.csv --source fish_video.mp4 --model best.pt

依赖: pip install motmetrics
"""
import os
import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import motmetrics as mm
except ImportError:
    print("请安装 motmetrics: pip install motmetrics")
    raise

from ultralytics import YOLO


def load_csv_annot(path):
    """加载 CSV 标注，列: frame_id, track_id, x_min, y_min, x_max, y_max, class, [confidence]"""
    df = pd.read_csv(path)
    df = df.astype({"frame_id": int, "track_id": int, "x_min": float, "y_min": float, "x_max": float, "y_max": float})
    if "class" in df.columns:
        df["class"] = df["class"].astype(int)
    return df


def box_iou(boxes1, boxes2):
    """boxes: (N,4) x_min,y_min,x_max,y_max. 返回 (N,M) IoU."""
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    b = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = a[:, None] + b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def run_predictions(model_path, source, tracker="bytetrack.yaml", conf=0.25, iou=0.45):
    """跑模型得到与 GT CSV 同格式的预测；返回 (pred_df, num_frames, loop_elapsed)。
    loop_elapsed 为从第一帧到最后一帧的耗时（不含模型加载），用于算 FPS。"""
    model = YOLO(model_path)
    rows = []
    frame_id = 0
    t_loop_start = None
    for result in model.track(
        source=source,
        tracker=tracker,
        conf=conf,
        iou=iou,
        show=False,
        save=False,
        persist=True,
        stream=True,
        verbose=False,
    ):
        if t_loop_start is None:
            t_loop_start = time.perf_counter()
        frame_id += 1
        if result.boxes.id is None:
            continue
        ids = result.boxes.id.int().cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        clss = result.boxes.cls.int().cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        for tid, box, cls, c in zip(ids, boxes, clss, confs):
            x1, y1, x2, y2 = box
            rows.append({
                "frame_id": frame_id,
                "track_id": int(tid),
                "x_min": float(x1), "y_min": float(y1), "x_max": float(x2), "y_max": float(y2),
                "class": int(cls),
                "confidence": float(c),
            })
    t_loop_end = time.perf_counter()
    loop_elapsed = (t_loop_end - t_loop_start) if t_loop_start is not None else 0.0
    return pd.DataFrame(rows), frame_id, loop_elapsed


def compute_mot_metrics(gt_df, pred_df, iou_threshold=0.5):
    """
    gt_df / pred_df: 列含 frame_id, track_id, x_min, y_min, x_max, y_max.
    返回 (summary_df, acc).
    """
    acc = mm.MOTAccumulator(auto_id=True)
    frames = sorted(set(gt_df["frame_id"].tolist()) | set(pred_df["frame_id"].tolist()))

    for fid in frames:
        g = gt_df[gt_df["frame_id"] == fid]
        p = pred_df[pred_df["frame_id"] == fid]
        gt_ids = g["track_id"].values
        pred_ids = p["track_id"].values
        gt_boxes = g[["x_min", "y_min", "x_max", "y_max"]].values
        pred_boxes = p[["x_min", "y_min", "x_max", "y_max"]].values

        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            continue
        if len(gt_boxes) == 0:
            acc.update(gt_ids, pred_ids, np.zeros((0, len(pred_ids))))
            continue
        if len(pred_boxes) == 0:
            acc.update(gt_ids, pred_ids, np.zeros((len(gt_ids), 0)))
            continue

        iou = box_iou(gt_boxes, pred_boxes)
        # motmetrics 需要 cost/distance：1 - iou，越小越相似
        dist = 1.0 - iou
        acc.update(gt_ids, pred_ids, dist)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["mota", "motp", "idf1", "idp", "idr", "num_frames", "num_switches", "num_fragmentations"], name="track")
    return summary, acc


def compute_mt(gt_df, pred_df, iou_threshold=0.5, mt_ratio=0.8):
    """
    MT（Mostly Tracked）：有多少条真实轨迹被「同一预测 ID」跟住了 ≥ mt_ratio 的寿命（默认 80%）。
    每帧做 IoU 贪心匹配，统计每条 GT 轨迹被哪个 pred_id 覆盖了多少帧，取覆盖最多的 pred 占比 ≥ mt_ratio 则计为 MT。
    返回 (mt_count, num_gt_tracks)。
    """
    gt_track_frames = gt_df.groupby("track_id")["frame_id"].apply(lambda x: set(x)).to_dict()
    gt_coverage = {}  # gt_id -> { pred_id: set of frame_id }

    frames = sorted(set(gt_df["frame_id"].tolist()) | set(pred_df["frame_id"].tolist()))
    for fid in frames:
        g = gt_df[gt_df["frame_id"] == fid]
        p = pred_df[pred_df["frame_id"] == fid]
        gt_ids = g["track_id"].values
        pred_ids = p["track_id"].values
        gt_boxes = g[["x_min", "y_min", "x_max", "y_max"]].values
        pred_boxes = p[["x_min", "y_min", "x_max", "y_max"]].values
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            continue
        iou = box_iou(gt_boxes, pred_boxes)
        # 贪心匹配：按 IoU 从高到低配对
        used_pred = set()
        for _ in range(min(len(gt_boxes), len(pred_boxes))):
            best_iou = iou_threshold
            best_gi, best_pj = None, None
            for gi in range(len(gt_boxes)):
                for pj in range(len(pred_boxes)):
                    if pj in used_pred:
                        continue
                    if iou[gi, pj] >= best_iou:
                        best_iou = iou[gi, pj]
                        best_gi, best_pj = gi, pj
            if best_gi is None:
                break
            used_pred.add(best_pj)
            gt_id = int(gt_ids[best_gi])
            pred_id = int(pred_ids[best_pj])
            gt_coverage.setdefault(gt_id, {}).setdefault(pred_id, set()).add(fid)
            iou[best_gi, :] = -1
            iou[:, best_pj] = -1

    num_gt_tracks = len(gt_track_frames)
    mt_count = 0
    for gt_id, life in gt_track_frames.items():
        n_life = len(life)
        if n_life == 0:
            continue
        pred_frames = gt_coverage.get(gt_id, {})
        best_coverage = max((len(s) for s in pred_frames.values()), default=0)
        if best_coverage / n_life >= mt_ratio:
            mt_count += 1
    return mt_count, num_gt_tracks


def compute_class_metrics(gt_df, pred_df, iou_threshold=0.5, class_feed=0):
    """
    在 IoU 匹配上的基础上，统计「摄食/普通」分类的精确率、召回率。
    class_feed: GT/预测里表示摄食的类别号（0 或 1）。
    """
    gt_df = gt_df.copy()
    pred_df = pred_df.copy()
    if "class" not in gt_df.columns:
        return None
    if "class" not in pred_df.columns:
        return None

    frames = sorted(set(gt_df["frame_id"]) & set(pred_df["frame_id"]))
    tp_feed = fp_feed = fn_feed = 0
    tp_norm = fp_norm = fn_norm = 0

    for fid in frames:
        g = gt_df[gt_df["frame_id"] == fid]
        p = pred_df[pred_df["frame_id"] == fid]
        gt_boxes = g[["x_min", "y_min", "x_max", "y_max"]].values
        pred_boxes = p[["x_min", "y_min", "x_max", "y_max"]].values
        gt_cls = g["class"].values
        pred_cls = p["class"].values

        if len(gt_boxes) == 0:
            fp_feed += np.sum(pred_cls == class_feed)
            fp_norm += np.sum(pred_cls != class_feed)
            continue
        if len(pred_boxes) == 0:
            fn_feed += np.sum(gt_cls == class_feed)
            fn_norm += np.sum(gt_cls != class_feed)
            continue

        iou = box_iou(gt_boxes, pred_boxes)
        matched_gt, matched_pred = set(), set()
        for gi in range(len(gt_boxes)):
            for pi in range(len(pred_boxes)):
                if iou[gi, pi] >= iou_threshold and gi not in matched_gt and pi not in matched_pred:
                    matched_gt.add(gi)
                    matched_pred.add(pi)
                    if gt_cls[gi] == class_feed and pred_cls[pi] == class_feed:
                        tp_feed += 1
                    elif gt_cls[gi] == class_feed:
                        fn_feed += 1
                        fp_norm += 1
                    elif pred_cls[pi] == class_feed:
                        fp_feed += 1
                        fn_norm += 1
                    else:
                        tp_norm += 1
                    break
        for gi in range(len(gt_boxes)):
            if gi not in matched_gt:
                if gt_cls[gi] == class_feed:
                    fn_feed += 1
                else:
                    fn_norm += 1
        for pi in range(len(pred_boxes)):
            if pi not in matched_pred:
                if pred_cls[pi] == class_feed:
                    fp_feed += 1
                else:
                    fp_norm += 1

    def pr(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return p, r, f1

    p_feed, r_feed, f1_feed = pr(tp_feed, fp_feed, fn_feed)
    p_norm, r_norm, f1_norm = pr(tp_norm, fp_norm, fn_norm)
    return {
        "feeding": {"precision": p_feed, "recall": r_feed, "f1": f1_feed, "tp": tp_feed, "fp": fp_feed, "fn": fn_feed},
        "normal": {"precision": p_norm, "recall": r_norm, "f1": f1_norm, "tp": tp_norm, "fp": fp_norm, "fn": fn_norm},
    }


def main():
    parser = argparse.ArgumentParser(description="用修正后的 GT CSV 评估跟踪与识别")
    parser.add_argument("--gt", type=str, required=True, help="人工修正后的 GT CSV（与预标注同格式）")
    parser.add_argument("--source", type=str, required=True, help="视频路径（与 GT 对应）")
    parser.add_argument("--model", type=str, default="best.pt", help="模型路径")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml",
                    choices=["bytetrack.yaml", "botsort.yaml"],
                    help="跟踪算法：bytetrack.yaml / botsort.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="匹配用的 IoU 阈值")
    parser.add_argument("--save-pred", type=str, default=None, help="可选：保存本次预测 CSV 路径")
    args = parser.parse_args()

    if not os.path.isfile(args.gt):
        print(f"错误: GT 文件不存在 {args.gt}")
        return 1
    if not os.path.isfile(args.source):
        print(f"错误: 视频不存在 {args.source}")
        return 1
    if not os.path.isfile(args.model):
        print(f"错误: 模型不存在 {args.model}")
        return 1

    print("加载 GT ...")
    gt_df = load_csv_annot(args.gt)
    print(f"  GT 帧数: {gt_df['frame_id'].nunique()}, 框数: {len(gt_df)}")

    print("运行模型得到预测 ...")
    pred_df, num_frames, loop_elapsed = run_predictions(args.model, args.source, args.tracker, args.conf)
    fps = num_frames / loop_elapsed if loop_elapsed > 0 else 0
    print(f"  预测帧数: {num_frames}, 框数: {len(pred_df)}, 耗时: {loop_elapsed:.2f}s, FPS: {fps:.2f}")

    if args.save_pred:
        os.makedirs(os.path.dirname(args.save_pred) or ".", exist_ok=True)
        pred_df.to_csv(args.save_pred, index=False)
        print(f"  预测已保存: {args.save_pred}")

    print("\n计算跟踪指标 (MOTA, MOTP, IDF1, IDSW, Frag, MT, FPS) ...")
    summary, _ = compute_mot_metrics(gt_df, pred_df, args.iou_thresh)
    mt_count, num_gt_tracks = compute_mt(gt_df, pred_df, iou_threshold=args.iou_thresh, mt_ratio=0.8)
    print(summary.to_string())
    # 转为百分比显示（summary 单行，用 iloc[0] 取标量避免 FutureWarning）
    def _v(col):
        return float(summary[col].iloc[0]) if col in summary.columns else None
    if _v("mota") is not None:
        print(f"\n  MOTA = {_v('mota')*100:.2f}%")
    if _v("motp") is not None:
        print(f"  MOTP (avg IOU) = {(1 - _v('motp'))*100:.2f}%")
    if _v("idf1") is not None:
        print(f"  IDF1 = {_v('idf1')*100:.2f}%")
    print(f"  MT (Mostly Tracked, ≥80%) = {mt_count} / {num_gt_tracks}")
    print(f"  FPS = {fps:.2f}")

    if "class" in gt_df.columns and "class" in pred_df.columns:
        print("\n计算识别指标（摄食/普通）...")
        class_metrics = compute_class_metrics(gt_df, pred_df, args.iou_thresh, class_feed=0)
        if class_metrics:
            for name, m in [("摄食 (feeding)", class_metrics["feeding"]), ("普通 (normal)", class_metrics["normal"])]:
                print(f"  {name}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (tp={m['tp']} fp={m['fp']} fn={m['fn']})")

    print("\n评估完成。")
    return 0


if __name__ == "__main__":
    exit(main())
