"""
摄食行为目标追踪脚本
使用训练好的摄食检测模型(best.pt)对 fish_video.mp4 进行实时检测与追踪

运行方式（在已安装 ultralytics 的环境中）:
  cd EAW-Yolo11
  python3 run_feeding_track.py --model best.pt --source fish_video.mp4 --save 2>&1
"""
import os
import sys
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(description="摄食行为实时检测与追踪")
    parser.add_argument("--model", type=str, default="best.pt",
                        help="摄食检测模型路径 (默认: best.pt)")
    parser.add_argument("--source", type=str, default="fish_video.mp4",
                        help="视频路径 (默认: fish_video.mp4)")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml",
                        choices=["bytetrack.yaml", "botsort.yaml"],
                        help="追踪器类型")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="检测置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU阈值")
    parser.add_argument("--save", action="store_true",
                        help="是否保存输出视频")
    parser.add_argument("--no-show", action="store_true",
                        help="不显示窗口（仅保存时可用）")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="输出目录")
    parser.add_argument("--feeding-csv", type=str, default=None,
                        help="保存每条鱼摄食次数的 CSV 路径（默认: output/feeding_track/feeding_count.csv）")
    parser.add_argument("--min-frames", type=int, default=30,
                        help="最少出现帧数，少于此的轨迹视为误检并排除统计（默认: 30）")
    parser.add_argument("--export-pre-annotation", type=str, default=None, metavar="CSV_PATH",
                        help="导出预标注 CSV（便于人工修正后作 GT）：每行 frame_id,track_id,x_min,y_min,x_max,y_max,class,conf")
    args = parser.parse_args()

    model_path = args.model
    source = args.source

    if not os.path.isfile(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        return 1
    if not os.path.isfile(source):
        print(f"错误: 视频文件不存在 {source}")
        return 1

    print("=" * 60)
    print("摄食行为实时检测与追踪")
    print("=" * 60)
    print(f"模型: {model_path}")
    print(f"视频: {source}")
    print(f"追踪器: {args.tracker}")
    print(f"置信度阈值: {args.conf}")
    print("=" * 60)

    # 加载模型
    model = YOLO(model_path)

    # 类别名称（摄食=1, 普通状态=0）；model.names 为只读时使用训练时的类别名
    try:
        names = model.names
    except Exception:
        names = {1: "普通状态", 0: "摄食"}
    print(f"类别: {names}")
    print()

    # 每条鱼：摄食次数（被判定为摄食的帧数）、出现总帧数
    feeding_count = {}   # track_id -> 摄食帧数
    total_frames = {}   # track_id -> 出现总帧数
    CLASS_FEEDING = 0   # 类别 0 = 摄食（1 = 普通）

    # 无显示器时（SSH/无 DISPLAY）不弹窗，避免 qt.xcb 报错
    show_window = not args.no_show and bool(os.environ.get("DISPLAY"))
    if not args.no_show and not show_window:
        print("未检测到显示环境 (DISPLAY)，已关闭窗口显示；仅保存视频时请加 --no-show")

    # 预标注导出：写入 CSV 供人工修正后作 GT
    pre_ann_f = None
    if args.export_pre_annotation:
        pre_ann_path = args.export_pre_annotation
        os.makedirs(os.path.dirname(pre_ann_path) or ".", exist_ok=True)
        pre_ann_f = open(pre_ann_path, "w", encoding="utf-8")
        pre_ann_f.write("frame_id,track_id,x_min,y_min,x_max,y_max,class,confidence\n")
        print(f"预标注导出: {pre_ann_path}")

    # 若保存视频，先获取分辨率与 fps，并创建输出目录
    vid_writer = None
    if args.save:
        cap = cv2.VideoCapture(source)
        out_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        out_dir = os.path.join(args.output_dir, "feeding_track")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, Path(source).stem + ".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vid_writer = cv2.VideoWriter(out_path, fourcc, out_fps, (out_w, out_h))
        print(f"输出视频: {out_path}")

    # 执行追踪（不交给 ultralytics 写视频，由本脚本叠加摄食次数后写入）
    results = model.track(
        source=source,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        show=False,
        save=False,
        persist=True,
        stream=True,
        verbose=True,
    )

    frame_count = 0
    for result in results:
        frame_count += 1
        # 更新每条鱼的摄食/总帧数
        if result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().numpy()
            clss = result.boxes.cls.int().cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for tid, cls, box, conf in zip(ids, clss, boxes, confs):
                tid = int(tid)
                total_frames[tid] = total_frames.get(tid, 0) + 1
                if cls == CLASS_FEEDING:
                    feeding_count[tid] = feeding_count.get(tid, 0) + 1
                if pre_ann_f is not None:
                    x1, y1, x2, y2 = box
                    pre_ann_f.write(f"{frame_count},{tid},{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},{int(cls)},{conf:.4f}\n")

        # 用原图自己画框：摄食时刻框为红色，普通为绿色
        img = result.orig_img
        if img is None:
            continue
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = img.copy()

        if result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            clss = result.boxes.cls.int().cpu().numpy()
            for tid, box, cls in zip(ids, boxes, clss):
                tid = int(tid)
                x1, y1, x2, y2 = map(int, box)
                is_feeding = cls == CLASS_FEEDING
                # 摄食：红色框；普通：绿色框
                color = (0, 0, 255) if is_feeding else (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                # 标签：累计摄食次数 + 当前状态
                n = feeding_count.get(tid, 0)
                state = "Feeding" if is_feeding else "Normal"
                text = f"ID:{tid} Feed:{n} [{state}]"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 6, y1), (0, 0, 0), -1)
                cv2.putText(
                    img, text, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                )

        if args.save and vid_writer is not None:
            vid_writer.write(img)
        if show_window:
            cv2.imshow("feeding_track", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if frame_count % 30 == 0 and result.boxes.id is not None:
            print(f"帧 {frame_count}: 追踪到 {len(result.boxes.id)} 个目标")

    if vid_writer is not None:
        vid_writer.release()
    if show_window:
        cv2.destroyAllWindows()
    if pre_ann_f is not None:
        pre_ann_f.close()
        print(f"预标注已写入，请人工修正后保存为 GT 用于评估。")

    print()
    print("追踪完成。")
    # 输出每条鱼摄食次数
    print()
    print("=" * 60)
    print("每条鱼摄食行为统计（摄食次数 = 被判定为摄食的帧数）")
    print("=" * 60)
    if not feeding_count and not total_frames:
        print("未检测到目标。")
    else:
        all_ids = sorted(set(feeding_count.keys()) | set(total_frames.keys()))
        # 过滤误检：只保留出现帧数 >= min_frames 的轨迹
        min_f = args.min_frames
        valid_ids = [tid for tid in all_ids if total_frames.get(tid, 0) >= min_f]
        ignored = [tid for tid in all_ids if total_frames.get(tid, 0) < min_f]
        if ignored:
            print(f"  [已忽略 {len(ignored)} 条过短轨迹 (出现 < {min_f} 帧): ID {ignored}]")
            print()
        for tid in valid_ids:
            feed = feeding_count.get(tid, 0)
            total = total_frames.get(tid, 0)
            print(f"  鱼 ID {tid}: 出现 {total} 帧, 摄食 {feed} 帧 (摄食次数: {feed})")
        print("-" * 60)
        total_feed = sum(feeding_count.get(tid, 0) for tid in valid_ids)
        print(f"  有效轨迹 {len(valid_ids)} 条, 总摄食帧数 {total_feed}")

        # 保存 CSV（仅包含有效轨迹）
        csv_path = args.feeding_csv
        if csv_path is None:
            csv_path = os.path.join(args.output_dir, "feeding_count.csv")
        d = os.path.dirname(csv_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("track_id,total_frames,feeding_frames\n")
            for tid in valid_ids:
                f.write(f"{tid},{total_frames.get(tid, 0)},{feeding_count.get(tid, 0)}\n")
        print(f"  已保存: {csv_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
