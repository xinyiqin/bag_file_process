"""
将预标注/GT CSV 转为 MOT 格式，便于导入 CVAT 做 track 与框的修正（不标摄食/普通）。

CVAT 一条 track 只能一个 label，故采用单类别 fish，第 8 列（class）恒为 1；摄食/普通在导出后用脚本或模型按帧再算。

gt.txt 每行 9 列：frame,id,x,y,w,h,1,class,keyframe
  - 第 8 列 class 恒为 1（单类别 fish）；第 9 列 keyframe：每 n 帧为 1.0 便于在 CVAT 中按关键帧筛选。
labels.txt 仅一行：fish。

用法:
  python3 export_csv_to_mot.py output/pre_gt_fish_video.csv --video fish_video.mp4 --out-dir output/mot_export --zip output/mot_fish_video.zip
  zip 内仅含 gt/；可加 --keyframe-every 10。
"""
import argparse
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def load_csv(path):
    df = pd.read_csv(path)
    df = df.astype({"frame_id": int, "track_id": int, "x_min": float, "y_min": float, "x_max": float, "y_max": float})
    if "class" in df.columns:
        df["class"] = df["class"].astype(int)
    else:
        df["class"] = 1  # 默认普通
    return df


def csv_to_mot_gt(df, keyframe_every=1, round_bbox=2):
    """MOT gt 行 9 列: frame,id,x,y,w,h,1,class,keyframe。单类别 fish，第 8 列 class 恒为 1。"""
    x_min = np.round(df["x_min"].values, round_bbox)
    y_min = np.round(df["y_min"].values, round_bbox)
    w = np.round((df["x_max"] - df["x_min"]).values, round_bbox)
    h = np.round((df["y_max"] - df["y_min"]).values, round_bbox)
    lines = []
    for i in range(len(df)):
        frame_id = int(df.iloc[i]["frame_id"])
        track_id = int(df.iloc[i]["track_id"])
        keyframe = 1.0 if (frame_id - 1) % keyframe_every == 0 else 0.0
        row = f"{frame_id},{track_id},{x_min[i]},{y_min[i]},{w[i]},{h[i]},1,1,{keyframe}"
        lines.append(row)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="CSV 转 MOT 格式供 CVAT 导入")
    parser.add_argument("csv", type=str, help="预标注/GT CSV（含 frame_id,track_id,x_min,y_min,x_max,y_max,class）")
    parser.add_argument("--video", type=str, default=None, help="对应视频路径，用于生成 seqinfo.ini 的尺寸/帧率")
    parser.add_argument("--out-dir", type=str, default="output/mot_export", help="输出目录，其下生成 <seq>/gt.txt")
    parser.add_argument("--seq-name", type=str, default=None, help="序列名，默认用 CSV 文件名（无后缀）")
    parser.add_argument("--zip", type=str, default=None, metavar="ZIP_PATH", help="若指定，则打包为 zip 便于上传 CVAT")
    parser.add_argument("--keyframe-every", type=int, default=1, metavar="N", help="每 N 帧将 keyframe 设为 1.0，便于在 CVAT 中按关键帧筛选；默认 1 即每帧都是关键帧")
    args = parser.parse_args()

    df = load_csv(args.csv)
    out_dir = Path(args.out_dir)
    gt_dir = out_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    gt_txt = csv_to_mot_gt(df, keyframe_every=args.keyframe_every)
    (gt_dir / "gt.txt").write_text(gt_txt, encoding="utf-8")
    print(f"已写入: {gt_dir / 'gt.txt'}（9 列：frame,id,x,y,w,h,1,class,keyframe）")
    (gt_dir / "labels.txt").write_text("fish\n", encoding="utf-8")
    print(f"已写入: {gt_dir / 'labels.txt'}（单类别 fish；第 8 列恒为 1，摄食/普通导出后再算）")

    if args.video and os.path.isfile(args.video):
        import cv2
        cap = cv2.VideoCapture(args.video)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        cap.release()

    if args.zip:
        zip_path = Path(args.zip)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(gt_dir.rglob("*")):
                if f.is_file():
                    arcname = Path("gt") / f.relative_to(gt_dir)
                    z.write(f, arcname)
        print(f"已打包: {zip_path}（解压后仅含 gt/ 文件夹：gt.txt、labels.txt）")

    print("下一步: CVAT 中「导入标注」选 MOT 格式，上传上述 zip。")
    return 0


if __name__ == "__main__":
    exit(main())
