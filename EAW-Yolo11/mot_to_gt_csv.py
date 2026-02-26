"""
将 CVAT 导出的 MOT 格式（或任意 MOT 1.1 gt.txt）转回本项目的 GT CSV。

单类别 fish：MOT 第 8 列恒为 1，转成 CSV 后 class 列为 1；摄食/普通需在导出后由脚本或模型按帧再算（如跑一次分类）。

MOT gt.txt 每行 9 列: frame,id,x,y,w,h,1,class,keyframe（class 恒为 1）
  - 转为 CSV: frame_id, track_id, x_min, y_min, x_max, y_max, class（class=1 占位，后续脚本可按帧写 0/1）

用法:
  python3 mot_to_gt_csv.py output/mot_export/gt/gt.txt --out output/gt_fish_video.csv
"""
import argparse
import pandas as pd
from pathlib import Path


def read_mot_gt(path):
    """读取 MOT gt.txt，返回 DataFrame 列 frame_id, track_id, x_min, y_min, x_max, y_max, class"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame_id = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            # 单类别 fish，第 8 列恒为 1；摄食/普通由后续脚本按帧再算
            cls = int(float(parts[7])) if len(parts) > 7 else 1
            rows.append({
                "frame_id": frame_id,
                "track_id": track_id,
                "x_min": x,
                "y_min": y,
                "x_max": x + w,
                "y_max": y + h,
                "class": cls,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="MOT gt.txt 转 GT CSV")
    parser.add_argument("gt_txt", type=str, help="MOT 格式 gt.txt 路径（或 zip 内路径）")
    parser.add_argument("--out", type=str, default=None, help="输出 CSV 路径，默认与 gt.txt 同目录且名为 gt.csv")
    args = parser.parse_args()

    path = Path(args.gt_txt)
    if not path.exists():
        print(f"错误: 文件不存在 {path}")
        return 1

    df = read_mot_gt(path)
    if df.empty:
        print("错误: gt.txt 无有效行")
        return 1

    out_path = args.out or str(path.parent / "gt.csv")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"已写入: {out_path} (行数={len(df)})")
    return 0


if __name__ == "__main__":
    exit(main())
