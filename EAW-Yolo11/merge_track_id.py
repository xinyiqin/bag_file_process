"""
在预标注/GT CSV 中批量把某些 track_id 改为目标 id，便于「同一条鱼跟丢后以新 id 出现」时合并为同一 id。

CVAT 里无法方便地把一条轨迹的 id 批量改回之前的 id，在 CSV 里先改好再导入会更省事。

用法:
  python3 merge_track_id.py output/pre_gt_fish_video.csv --map 8:6 --out output/pre_gt_merged.csv
  # 把 track_id=8 全部改为 6（例如 6 跟丢后 reappear 成 8，合并回 6）
  python3 merge_track_id.py output/pre_gt.csv --map 8:6 --map 10:6 --out output/pre_gt_merged.csv
  # 把 8 和 10 都改为 6（多条轨迹合并为同一 id）
"""
import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="批量将 CSV 中指定 track_id 改为目标 id（可多组）"
    )
    parser.add_argument("csv", type=str, help="预标注/GT CSV 路径")
    parser.add_argument(
        "--map",
        type=str,
        action="append",
        metavar="FROM:TO",
        help="将 track_id=FROM 改为 TO，可多次指定，如 --map 8:6 --map 10:6",
    )
    parser.add_argument("--out", type=str, default=None, help="输出 CSV 路径，不指定则打印到 stdout")
    args = parser.parse_args()

    if not args.map:
        print("请至少指定一次 --map FROM:TO，例如 --map 8:6")
        return 1

    path = Path(args.csv)
    if not path.exists():
        print(f"错误: 文件不存在 {path}")
        return 1

    df = pd.read_csv(path)
    if "track_id" not in df.columns:
        print("错误: CSV 中无 track_id 列")
        return 1

    mapping = {}  # from_id -> to_id
    for s in args.map:
        if ":" not in s:
            print(f"错误: --map 格式应为 FROM:TO，例如 8:6，当前为 {s}")
            return 1
        from_id, to_id = s.strip().split(":", 1)
        try:
            from_id, to_id = int(from_id), int(to_id)
        except ValueError:
            print(f"错误: FROM 和 TO 须为整数，当前为 {s}")
            return 1
        mapping[from_id] = to_id

    orig = df["track_id"].copy()
    for from_id, to_id in mapping.items():
        cnt = (orig == from_id).sum()
        if cnt > 0:
            print(f"  track_id {from_id} -> {to_id}，修改行数: {cnt}")
    n_changed = orig.isin(mapping.keys()).sum()
    df["track_id"] = orig.replace(mapping)

    if n_changed > 0:
        print(f"共修改 {n_changed} 行")
    else:
        print("未匹配到待替换的 track_id，请检查 --map 中的 FROM 是否存在于 CSV")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"已写入: {out_path}")
    else:
        print(df.to_csv(index=False))

    return 0


if __name__ == "__main__":
    exit(main())
