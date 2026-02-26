"""
对预标注 CSV 做自动检查，标出「可能有误」的 track_id，便于优先人工核对。

可疑情况：
  - 新 ID 突然出现：前一帧目标数已较多且稳定，本帧多了一个新 ID（可能是 ID 分裂/误检）。
  - 过短轨迹：该 ID 只出现很少帧，可能是误检或碎片。
  - 同帧重复 ID：同一帧同一 track_id 出现多次（异常）。
  - 突然消失的 ID：该 ID 在视频未结束前就不再出现（末次出现帧 < 视频最大帧），可能是跟丢或 ID 切换。

用法:
  python3 check_pre_annotation.py output/pre_gt_fish_video.csv
  python3 check_pre_annotation.py output/pre_gt_fish_video.csv --out-flagged output/flagged.csv --min-frames 30
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def load_csv(path):
    df = pd.read_csv(path)
    df = df.astype({"frame_id": int, "track_id": int})
    return df


def check_suspicious(df, min_frames=30, stable_count_window=5, count_increase_threshold=1):
    """
    返回带 suspicious_reason 列的 DataFrame，以及可疑项汇总。
    """
    df = df.sort_values(["frame_id", "track_id"]).reset_index(drop=True)
    df["suspicious_reason"] = ""

    # 每个 track_id 的首次/末次帧、出现帧数
    first_frame = df.groupby("track_id")["frame_id"].min()
    last_frame = df.groupby("track_id")["frame_id"].max()
    frame_count = df.groupby("track_id").size()

    # 每帧的目标数
    count_per_frame = df.groupby("frame_id").size().reindex(range(df["frame_id"].min(), df["frame_id"].max() + 1), fill_value=0)

    def add_reason(cond, reason):
        idx = df.index[cond]
        for i in idx:
            old = df.at[i, "suspicious_reason"]
            df.at[i, "suspicious_reason"] = f"{old};{reason}".strip(";") if old else reason

    # 1) 过短轨迹
    short_tracks = frame_count[frame_count < min_frames].index.tolist()
    for tid in short_tracks:
        add_reason(df["track_id"] == tid, f"short_track({frame_count[tid]}frames)")

    # 2) 新 ID 突然出现：该 ID 首次出现的那一帧，若前一帧目标数已经较多且本帧只多了 1 个新 ID，标为可疑
    for tid in df["track_id"].unique():
        f0 = first_frame.get(tid, None)
        if f0 is None or f0 <= 1:
            continue
        prev_count = count_per_frame.get(f0 - 1, 0)
        curr_count = count_per_frame.get(f0, 0)
        prev_ids = set(df[df["frame_id"] == f0 - 1]["track_id"])
        curr_ids = set(df[df["frame_id"] == f0]["track_id"])
        new_ids_this_frame = curr_ids - prev_ids
        if tid not in new_ids_this_frame:
            continue
        max_count = int(count_per_frame.max())
        if prev_count >= max(2, max_count * 0.5) and len(new_ids_this_frame) == 1 and curr_count == prev_count + 1:
            add_reason((df["track_id"] == tid) & (df["frame_id"] == f0), f"new_id_when_stable(prev={prev_count})")

    # 3) 同帧重复同一 track_id
    dup = df.groupby(["frame_id", "track_id"]).size()
    dup = dup[dup > 1]
    for (fid, tid), _ in dup.items():
        add_reason((df["frame_id"] == fid) & (df["track_id"] == tid), "duplicate_id_same_frame")

    # 4) 突然消失的 ID：该 track 末次出现帧 < 视频最大帧，仅在「消失的那一帧」（末次出现帧）标为可疑
    max_frame = df["frame_id"].max()
    for tid in df["track_id"].unique():
        last_f = last_frame.get(tid, None)
        if last_f is None or last_f >= max_frame:
            continue
        n_frames = frame_count.get(tid, 0)
        if n_frames < min_frames:
            continue  # 过短轨迹已标，不再重复标突然消失
        add_reason((df["track_id"] == tid) & (df["frame_id"] == last_f), f"id_suddenly_disappeared(last={last_f},video_end={max_frame})")

    # 汇总
    has_reason = df["suspicious_reason"].str.len() > 0
    suspicious_rows = df[has_reason]
    num_tracks_disappeared = len([tid for tid in df["track_id"].unique()
                                  if last_frame.get(tid, 0) < max_frame and frame_count.get(tid, 0) >= min_frames])
    summary = {
        "short_track": len(short_tracks),
        "new_id_when_stable": int((suspicious_rows["suspicious_reason"].str.contains("new_id_when_stable", na=False)).sum() > 0 and 1 or 0),
        "duplicate_id": dup.shape[0],
        "id_suddenly_disappeared": int(num_tracks_disappeared),
        "total_flagged_rows": has_reason.sum(),
    }
    return df, summary, short_tracks


def main():
    parser = argparse.ArgumentParser(description="预标注 CSV 自动检查可疑 track_id")
    parser.add_argument("csv", type=str, help="预标注 CSV 路径")
    parser.add_argument("--min-frames", type=int, default=30, help="少于此帧数的轨迹标为过短（默认 30）")
    parser.add_argument("--out-flagged", type=str, default=None, help="输出带 suspicious_reason 列的 CSV")
    parser.add_argument("--out-report", type=str, default=None, help="输出文字报告到该文件")
    args = parser.parse_args()

    df = load_csv(args.csv)
    df, summary, short_tracks = check_suspicious(df, min_frames=args.min_frames)

    lines = []
    lines.append("=" * 60)
    lines.append("预标注可疑项检查报告")
    lines.append("=" * 60)
    lines.append(f"总行数: {len(df)}, 总帧数: {df['frame_id'].nunique()}, 轨迹数: {df['track_id'].nunique()}")
    lines.append("")
    lines.append("【自动标出的可疑情况】")
    lines.append(f"  1) 过短轨迹 (出现 < {args.min_frames} 帧): {len(short_tracks)} 条, ID = {short_tracks[:20]}{'...' if len(short_tracks) > 20 else ''}")
    lines.append(f"  2) 新 ID 在目标数已较多时突然出现: 见下方明细或 CSV 列 suspicious_reason")
    lines.append(f"  3) 同帧重复同一 ID: {summary['duplicate_id']} 处")
    lines.append(f"  4) 突然消失的 ID (末次出现 < 视频末帧): {summary['id_suddenly_disappeared']} 条")
    lines.append(f"  涉及行数（含多类）: {summary['total_flagged_rows']}")
    lines.append("")
    flagged = df[df["suspicious_reason"].str.len() > 0]
    if len(flagged) > 0:
        lines.append(f"【可疑行明细（全部 {len(flagged)} 行）】")
        for _, row in flagged.iterrows():
            lines.append(f"  frame_id={row['frame_id']} track_id={row['track_id']} reason={row['suspicious_reason']}")
    lines.append("=" * 60)

    report_text = "\n".join(lines)
    print(report_text)

    if args.out_report:
        Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_report, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"报告已保存: {args.out_report}")

    if args.out_flagged:
        Path(args.out_flagged).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_flagged, index=False)
        print(f"带 suspicious_reason 的 CSV 已保存: {args.out_flagged}")

    return 0


if __name__ == "__main__":
    exit(main())
