#!/usr/bin/env python3
"""
从已提取的图像帧中按需生成部分时长的 MP4 视频

用法示例：

    python create_partial_video.py \
        /path/to/output/20251104test/images \
        --fps 30 \
        --start-sec 600 \
        --duration-sec 600 \
        --output /path/to/output/20251104test/segment_10min.mp4

脚本会从指定目录中按 frame_000000.png 的命名规则读取图像，
从给定起始时间（或起始帧）开始，生成一定时长（或帧数）的影片。
"""

import argparse
import os
import subprocess
from pathlib import Path
from typing import List, Optional

import cv2

FRAME_EXTENSION = ".png"


def get_frame_filename(index: int) -> str:
    return f"frame_{index:06d}{FRAME_EXTENSION}"


def find_first_existing_frame(images_dir: Path, start_index: int, end_index: int) -> Optional[int]:
    """在指定范围内寻找首个存在的帧索引。"""
    for idx in range(start_index, end_index):
        if (images_dir / get_frame_filename(idx)).exists():
            return idx
    return None


def open_video_writer(video_path: Path, fps: float, size) -> Optional[cv2.VideoWriter]:
    """尝试按不同编码打开 VideoWriter。"""
    fourcc_options = [
        ('avc1', 'H.264'),
        ('h264', 'H.264'),
        ('X264', 'X.264'),
        ('mp4v', 'MPEG-4'),
    ]

    video_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = str(video_path) + '.tmp'

    for codec, name in fourcc_options:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(temp_path, fourcc, fps, size)
            if writer.isOpened():
                print(f"使用编码器: {name} ({codec})")
                return writer
            writer.release()
        except Exception:
            continue

    print("警告: 所有编码器尝试失败。")
    return None


def collect_frame_paths(images_dir: Path, start_frame: int, frame_count: int) -> List[Path]:
    end_frame = start_frame + frame_count
    frame_paths: List[Path] = []

    for idx in range(start_frame, end_frame):
        frame_path = images_dir / get_frame_filename(idx)
        if not frame_path.exists():
            print(f"警告: 缺少帧 {frame_path.name}，在此截断。")
            break
        frame_paths.append(frame_path)

    if not frame_paths:
        raise FileNotFoundError("指定范围内未找到任何帧文件")

    return frame_paths


def create_video_with_ffmpeg(frame_paths: List[Path], video_path: Path, fps: float) -> None:
    print("尝试使用 ffmpeg 创建视频...")

    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("未找到 ffmpeg，可安装后重试，或在容器内安装 ffmpeg。")

    list_file = video_path.with_suffix('.frames.txt')

    def quote_path(path: Path) -> str:
        return str(path).replace("'", "'\\''")

    try:
        with open(list_file, 'w', encoding='utf-8') as f:
            for frame_path in frame_paths:
                f.write(f"file '{quote_path(frame_path)}'\n")
                f.write(f"duration {1.0 / fps}\n")
            # 重复最后一帧，确保持续时间
            f.write(f"file '{quote_path(frame_paths[-1])}'\n")

        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(list_file),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            str(video_path)
        ]

        subprocess.run(cmd, check=True)
    finally:
        if list_file.exists():
            list_file.unlink()


def build_video_from_frames(images_dir: Path, video_path: Path, start_frame: int, frame_count: int, fps: float) -> None:
    if frame_count <= 0:
        raise ValueError("frame_count 必须为正数")

    frame_paths = collect_frame_paths(images_dir, start_frame, frame_count)

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"无法读取首帧图像 {frame_paths[0].name}")

    height, width = first_frame.shape[:2]
    writer = open_video_writer(video_path, fps, (width, height))

    if writer is None:
        create_video_with_ffmpeg(frame_paths, video_path, fps)
        print("\n生成完成 (ffmpeg):")
        print(f"  输出文件: {video_path}")
        print(f"  写入帧数: {len(frame_paths)}")
        print(f"  帧率: {fps} fps")
        print(f"  分辨率: {width}x{height}")
        return

    temp_path = str(video_path) + '.tmp'
    frames_written = 0

    try:
        for idx, frame_path in enumerate(frame_paths, start=1):
            img = cv2.imread(str(frame_path))
            if img is None:
                print(f"警告: 无法读取 {frame_path.name}，跳过。")
                continue

            if img.shape[:2] != (height, width):
                print(f"警告: 帧 {frame_path.name} 分辨率变化，跳过。")
                continue

            writer.write(img)
            frames_written += 1

            if frames_written % 50 == 0:
                print(f"已写入 {frames_written}/{len(frame_paths)} 帧...")
    finally:
        writer.release()

    if frames_written == 0:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print("警告: VideoWriter 未写入任何帧，改用 ffmpeg 生成。")
        create_video_with_ffmpeg(frame_paths, video_path, fps)
        print("\n生成完成 (ffmpeg):")
        print(f"  输出文件: {video_path}")
        print(f"  写入帧数: {len(frame_paths)}")
        print(f"  帧率: {fps} fps")
        print(f"  分辨率: {width}x{height}")
        return

    if os.path.exists(temp_path):
        os.rename(temp_path, str(video_path))

    print("\n生成完成:")
    print(f"  输出文件: {video_path}")
    print(f"  写入帧数: {frames_written}")
    print(f"  帧率: {fps} fps")
    print(f"  分辨率: {width}x{height}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从图像帧目录中截取部分时间段生成视频"
    )

    parser.add_argument(
        "images_dir",
        help="帧图像所在目录（包含 frame_000000.png 等文件）"
    )
    parser.add_argument(
        "--output",
        default="segment.mp4",
        help="输出视频路径"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="帧率（需与原始提取时一致）"
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=None,
        help="起始时间（秒）"
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=None,
        help="视频持续时间（秒），默认 600 秒 = 10 分钟"
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="起始帧编号（优先级高于 --start-sec）"
    )
    parser.add_argument(
        "--frame-count",
        type=int,
        default=None,
        help="输出帧数（优先级高于 --duration-sec）"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    images_dir = Path(args.images_dir).resolve()
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"帧目录不存在: {images_dir}")

    start_frame = args.start_frame
    if start_frame is None:
        if args.start_sec is None:
            args.start_sec = 0.0
        start_frame = int(args.start_sec * args.fps)

    if start_frame < 0:
        raise ValueError("起始帧不能为负数")

    frame_count = args.frame_count
    if frame_count is None:
        duration_sec = args.duration_sec if args.duration_sec is not None else 600.0
        if duration_sec <= 0:
            raise ValueError("duration-sec 必须大于 0")
        frame_count = int(duration_sec * args.fps)

    if frame_count <= 0:
        raise ValueError("frame-count 必须大于 0")

    output_path = Path(args.output).resolve()

    print("参数信息:")
    print(f"  帧目录: {images_dir}")
    print(f"  输出文件: {output_path}")
    print(f"  起始帧: {start_frame}")
    print(f"  输出帧数: {frame_count}")
    print(f"  帧率: {args.fps} fps")

    build_video_from_frames(
        images_dir=images_dir,
        video_path=output_path,
        start_frame=start_frame,
        frame_count=frame_count,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()

