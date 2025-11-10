#!/usr/bin/env python3
"""
从ROS bag文件中提取图像并转换为MP4视频
Extract images from ROS bag file and convert to MP4 video
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path

# 尝试导入rosbag库
try:
    import rosbag
    ROSBAG_AVAILABLE = True
except ImportError:
    ROSBAG_AVAILABLE = False
    print("警告: 未找到rosbag库，将尝试其他方法")

try:
    from cv_bridge import CvBridge
except ImportError:
    print("警告: 未找到cv_bridge，尝试使用替代方法")
    CvBridge = None


IMAGE_EXTENSION = ".png"


def get_frame_filename(index: int) -> str:
    return f"frame_{index:06d}{IMAGE_EXTENSION}"


def check_existing_frames(images_dir):
    """
    检查已存在的连续图像帧数量
    
    Returns:
        已存在的连续帧数量（从0开始连续的帧数）
    """
    if not images_dir.exists():
        return 0

    count = 0
    while True:
        image_path = images_dir / get_frame_filename(count)
        if image_path.exists():
            count += 1
        else:
            break

    return count


def extract_images_from_bag(bag_file, output_dir, image_topic=None, fps=30, ffmpeg_timeout=None):
    """
    从ROS bag文件中提取图像
    
    Args:
        bag_file: bag文件路径
        output_dir: 输出目录
        image_topic: 图像话题名称（如果为None，自动检测）
        fps: 输出视频帧率
        ffmpeg_timeout: ffmpeg 处理超时时间（秒），None 表示不限制
    """
    print(f"正在读取bag文件: {bag_file}")
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # 检查已存在的帧数（断点续传）
    existing_frame_count = check_existing_frames(images_dir)
    if existing_frame_count > 0:
        print(f"🔄 检测到已有 {existing_frame_count} 帧，将从断点继续处理...")
    
    # 初始化cv_bridge
    if CvBridge:
        bridge = CvBridge()
    else:
        bridge = None
    
    # 打开bag文件
    if not ROSBAG_AVAILABLE:
        print("\n错误: 需要安装rosbag库")
        print("\n请选择以下方法之一:")
        print("1. 安装ROS 1 (推荐):")
        print("   Ubuntu/Debian: sudo apt-get install ros-<distro>-rosbag ros-<distro>-cv-bridge")
        print("2. 使用pip安装 (需要先安装ROS):")
        print("   pip install rospy-msgs")
        print("3. 尝试使用rosbag命令行工具:")
        print("   rosbag play <bag_file> 配合其他工具")
        sys.exit(1)
    
    try:
        bag = rosbag.Bag(bag_file, 'r')
    except Exception as e:
        print(f"错误: 无法打开bag文件: {e}")
        print("\n可能需要安装ROS环境或rosbag库")
        sys.exit(1)
    
    # 获取所有话题信息
    info = bag.get_type_and_topic_info()
    topics = info.topics
    
    print("\n找到的话题:")
    for topic_name, topic_info in topics.items():
        print(f"  {topic_name}: {topic_info.msg_type} ({topic_info.message_count} 条消息)")
    
    # 自动检测图像话题
    if image_topic is None:
        image_topics = []
        for topic_name, topic_info in topics.items():
            msg_type = topic_info.msg_type
            if 'Image' in msg_type or 'CompressedImage' in msg_type:
                image_topics.append(topic_name)
        
        if len(image_topics) == 0:
            print("\n错误: 未找到图像话题")
            bag.close()
            sys.exit(1)
        elif len(image_topics) == 1:
            image_topic = image_topics[0]
            print(f"\n自动检测到图像话题: {image_topic}")
        else:
            print(f"\n找到多个图像话题:")
            for i, topic in enumerate(image_topics):
                print(f"  {i+1}. {topic}")
            print("\n请使用 --topic 参数指定要使用的话题")
            bag.close()
            sys.exit(1)
    
    # 检查话题是否存在
    if image_topic not in topics:
        print(f"\n错误: 话题 '{image_topic}' 不存在")
        bag.close()
        sys.exit(1)
    
    # 提取图像
    print(f"\n正在从话题 '{image_topic}' 提取图像...")
    frame_count = existing_frame_count

    if existing_frame_count > 0:
        print(f"已存在 {existing_frame_count} 帧，将跳过对应消息并继续追加新帧...")
    
    # 跳过已处理的消息数量（使用已存在的帧数）
    skip_count = existing_frame_count
    
    try:
        message_iterator = bag.read_messages(topics=[image_topic])
        for topic, msg, t in message_iterator:
            # 如果还有需要跳过的消息，跳过
            if skip_count > 0:
                skip_count -= 1
                continue
            
            try:
                # 处理图像消息
                if hasattr(msg, 'data'):  # sensor_msgs/Image
                    if bridge:
                        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
                    else:
                        # 尝试直接转换
                        # sensor_msgs/Image格式: data字段是uint8数组
                        # 需要根据encoding和width/height重建图像
                        encoding = msg.encoding
                        width = msg.width
                        height = msg.height
                        data = np.frombuffer(msg.data, dtype=np.uint8)
                        
                        if encoding == 'rgb8':
                            cv_image = data.reshape((height, width, 3))
                            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                        elif encoding == 'bgr8':
                            cv_image = data.reshape((height, width, 3))
                        elif encoding == 'mono8':
                            cv_image = data.reshape((height, width))
                        else:
                            print(f"警告: 不支持的编码格式 {encoding}，跳过")
                            continue
                
                elif hasattr(msg, 'format'):  # sensor_msgs/CompressedImage
                    # 压缩图像
                    data = np.frombuffer(msg.data, np.uint8)
                    cv_image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    if cv_image is None:
                        print("警告: 无法解码压缩图像，跳过")
                        continue
                
                else:
                    print(f"警告: 未知的消息类型，跳过")
                    continue
                
                # 保存图像
                image_filename = images_dir / get_frame_filename(frame_count)
                cv2.imwrite(str(image_filename),cv_image,[cv2.IMWRITE_PNG_COMPRESSION, 0])
                frame_count += 1

                if (frame_count - existing_frame_count) % 10 == 0:
                    print(f"已提取 {frame_count} 帧（新增 {frame_count - existing_frame_count} 帧）...")
                    
            except Exception as e:
                print(f"警告: 处理消息时出错: {e}")
                continue
    
    except Exception as e:
        print(f"错误: 读取bag文件时出错: {e}")
        bag.close()
        sys.exit(1)
    
    finally:
        bag.close()
    
    # 显示提取结果
    new_frames = frame_count - existing_frame_count
    if new_frames <= 0:
        print(f"\n共 {frame_count} 帧图像（全部来自已存在的文件）")
    else:
        print(f"\n共提取 {frame_count} 帧图像（新提取: {new_frames} 帧）")
    
    if frame_count == 0:
        print("错误: 未提取到任何图像")
        sys.exit(1)
    
    # 创建视频
    video_path = output_dir / "output.mp4"
    
    # 检查视频是否已存在
    if video_path.exists() and video_path.stat().st_size > 0:
        print(f"\n✓ 视频已存在: {video_path}")
        print(f"  总帧数: {frame_count}")
        return video_path
    
    print(f"\n正在创建MP4视频...")

    # 获取第一帧的尺寸
    first_frame_path = None
    for i in range(frame_count):
        candidate = images_dir / get_frame_filename(i)
        if candidate.exists():
            first_frame_path = candidate
            break

    if first_frame_path is None:
        print("错误: 未找到任何图像帧文件，无法创建视频")
        sys.exit(1)

    first_frame = cv2.imread(str(first_frame_path))
    if first_frame is None:
        print(f"错误: 无法读取首帧图像 {first_frame_path}")
        sys.exit(1)

    height, width = first_frame.shape[:2]
    
    # 尝试使用更兼容的编码器
    # 优先尝试 H.264 (avc1), 如果不可用则使用 mp4v
    fourcc_options = [
        ('avc1', 'H.264'),
        ('h264', 'H.264'),
        ('X264', 'X.264'),
        ('mp4v', 'MPEG-4'),
    ]
    
    out = None
    used_codec = None
    temp_path = str(video_path) + '.tmp'

    for codec, name in fourcc_options:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

            if out.isOpened():
                print(f"使用编码器: {name} ({codec})")
                used_codec = name
                break
            else:
                out.release()
                out = None
        except Exception as e:
            if out:
                out.release()
                out = None
            continue

    if out is None or not out.isOpened():
        print("警告: 无法使用视频编码器，尝试使用ffmpeg")
        # 使用ffmpeg作为备选方案
        return create_video_with_ffmpeg(frame_count, video_path, fps, output_dir, timeout=ffmpeg_timeout)

    # 写入所有帧
    frames_written = 0
    for i in range(frame_count):
        frame_path = images_dir / get_frame_filename(i)
        if not frame_path.exists():
            print(f"警告: 缺少帧文件 {frame_path}，跳过")
            continue

        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"警告: 无法读取帧文件 {frame_path}，跳过")
            continue

        out.write(img)
        frames_written += 1

        if frames_written % 50 == 0:
            print(f"已写入 {frames_written}/{frame_count} 帧...")

    out.release()

    if frames_written == 0:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print("警告: 未写入任何帧，尝试使用ffmpeg重新创建视频")
        return create_video_with_ffmpeg(frame_count, video_path, fps, output_dir, timeout=ffmpeg_timeout)
    
    # 如果使用了临时文件，重命名
    if os.path.exists(temp_path):
        os.rename(temp_path, str(video_path))
    
    print(f"\n完成! 视频已保存到: {video_path}")
    print(f"  总帧数: {frame_count}")
    print(f"  实际写入: {frames_written} 帧")
    print(f"  帧率: {fps} fps")
    print(f"  分辨率: {width}x{height}")
    print(f"  编码器: {used_codec}")
    
    return video_path


def create_video_with_ffmpeg(frame_count, video_path, fps, output_dir, timeout=None):
    """使用ffmpeg创建视频（如果OpenCV编码器不可用）"""
    import subprocess
    
    print("\n使用ffmpeg创建视频...")

    if frame_count <= 0:
        print("错误: 没有可用的帧来创建视频。")
        return None
    
    if timeout is None:
        print("提示: ffmpeg不会设置超时时间，将持续等待直到完成。")
    else:
        print(f"提示: ffmpeg超时时间设置为 {timeout} 秒。")
    
    # 检查ffmpeg是否可用
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("错误: 未找到ffmpeg，请安装ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Linux: sudo apt-get install ffmpeg")
        return None
    
    # 创建临时文件列表
    images_dir = output_dir / "images"
    list_file = output_dir / "image_list.txt"
    
    # 确保images目录存在
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # 在Docker容器内使用相对路径（相对于workspace）
    # 获取相对于output_dir的路径
    list_file_path = str(list_file)
    original_cwd = os.getcwd()
    
    try:
        valid_frames = 0
        last_index = -1

        with open(list_file, 'w', encoding='utf-8') as f:
            for i in range(frame_count):
                image_path = images_dir / get_frame_filename(i)
                if not image_path.exists():
                    print(f"警告: 缺少帧文件 {image_path}，跳过")
                    continue

                # 使用相对路径（相对于列表文件）
                image_relative = f"images/{get_frame_filename(i)}"
                f.write(f"file '{image_relative}'\n")
                f.write(f"duration {1.0/fps}\n")
                valid_frames += 1
                last_index = i

            if valid_frames > 0:
                # 最后一帧需要重复一次
                last_image_relative = f"images/{get_frame_filename(last_index)}"
                f.write(f"file '{last_image_relative}'\n")

        if valid_frames == 0:
            print("错误: 没有有效的帧文件供ffmpeg使用。")
            if os.path.exists(list_file):
                os.remove(list_file)
            return None
        
        # 验证列表文件内容
        if not os.path.exists(list_file):
            print(f"错误: 无法创建列表文件: {list_file}")
            return None
        
        # 使用ffmpeg创建视频
        # 方法1: 尝试使用concat格式
        # 切换到output_dir目录运行ffmpeg，这样相对路径才能正确工作
        try:
            os.chdir(str(output_dir))
            
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', 'image_list.txt',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                'output.mp4'
            ]
            
            run_kwargs = {
                'check': True,
                'capture_output': True,
                'text': True,
            }
            if timeout is not None:
                run_kwargs['timeout'] = timeout

            subprocess.run(cmd, **run_kwargs)
            
            # 恢复原始工作目录
            os.chdir(original_cwd)
            
            # 清理临时文件
            if os.path.exists(list_file):
                os.remove(list_file)
            
            print(f"\n✓ 使用ffmpeg成功创建视频: {video_path}")
            return video_path
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            # 恢复原始工作目录
            os.chdir(original_cwd)
            # 如果concat方法失败，尝试使用图像序列方法
            print(f"警告: concat方法失败，尝试使用图像序列方法...")
            if isinstance(e, subprocess.TimeoutExpired):
                print("原因: ffmpeg处理超时")
            else:
                print(f"ffmpeg错误: {e.stderr if hasattr(e, 'stderr') else '未知错误'}")
            # 清理可能未完成的输出文件
            partial_output = output_dir / "output.mp4"
            if partial_output.exists():
                partial_output.unlink()
            
            # 方法2: 使用图像序列（更可靠）
            try:
                # 使用glob模式读取图像
                pattern = f'images/frame_%06d{IMAGE_EXTENSION}'
                cmd = [
                    'ffmpeg', '-y',
                    '-framerate', str(fps),
                    '-i', pattern,
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-r', str(fps),
                    '-movflags', '+faststart',
                    'output.mp4'
                ]
                
                run_kwargs = {
                    'check': True,
                    'capture_output': True,
                    'text': True,
                    'cwd': str(output_dir),
                }
                if timeout is not None:
                    run_kwargs['timeout'] = timeout

                subprocess.run(cmd, **run_kwargs)
                
                # 清理临时文件
                if os.path.exists(list_file):
                    os.remove(list_file)
                
                print(f"\n✓ 使用ffmpeg（图像序列）成功创建视频: {video_path}")
                return video_path
                
            except subprocess.CalledProcessError as e2:
                print(f"错误: ffmpeg处理失败")
                print(f"详细错误: {e2.stderr if hasattr(e2, 'stderr') else str(e2)}")
                
                # 显示调试信息
                print(f"\n调试信息:")
                print(f"  列表文件: {list_file_path}")
                print(f"  列表文件存在: {os.path.exists(list_file)}")
                if os.path.exists(list_file):
                    print(f"  列表文件内容（前5行）:")
                    with open(list_file, 'r') as f:
                        for i, line in enumerate(f):
                            if i < 5:
                                print(f"    {line.strip()}")
                
                # 清理临时文件
                if os.path.exists(list_file):
                    os.remove(list_file)
                return None
            except subprocess.TimeoutExpired:
                print("错误: ffmpeg处理超时")
                if os.path.exists(list_file):
                    os.remove(list_file)
                # 移除可能未完成的输出文件
                partial_output = output_dir / "output.mp4"
                if partial_output.exists():
                    partial_output.unlink()
                return None
                
    except Exception as e:
        # 确保恢复工作目录
        try:
            os.chdir(original_cwd)
        except:
            pass
        print(f"错误: 创建视频时出错: {e}")
        if os.path.exists(list_file):
            os.remove(list_file)
        return None


def main():
    parser = argparse.ArgumentParser(
        description='从ROS bag文件中提取图像并转换为MP4视频',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  单文件模式:
    python bag_to_video.py 20251103test.bag
  批处理模式:
    python bag_to_video.py --batch
    python bag_to_video.py --batch /path/to/bag_files
        """
    )
    
    parser.add_argument('bag_file', nargs='?', help='ROS bag文件路径')
    parser.add_argument('-o', '--output', default='output', help='输出目录 (默认: output)')
    parser.add_argument('-t', '--topic', default=None, help='图像话题名称 (默认: 自动检测)')
    parser.add_argument('-f', '--fps', type=int, default=30, help='输出视频帧率 (默认: 30)')
    parser.add_argument('--batch', nargs='?', const='.', help='批处理模式: 处理目录下所有 .bag 文件 (默认当前目录)')
    parser.add_argument('--ffmpeg-timeout', type=int, default=None, help='ffmpeg处理超时时间（秒），默认无限制')

    args = parser.parse_args()

    # --- 批处理模式 ---
    if args.batch:
        bag_dir = Path(args.batch).resolve()
        if not bag_dir.exists() or not bag_dir.is_dir():
            print(f"错误: 目录不存在: {bag_dir}")
            sys.exit(1)
        
        print(f"\n📂 批处理模式启动: {bag_dir}")
        bag_files = sorted(bag_dir.glob("*.bag"))
        
        if not bag_files:
            print("⚠️ 未找到任何 .bag 文件。")
            sys.exit(0)
        
        print(f"找到 {len(bag_files)} 个 bag 文件。")
        
        # 创建总输出文件夹
        output_root = Path(args.output).resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        skipped_count = 0
        processed_count = 0
        failed_count = 0

        for bag_file in bag_files:
            try:
                print("\n--------------------------------------------")
                print(f"🎞️ 检查: {bag_file.name}")
                sub_output = output_root / bag_file.stem
                video_path = sub_output / "output.mp4"
                
                # 检查是否已经处理完成
                if video_path.exists() and video_path.stat().st_size > 0:
                    print(f"⏭️  跳过: {bag_file.name} (已存在 output.mp4)")
                    skipped_count += 1
                    continue
                
                print(f"🎞️ 正在处理: {bag_file.name}")
                extract_images_from_bag(str(bag_file), str(sub_output), args.topic, args.fps, ffmpeg_timeout=args.ffmpeg_timeout)
                print(f"✅ 完成: {bag_file.name}")
                processed_count += 1
            except Exception as e:
                print(f"❌ 处理 {bag_file.name} 失败: {e}")
                failed_count += 1
                continue
        
        print("\n" + "="*50)
        print("📊 批处理统计:")
        print(f"  总文件数: {len(bag_files)}")
        print(f"  已处理: {processed_count}")
        print(f"  已跳过: {skipped_count}")
        print(f"  失败: {failed_count}")
        print("="*50)
        print(f"\n✅ 所有文件处理完成！")
        print(f"输出路径: {output_root}")
        sys.exit(0)

    # --- 单文件模式 ---
    if not args.bag_file:
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(args.bag_file):
        print(f"错误: 文件不存在: {args.bag_file}")
        sys.exit(1)

    extract_images_from_bag(args.bag_file, args.output, args.topic, args.fps, ffmpeg_timeout=args.ffmpeg_timeout)


if __name__ == '__main__':
    main()

