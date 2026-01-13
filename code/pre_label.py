import argparse
import cv2
import numpy as np
import os
from pathlib import Path

# YOLO 类别（这里只有一个类别：fish）
CLASS_ID = 0
DEFAULT_LABEL = "fish_general"  # 自动预标注为普通状态

# ========== 图像处理参数（可微调） ==========
# 黑色鱼体的检测阈值
LOWER_BLACK = np.array([0, 0, 0])
UPPER_BLACK = np.array([180, 255, 60])  # HSV 中较低亮度

# 背景判定阈值：像素在多少比例的帧中被检测为“黑色”即认为是背景
BACKGROUND_RATIO = 0.7

# 面积阈值（避免背景小噪点），默认100，可根据分辨率调整
MIN_CONTOUR_AREA = 100

# 形态学核
MORPH_KERNEL = np.ones((3, 3), np.uint8)


def load_image_files(directory: Path):
    return sorted(
        [p for p in directory.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )


def build_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLACK, UPPER_BLACK)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    mask = cv2.dilate(mask, MORPH_KERNEL, iterations=1)
    return mask


def compute_background_mask(image_paths, background_dir: Path):
    background_accum = None
    count = len(image_paths)

    for idx, path in enumerate(image_paths, 1):
        img = cv2.imread(str(path))
        if img is None:
            continue
        mask = build_mask(img)
        if background_accum is None:
            background_accum = mask.astype(np.uint32)
        else:
            background_accum += mask.astype(np.uint32)

        if idx % 50 == 0 or idx == count:
            print(f"背景统计进度: {idx}/{count}")

    if background_accum is None:
        raise RuntimeError("未能读取任何有效图像，无法构建背景。")

    threshold = 255 * count * BACKGROUND_RATIO
    background_mask = np.where(background_accum >= threshold, 255, 0).astype(np.uint8)
    background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    background_mask = cv2.dilate(background_mask, MORPH_KERNEL, iterations=1)

    background_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(background_dir / "background_mask.png"), background_mask)
    return background_mask


def write_labelme_json(path: Path, img_shape, boxes, labelme_dir: Path):
    import json

    h, w = img_shape[:2]
    labelme_dir.mkdir(parents=True, exist_ok=True)
    json_path = labelme_dir / (path.stem + ".json")
    img_rel = os.path.relpath(str(path), str(labelme_dir))

    shapes = []
    for x, y, bw, bh in boxes:
        shapes.append({
            "label": DEFAULT_LABEL,
            "points": [[float(x), float(y)], [float(x + bw), float(y + bh)]],
            "group_id": 0,
            "shape_type": "rectangle",
            "flags": {}
        })

    data = {
        "version": "5.3.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": img_rel,
        "imageData": None,
        "imageHeight": int(h),
        "imageWidth": int(w),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_images(
    image_paths,
    background_mask,
    output_dir: Path,
    vis_dir: Path,
    background_dir: Path,
    labelme_dir: Path,
):
    h, w = None, None
    background_inv = cv2.bitwise_not(background_mask)

    for idx, path in enumerate(image_paths, 1):
        img = cv2.imread(str(path))
        if img is None:
            continue

        if h is None or w is None:
            h, w = img.shape[:2]

        mask = build_mask(img)
        foreground_mask = cv2.bitwise_and(mask, mask, mask=background_inv)

        contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        output_dir.mkdir(parents=True, exist_ok=True)
        label_path = output_dir / (path.stem + ".txt")
        boxes = []

        with open(label_path, "w") as f:
            for c in contours:
                x, y, bw, bh = cv2.boundingRect(c)
                if bw * bh < MIN_CONTOUR_AREA:
                    continue

                x_center = (x + bw / 2) / w
                y_center = (y + bh / 2) / h
                norm_w = bw / w
                norm_h = bh / h

                f.write(f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                boxes.append((x, y, bw, bh))

        vis_img = img.copy()
        for (x, y, bw, bh) in boxes:
            cv2.rectangle(vis_img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        vis_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(vis_dir / path.name), vis_img)
        background_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(background_dir / f"mask_{path.stem}.png"), foreground_mask)
        write_labelme_json(path, img.shape, boxes, labelme_dir)  # 生成 Labelme JSON

        if idx % 50 == 0 or idx == len(image_paths):
            print(f"处理进度: {idx}/{len(image_paths)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="基于数据根目录批量生成 YOLO 预标注、可视化和 Labelme JSON"
    )
    parser.add_argument(
        "data_root",
        help="数据根目录或同名 .bag 文件路径（脚本会自动在根目录下寻找 images/、labels/ 等子目录）",
    )
    parser.add_argument(
        "--images-dir",
        help="自定义图像目录（默认: <data_root>/images）",
    )
    parser.add_argument(
        "--labels-dir",
        help="自定义 YOLO 标签目录（默认: <data_root>/labels）",
    )
    parser.add_argument(
        "--visuals-dir",
        help="自定义可视化输出目录（默认: <data_root>/visuals）",
    )
    parser.add_argument(
        "--background-dir",
        help="自定义背景输出目录（默认: <data_root>/background）",
    )
    parser.add_argument(
        "--labelme-dir",
        help="自定义 Labelme JSON 输出目录（默认: <data_root>/labelme）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    if data_root.suffix == ".bag":
        candidate = data_root.with_suffix("")
        if candidate.exists():
            data_root = candidate
        else:
            data_root = data_root.parent / data_root.stem
    if not data_root.exists():
        raise FileNotFoundError(f"数据根目录不存在: {data_root}")

    input_dir = Path(args.images_dir).expanduser().resolve() if args.images_dir else data_root / "images"
    output_dir = Path(args.labels_dir).expanduser().resolve() if args.labels_dir else data_root / "labels"
    vis_dir = Path(args.visuals_dir).expanduser().resolve() if args.visuals_dir else data_root / "visuals"
    background_dir = Path(args.background_dir).expanduser().resolve() if args.background_dir else data_root / "background"
    labelme_dir = Path(args.labelme_dir).expanduser().resolve() if args.labelme_dir else data_root / "labelme"

    if not input_dir.exists():
        raise FileNotFoundError(f"图像目录不存在: {input_dir}")

    image_paths = load_image_files(input_dir)
    if not image_paths:
        raise RuntimeError(f"目录中未找到图像文件: {input_dir}")

    print(f"数据根目录: {data_root}")
    print(f"图像目录: {input_dir}")
    print(f"YOLO 标签输出: {output_dir}")
    print(f"可视化输出: {vis_dir}")
    print(f"背景输出: {background_dir}")
    print(f"Labelme 输出: {labelme_dir}")

    print("第一阶段：统计背景...")
    background_mask = compute_background_mask(image_paths, background_dir)

    print("第二阶段：生成检测与标注...")
    process_images(image_paths, background_mask, output_dir, vis_dir, background_dir, labelme_dir)

    print("✅ 标注完成！")
    print(f"标签保存在: {output_dir}")
    print(f"可视化结果保存在: {vis_dir}")
    print(f"背景调试输出保存在: {background_dir}")
    print(f"Labelme JSON 保存在: {labelme_dir}")


if __name__ == "__main__":
    main()
