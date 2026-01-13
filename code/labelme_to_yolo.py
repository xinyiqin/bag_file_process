import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def parse_labelme_json(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def rectangle_points_to_bbox(points: List[List[float]]) -> Tuple[float, float, float, float]:
    (x1, y1), (x2, y2) = points
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    return x_min, y_min, x_max, y_max


def bbox_to_yolo_format(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Tuple[float, float, float, float]:
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0

    return (
        cx / width,
        cy / height,
        bw / width,
        bh / height,
    )


def convert_labelme_to_yolo(
    json_path: Path,
    image_root: Path,
    yolo_output_dir: Path,
    vis_output_dir: Path,
    label_to_id: Dict[str, int],
    label_colors: Dict[str, Tuple[int, int, int]],
    overwrite: bool = True,
):
    data = parse_labelme_json(json_path)
    image_path = data.get("imagePath")

    if not image_path:
        raise ValueError(f"JSON {json_path} 缺少 imagePath 字段")

    # 支持相对路径
    image_full_path = (json_path.parent / image_path).resolve()
    if not image_full_path.exists() and image_root:
        candidate = (image_root / image_path).resolve()
        if candidate.exists():
            image_full_path = candidate

    if not image_full_path.exists():
        raise FileNotFoundError(f"找不到图像文件: {image_full_path}")

    img = cv2.imread(str(image_full_path))
    if img is None:
        raise RuntimeError(f"无法读取图像: {image_full_path}")

    height, width = data.get("imageHeight"), data.get("imageWidth")
    if height is None or width is None:
        height, width = img.shape[:2]

    shapes = data.get("shapes", [])
    boxes = []

    for shape in shapes:
        if shape.get("shape_type") != "rectangle":
            print(f"跳过非矩形标注: {json_path} -> {shape.get('shape_type')}")
            continue

        label = shape.get("label")
        if not label:
            print(f"跳过缺少 label 的标注: {json_path}")
            continue
        if label not in label_to_id:
            print(f"⚠️ 未知标签 '{label}'，自动跳过。可检查数据或重新生成 class 映射。")
            continue

        points = shape.get("points")
        if not points or len(points) != 2:
            print(f"标注格式异常（非两点矩形）: {json_path}")
            continue

        x1, y1, x2, y2 = rectangle_points_to_bbox(points)
        boxes.append((label, x1, y1, x2, y2))

    yolo_output_dir.mkdir(parents=True, exist_ok=True)
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = yolo_output_dir / (json_path.stem + ".txt")
    if txt_path.exists() and not overwrite:
        print(f"跳过已有 txt: {txt_path}")
    else:
        with open(txt_path, "w", encoding="utf-8") as f:
            for (label, x1, y1, x2, y2) in boxes:
                x_center, y_center, norm_w, norm_h = bbox_to_yolo_format(x1, y1, x2, y2, width, height)
                f.write(f"{label_to_id[label]} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    vis_img = img.copy()
    for (label, x1, y1, x2, y2) in boxes:
        color = label_colors.get(label, (0, 255, 0))
        cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            vis_img,
            label,
            (int(x1), int(max(0, y1 - 5))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(vis_output_dir / (json_path.stem + ".png")), vis_img)


def parse_args():
    parser = argparse.ArgumentParser(description="Labelme JSON -> YOLO txt 并生成可视化（使用统一数据根目录）")
    parser.add_argument(
        "data_root",
        help="数据根目录或同名 .bag 文件路径（默认使用 <root>/labelme、<root>/images、<root>/labels_refined、<root>/visuals_refined）",
    )
    parser.add_argument("--labelme-dir", help="自定义 Labelme JSON 目录（默认: <root>/labelme）")
    parser.add_argument("--image-dir", help="自定义图像目录（默认: <root>/images）")
    parser.add_argument("--yolo-dir", help="自定义 YOLO txt 输出目录（默认: <root>/labels_refined）")
    parser.add_argument("--visual-dir", help="自定义可视化输出目录（默认: <root>/visuals_refined）")
    parser.add_argument("--keep-label", action="store_true", help="保留原有 YOLO txt，不覆盖")
    parser.add_argument("--classes-file", help="写出或更新 YOLO classes 名称文件（默认: <yolo-dir>/classes.txt）")
    return parser.parse_args()


def main():
    args = parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    if data_root.suffix == ".bag":
        candidate = data_root.with_suffix("")
        data_root = candidate if candidate.exists() else data_root.parent / data_root.stem
    if not data_root.exists():
        raise FileNotFoundError(f"数据根目录不存在: {data_root}")

    labelme_dir = Path(args.labelme_dir).expanduser().resolve() if args.labelme_dir else data_root / "labelme"
    image_dir = Path(args.image_dir).expanduser().resolve() if args.image_dir else data_root / "images"
    yolo_dir = Path(args.yolo_dir).expanduser().resolve() if args.yolo_dir else data_root / "labels_refined"
    visual_dir = Path(args.visual_dir).expanduser().resolve() if args.visual_dir else data_root / "visuals_refined"

    if not labelme_dir.exists():
        raise FileNotFoundError(f"Labelme 目录不存在: {labelme_dir}")
    if not image_dir.exists():
        raise FileNotFoundError(f"图像目录不存在: {image_dir}")

    json_files = sorted(labelme_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"未在 {labelme_dir} 找到任何 JSON 文件")

    print(f"数据根目录: {data_root}")
    print(f"Labelme JSON: {labelme_dir}")
    print(f"图像目录: {image_dir}")
    print(f"YOLO 输出目录: {yolo_dir}")
    print(f"可视化输出目录: {visual_dir}")

    # 收集全部标签，构建映射（按字典序确保稳定）
    label_set = set()
    for json_path in json_files:
        data = parse_labelme_json(json_path)
        for shape in data.get("shapes", []):
            if shape.get("shape_type") != "rectangle":
                continue
            label = shape.get("label")
            if label:
                label_set.add(label)

    if not label_set:
        raise RuntimeError(f"在 {labelme_dir} 中未找到任何矩形标注的标签")

    label_to_id = {label: idx for idx, label in enumerate(sorted(label_set))}

    classes_file = Path(args.classes_file).expanduser().resolve() if args.classes_file else (yolo_dir / "classes.txt")
    classes_file.parent.mkdir(parents=True, exist_ok=True)
    with open(classes_file, "w", encoding="utf-8") as cf:
        for label in sorted(label_to_id, key=lambda k: label_to_id[k]):
            cf.write(f"{label}\n")

    print("类别映射:")
    for label, idx in label_to_id.items():
        print(f"  {idx}: {label}")

    preferred_colors = {
        "fish_general": (255, 0, 0),  # BGR: 蓝色
        "fish_feeding": (0, 255, 255),  # BGR: 黄色
    }

    rng = np.random.default_rng(seed=42)
    label_colors = {}
    for label in label_to_id:
        if label in preferred_colors:
            label_colors[label] = preferred_colors[label]
            continue
        color = tuple(int(c) for c in rng.integers(0, 256, size=3))
        if sum(color) > 600:
            color = tuple(int(c * 0.7) for c in color)
        label_colors[label] = (int(color[2]), int(color[1]), int(color[0]))

    for json_path in json_files:
        convert_labelme_to_yolo(
            json_path=json_path,
            image_root=image_dir,
            yolo_output_dir=yolo_dir,
            vis_output_dir=visual_dir,
            label_to_id=label_to_id,
            label_colors=label_colors,
            overwrite=not args.keep_label,
        )

    print("✅ 转换完成！")
    print(f"YOLO txt 输出目录: {yolo_dir}")
    print(f"可视化输出目录: {visual_dir}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
将 Labelme 的矩形标注 JSON 转换为 YOLO txt，并输出可视化结果。
"""
