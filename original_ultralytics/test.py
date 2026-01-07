from ultralytics import YOLO

# 加载模型
model = YOLO(r"/mtc/gongruihao/qinxinyi/ultralytics/runs/detect/fish_small3/weights/best.pt")
model = YOLO(r"/mtc/gongruihao/qinxinyi/ultralytics/runs/detect/fish_nano5/weights/best.pt")


# 在 test 集上评估性能
metrics = model.val(
    data="/mtc/gongruihao/qinxinyi/dataset/fish_dataset_single_class/data.yaml",  # 数据配置文件
    split="test",               # 使用 test 集
    imgsz=640,                  # 图像尺寸
    batch=16,                   # 批大小
    save_json=True,             # 可选：生成 COCO 格式结果
    plots=True                  # 自动绘制混淆矩阵、PR曲线等
)

# 打印主要评估指标
print(metrics)
