from ultralytics import YOLO

model = YOLO(
    "/mtc/qinxinyi/bag_file_process/EAW-Yolo11/ultralytics/cfg/models/11/(A+B)eaw-yolo11-fefm.yaml"
)

results = model.train(
    data="fish-detection.yaml",
    epochs=220,
    imgsz=640,
    batch=32,
    optimizer="SGD",
    lr0=0.001,  # 降低学习率：从头训练需要更小的学习率
    momentum=0.937,
    weight_decay=5e-4,
    cos_lr=True,
    warmup_epochs=10,  # 增加warmup epochs
    patience=35,
    augment=True,
    fliplr=0.5,
    flipud=0.0,
    mosaic=0.8,
    mixup=0.0,
    close_mosaic=20,
    pretrained=False,   # 不使用预训练权重（因为类别数从80改为1，权重不兼容）
    conf=0.001,  # 验证时使用更低的置信度阈值，避免过滤掉所有预测
    iou=0.6,  # 稍微降低IoU阈值
    name="fish_small_fefm",
    device="0",
)
