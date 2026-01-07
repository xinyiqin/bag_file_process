from ultralytics import YOLO

model = YOLO(
    "/mtc/qinxinyi/bag_file_process/EAW-Yolo11/ultralytics/cfg/models/11/eaw-yolo11.yaml"
)

results = model.train(
    data="fish-detection.yaml",
    epochs=220,
    imgsz=640,
    batch=32,
    optimizer="SGD",
    lr0=0.01,
    momentum=0.937,
    weight_decay=5e-4,
    cos_lr=True,
    warmup_epochs=5,
    patience=35,
    augment=True,
    fliplr=0.5,
    flipud=0.0,
    mosaic=0.8,
    mixup=0.0,
    close_mosaic=20,
    pretrained=True,   # 会自动尝试加载对应尺度的预训练权重
    name="fish_small_cbam",
    device="0",
)
