from ultralytics import YOLO
import torch

# 方案1: 使用预训练权重，然后修改类别数
# 这会加载backbone的预训练权重，但检测头会随机初始化
model = YOLO("yolo11n.pt")  # 加载预训练权重

# 修改模型配置
model.model.nc = 1  # 将类别数改为1
model.model.names = {0: "Fish"}  # 更新类别名称

# 重新初始化检测头的最后一层（类别预测层）
for m in model.model.modules():
    if isinstance(m, torch.nn.Conv2d) and hasattr(m, 'weight'):
        # 找到检测头的类别预测层并重新初始化
        if m.out_channels == 80:  # 原来的类别数
            # 如果最后一层的输出通道是80，说明是类别预测层
            # 需要找到并重新初始化
            pass

# 方案2: 直接训练，但使用更保守的参数
results = model.train(
    data="fish-detection.yaml",
    epochs=220,
    imgsz=640,
    batch=32,
    optimizer="SGD",
    lr0=0.001,  # 较小的学习率
    momentum=0.937,
    weight_decay=5e-4,
    cos_lr=True,
    warmup_epochs=10,
    patience=35,
    augment=True,
    fliplr=0.5,
    flipud=0.0,
    mosaic=0.8,
    mixup=0.0,
    close_mosaic=20,
    conf=0.001,  # 验证时使用更低的置信度阈值
    iou=0.6,
    name="fish_small_fefm_pretrained",
    device="0",
)

