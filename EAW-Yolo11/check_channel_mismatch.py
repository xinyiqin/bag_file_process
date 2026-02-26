"""检查结合版本中FEFM的通道数是否匹配"""
import sys
sys.path.insert(0, '/mtc/qinxinyi/bag_file_process/EAW-Yolo11')

from ultralytics import YOLO

print("=" * 60)
print("检查结合版本中FEFM的通道数匹配情况")
print("=" * 60)

# 加载模型
model = YOLO(
    "/mtc/qinxinyi/bag_file_process/EAW-Yolo11/ultralytics/cfg/models/11/(A+B)eaw-yolo11-fefm.yaml"
)

# 打印模型结构，找到FEFM模块
print("\n查找FEFM模块及其输入输出通道数:")
print("-" * 60)

fefm_modules = []
for i, module in enumerate(model.model.model):
    if hasattr(module, 'type') and 'FEFM' in str(module.type):
        fefm_modules.append((i, module))
        print(f"\n模块 {i}: {module.type}")
        print(f"  输入来源: {module.f}")
        if hasattr(module, 'args'):
            print(f"  FEFM参数: {module.args}")

# 检查通道流
print("\n" + "=" * 60)
print("检查通道流:")
print("-" * 60)

# 模拟通道流（基于YAML配置）
# Backbone输出:
# layer 6: 512 (P4)
# layer 4: 256 (P3)  
# layer 10: 1024 (P5 after C2AIFI)

# Head:
# layer 11: upsample -> 1024
# layer 12: FEFM([11, 6], [256, 128, 384]) -> 384
# layer 13: EC3k2(384 -> 512) -> 512
# layer 14: upsample -> 512
# layer 15: FEFM([14, 4], [128, 128, 256]) -> 256
# layer 16: C3k2(256 -> 256) -> 256
# layer 17: Conv(256 -> 256) -> 256
# layer 18: FEFM([17, 13], [64, 128, 192]) -> 192
# layer 19: C3k2(192 -> 512) -> 512
# layer 20: Conv(512 -> 512) -> 512
# layer 21: FEFM([20, 10], [128, 256, 384]) -> 384
# layer 22: C3k2(384 -> 1024) -> 1024

print("\n问题分析:")
print("-" * 60)
print("layer 12 FEFM: 输入应该是 [1024, 512]，但参数是 [256, 128, 384]")
print("  - 第一个输入(upsample后): 1024通道，但FEFM期望256通道")
print("  - 第二个输入(P4): 512通道，但FEFM期望128通道")
print("  - 需要先降维！")

print("\nlayer 15 FEFM: 输入应该是 [512, 256]，但参数是 [128, 128, 256]")
print("  - 第一个输入(upsample后): 512通道，但FEFM期望128通道")
print("  - 第二个输入(P3): 256通道，但FEFM期望128通道")
print("  - 需要先降维！")

print("\n建议修复:")
print("-" * 60)
print("在FEFM之前添加Conv层来调整通道数，或者修改FEFM参数")



