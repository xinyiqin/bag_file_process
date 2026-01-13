"""
对比EAW-Yolo11和原始YOLO11模型的性能指标
包括: Params, FLOPs, 推理显存, FPS
"""

import torch
import time
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_num_params, get_flops
import gc

def get_gpu_memory(device_id=0):
    """获取GPU显存使用情况 (MB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(device_id) / 1024**2
    return 0.0

def clear_gpu_cache():
    """清理GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def measure_inference_memory(model, imgsz=640, device="0", warmup=10, num_runs=50):
    """测量推理时的显存占用"""
    device_str = device if isinstance(device, str) else str(device)
    device_obj = torch.device(f"cuda:{device_str}" if torch.cuda.is_available() else "cpu")
    
    # 清除缓存
    clear_gpu_cache()
    
    # 创建输入
    dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device_obj)
    
    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # 同步并清除缓存
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    clear_gpu_cache()
    
    # 测量显存
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device_obj)
    
    with torch.no_grad():
        _ = model(dummy_input)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated(device_obj) / 1024**2  # MB
        return peak_memory
    
    return 0.0

def measure_fps(model, imgsz=640, device="0", warmup=30, num_runs=200):
    """测量FPS (Frames Per Second) - 直接测量模型forward pass时间"""
    device_str = device if isinstance(device, str) else str(device)
    device_obj = torch.device(f"cuda:{device_str}" if torch.cuda.is_available() else "cpu")
    
    # 创建输入 - 使用numpy array格式，模拟真实预测场景
    import numpy as np
    dummy_input_np = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    
    # 如果是YOLO包装器，使用predict方法（更接近真实使用场景）
    if hasattr(model, 'predict'):
        # Warmup
        for _ in range(warmup):
            _ = model.predict(dummy_input_np, verbose=False, imgsz=imgsz)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 测量推理时间
        times = []
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            _ = model.predict(dummy_input_np, verbose=False, imgsz=imgsz)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
    else:
        # 如果是底层PyTorch模型，直接forward
        dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device_obj)
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 测量推理时间
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.time()
                _ = model(dummy_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
    
    # 计算FPS (排除最快和最慢的10%)
    times = sorted(times)
    trim = int(len(times) * 0.1)
    if trim > 0:
        times = times[trim:-trim]
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time if avg_time > 0 else 0.0
    
    return fps

def compare_models():
    """对比两个模型的性能指标"""
    
    print("=" * 80)
    print("模型性能对比: EAW-Yolo11 vs 原始YOLO11")
    print("=" * 80)
    
    # 模型路径配置
    device = "0"
    imgsz = 640
    
    # 1. EAW-Yolo11模型 (使用训练后的权重或yaml配置)
    print("\n[1/2] 加载EAW-Yolo11模型...")
    eaw_yolo_path = "/mtc/qinxinyi/bag_file_process/EAW-Yolo11/runs/detect/fish_small_cbam/weights/best.pt"
    
    # 如果没有训练好的权重，使用yaml配置创建模型
    import os
    if os.path.exists(eaw_yolo_path):
        print(f"  使用训练权重: {eaw_yolo_path}")
        eaw_yolo = YOLO(eaw_yolo_path)
        eaw_model = eaw_yolo.model  # 获取底层PyTorch模型
    else:
        print(f"  权重文件不存在，使用yaml配置创建模型")
        eaw_yolo = YOLO("/mtc/qinxinyi/bag_file_process/EAW-Yolo11/ultralytics/cfg/models/11/eaw-yolo11.yaml")
        eaw_model = eaw_yolo.model  # 获取底层PyTorch模型
    
    eaw_model.to(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    eaw_model.eval()
    
    # 为了FPS测试，保留YOLO包装器（因为可能包含优化）
    eaw_model_for_fps = eaw_yolo
    
    # 2. 原始YOLO11模型
    print("\n[2/2] 加载原始YOLO11模型...")
    # 使用原始YOLO11的yaml配置，或尝试加载预训练权重
    try:
        original_yolo = YOLO("yolo11n.pt")  # 尝试加载预训练的nano版本
        print("  使用预训练权重: yolo11n.pt")
    except:
        try:
            original_yolo = YOLO("yolo11n.yaml")  # 使用yaml配置
            print("  使用yaml配置: yolo11n.yaml")
        except:
            # 使用本地yaml文件
            original_yaml = "/mtc/qinxinyi/bag_file_process/EAW-Yolo11/ultralytics/cfg/models/11/yolo11.yaml"
            original_yolo = YOLO(original_yaml)
            print(f"  使用本地yaml配置: {original_yaml}")
    
    original_model = original_yolo.model  # 获取底层PyTorch模型
    original_model.to(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    original_model.eval()
    
    # 为了FPS测试，保留YOLO包装器
    original_model_for_fps = original_yolo
    
    print("\n" + "=" * 80)
    print("开始计算性能指标...")
    print("=" * 80)
    
    results = {}
    
    # 计算EAW-Yolo11的指标
    print("\n计算EAW-Yolo11指标...")
    print("  - 参数量 (Params)...")
    eaw_params = get_num_params(eaw_model)
    
    print("  - FLOPs...")
    eaw_flops = get_flops(eaw_model, imgsz=imgsz)
    
    print("  - 推理显存...")
    eaw_memory = measure_inference_memory(eaw_model, imgsz=imgsz, device=device)
    
    print("  - FPS...")
    eaw_fps = measure_fps(eaw_model_for_fps, imgsz=imgsz, device=device)
    
    clear_gpu_cache()
    
    # 计算原始YOLO11的指标
    print("\n计算原始YOLO11指标...")
    print("  - 参数量 (Params)...")
    original_params = get_num_params(original_model)
    
    print("  - FLOPs...")
    original_flops = get_flops(original_model, imgsz=imgsz)
    
    print("  - 推理显存...")
    original_memory = measure_inference_memory(original_model, imgsz=imgsz, device=device)
    
    print("  - FPS...")
    original_fps = measure_fps(original_model_for_fps, imgsz=imgsz, device=device)
    
    # 计算差异百分比
    params_diff = ((eaw_params - original_params) / original_params) * 100
    flops_diff = ((eaw_flops - original_flops) / original_flops) * 100 if original_flops > 0 else 0
    memory_diff = ((eaw_memory - original_memory) / original_memory) * 100 if original_memory > 0 else 0
    fps_diff = ((eaw_fps - original_fps) / original_fps) * 100 if original_fps > 0 else 0
    
    # 打印对比结果
    print("\n" + "=" * 80)
    print("性能对比结果")
    print("=" * 80)
    
    print(f"\n{'指标':<20} {'EAW-Yolo11':<20} {'原始YOLO11':<20} {'差异':<20}")
    print("-" * 80)
    
    # 参数量
    print(f"{'Params':<20} {eaw_params:>18,} {original_params:>18,} {params_diff:>18.2f}%")
    
    # FLOPs
    print(f"{'FLOPs (G)':<20} {eaw_flops:>18.2f} {original_flops:>18.2f} {flops_diff:>18.2f}%")
    
    # 推理显存
    print(f"{'推理显存 (MB)':<20} {eaw_memory:>18.2f} {original_memory:>18.2f} {memory_diff:>18.2f}%")
    
    # FPS
    print(f"{'FPS':<20} {eaw_fps:>18.2f} {original_fps:>18.2f} {fps_diff:>18.2f}%")
    
    print("\n" + "=" * 80)
    
    # 保存结果到文件
    output_file = "/mtc/qinxinyi/bag_file_process/EAW-Yolo11/model_comparison_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("模型性能对比: EAW-Yolo11 vs 原始YOLO11\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'指标':<20} {'EAW-Yolo11':<20} {'原始YOLO11':<20} {'差异':<20}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Params':<20} {eaw_params:>18,} {original_params:>18,} {params_diff:>18.2f}%\n")
        f.write(f"{'FLOPs (G)':<20} {eaw_flops:>18.2f} {original_flops:>18.2f} {flops_diff:>18.2f}%\n")
        f.write(f"{'推理显存 (MB)':<20} {eaw_memory:>18.2f} {original_memory:>18.2f} {memory_diff:>18.2f}%\n")
        f.write(f"{'FPS':<20} {eaw_fps:>18.2f} {original_fps:>18.2f} {fps_diff:>18.2f}%\n")
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"\n结果已保存到: {output_file}")
    
    return {
        "eaw": {
            "params": eaw_params,
            "flops": eaw_flops,
            "memory": eaw_memory,
            "fps": eaw_fps
        },
        "original": {
            "params": original_params,
            "flops": original_flops,
            "memory": original_memory,
            "fps": original_fps
        },
        "diff": {
            "params": params_diff,
            "flops": flops_diff,
            "memory": memory_diff,
            "fps": fps_diff
        }
    }

if __name__ == "__main__":
    try:
        results = compare_models()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

