import torch
import time
import numpy as np
import os
import sys
from ultralytics import YOLO

def benchmark(model_path, name, iterations=100, warmup=10):
    print(f"\nBenchmarking {name} ({model_path})...")
    
    # Load model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None

    # Create dummy input (standard YOLOv8 size 640x640)
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Warmup
    print(f"  Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        _ = model(img, verbose=False)

    # Benchmark loop
    print(f"  Running {iterations} iterations...")
    start_time = time.time()
    for _ in range(iterations):
        _ = model(img, verbose=False)
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = (total_time / iterations) * 1000  # in ms
    fps = iterations / total_time

    print(f"  -> Total time: {total_time:.4f} s")
    print(f"  -> Average latency: {avg_time:.2f} ms")
    print(f"  -> Raw Throughput: {fps:.2f} FPS")
    
    return {"name": name, "latency": avg_time, "fps": fps}

def main():
    # Helper to add tensorrt_libs to DLL path on Windows if needed
    if os.name == 'nt':
        try:
            venv_root = os.path.dirname(os.path.dirname(sys.executable))
            trt_libs = os.path.join(venv_root, 'Lib', 'site-packages', 'tensorrt_libs')
            if os.path.exists(trt_libs) and hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(trt_libs)
        except Exception:
            pass

    results = []

    # 1. PyTorch Baseline
    pt_model = "yolov8n.pt"
    if os.path.exists(pt_model):
        res = benchmark(pt_model, "PyTorch (CUDA)")
        if res: results.append(res)
    else:
        print(f"Warning: {pt_model} not found.")

    # 2. TensorRT Variants
    precisions = ['fp32', 'fp16', 'int8']
    for p in precisions:
        engine_path = f"yolov8n_{p}.engine"
        if os.path.exists(engine_path):
            res = benchmark(engine_path, f"TensorRT {p.upper()}")
            if res: results.append(res)
        else:
            # Fallback to the original yolov8n.engine as FP32 if it exists
            if p == 'fp32' and os.path.exists("yolov8n.engine"):
                 res = benchmark("yolov8n.engine", "TensorRT FP32 (Original)")
                 if res: results.append(res)
            else:
                 print(f"Warning: {engine_path} not found. Run 'onnx_to_tensorrt.py' first.")

    # Final Comparison Table
    if results:
        print("\n" + "="*50)
        print(f"{'BACKEND':<25} | {'LATENCY (ms)':<12} | {'FPS':<8}")
        print("-" * 50)
        for r in results:
            print(f"{r['name']:<25} | {r['latency']:<12.2f} | {r['fps']:<8.2f}")
        
        # Calculate max speedup
        baseline = results[0]['fps']
        best = max(results, key=lambda x: x['fps'])
        if best['fps'] > baseline:
            gain = best['fps'] / baseline
            print("-" * 50)
            print(f"Best performer: {best['name']} ({gain:.2f}x faster than baseline)")
        print("="*50)

if __name__ == "__main__":
    main()
