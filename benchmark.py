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
        return

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
    
    return avg_time, fps

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

    pt_model = "yolov8n.pt"
    engine_model = "yolov8n.engine"

    if not os.path.exists(pt_model):
        print(f"Error: {pt_model} not found.")
        return
    
    if not os.path.exists(engine_model):
        print(f"Error: {engine_model} not found. Please run 'onnx_to_tensorrt.py' first.")
        return

    # Run benchmarks
    torch_res = benchmark(pt_model, "PyTorch (CUDA)")
    trt_res = benchmark(engine_model, "TensorRT")

    if torch_res and trt_res:
        speedup = torch_res[1] / trt_res[1]
        print("\n" + "="*30)
        print("FINAL COMPARISON")
        print("="*30)
        print(f"PyTorch FPS: {torch_res[1]:.2f}")
        print(f"TensorRT FPS: {trt_res[1]:.2f}")
        
        if trt_res[1] > torch_res[1]:
            gain = (trt_res[1] / torch_res[1])
            print(f"\nTensorRT is {gain:.2f}x faster than PyTorch!")
        else:
            print("\nPyTorch performed better or equal in this test.")
        print("="*30)

if __name__ == "__main__":
    main()
