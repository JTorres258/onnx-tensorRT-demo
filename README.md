# YOLOv8 Nano ONNX and TensorRT Inference

This project demonstrates how to run YOLOv8 Nano (yolov8n) object detection on a webcam feed using different backends: PyTorch (standard), ONNX, and TensorRT for optimized performance on NVIDIA GPUs.

## 💻 System Requirements

This project was developed and tested on the following setup:
- **OS:** Windows 11 (Supports `os.add_dll_directory` for TensorRT)
- **Python:** 3.12.x
- **GPU:** NVIDIA GeForce RTX 2070 SUPER
- **NVIDIA Driver:** 581.57 (or newer)
- **CUDA:** 12.4 (or compatible higher version)
- **TensorRT:** 10.x (Installed via `pip`)

## 🚀 Setup

1. **Create and Activate a Virtual Environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
   *Note: This will install the CUDA 12.4 version of PyTorch.*

3. **Environment Configuration:**
   *Note: Our `onnx_to_tensorrt.py` and inference scripts automatically handle the TensorRT DLL paths, but make sure the `.venv\Lib\site-packages\tensorrt_libs` folder is accessible.*

## 📂 Project Structure

### 1. Model Preparation
- **`load_yolo.py`**: A simple script to verify the YOLOv8 model loads correctly and check for CUDA availability.
- **`export_onnx.py`**: Exports the standard `yolov8n.pt` model to `yolov8n.onnx` format.
- **`onnx_to_tensorrt.py`**: Converts the ONNX model into a TensorRT `.engine` file. This is customized for Windows systems where `trtexec` might be missing from the pip installation.
- **`benchmark.py`**: Compares the raw inference speed of PyTorch vs. TensorRT using dummy data (independently of any webcam limits).

### 2. Webcam Inference
- **`webcam_inference_torch.py`**: Runs real-time detection using the standard PyTorch backend.
- **`webcam_inference_trt.py`**: Runs optimized real-time detection using the TensorRT engine.

## 📊 Performance and FPS

Both inference scripts include an FPS counter displayed in the top-left corner. 

### Benchmark Results (YOLOv8 Nano on RTX 2070 SUPER)

Measured using `benchmark.py` over 100 iterations (640x640 input):

| Backend | Latency (ms) | Throughput (FPS) | Speedup |
| :--- | :--- | :--- | :--- |
| **PyTorch (CUDA)** | 11.51 ms | 86.85 FPS | 1.00x |
| **TensorRT FP32** | 6.07 ms | 164.65 FPS | 1.90x |
| **TensorRT FP16** | **4.81 ms** | **207.79 FPS** | **2.39x** |
| **TensorRT INT8** | 4.90 ms | 204.13 FPS | 2.35x |

> [!NOTE]
> **FP16** is the best performer on the RTX 2070 SUPER for YOLOv8n. INT8 did not show additional gains here, which often happens with very small models where the overhead of quantization kernels outweighs the compute savings.

*Result: TensorRT FP16 is approximately **2.4x faster** than standard PyTorch on this hardware.*

### Understanding Webcam FPS Limits
While the raw throughput shows ~155 FPS, most webcams are hardware-locked at 30 or 60 FPS due to hardware limits:
- **Webcam Bottleneck:** If both scripts show approximately 30.00 FPS, it means the hardware is the bottleneck, not the model.
- **TensorRT Optimization:** Even if the visible FPS is capped by the camera, TensorRT provides significant benefits:
    - **Lower Latency:** Faster "frame-to-detection" time.
    - **Reduced Resource Usage:** Lower GPU utilization and power consumption for the same workload.
- **Raw Speed:** To see the true potential of TensorRT without camera limits, you would need to run a benchmark script on a high-speed video file or a batch of images.

## 🛠 Troubleshooting

- **GPU not detected:** Run `python -c "import torch; print(torch.cuda.is_available())"`. If `False`, ensure you installed the CUDA-specific PyTorch build (`+cu124` or similar).
- **Missing TensorRT DLLs:** If `nvinfer.dll` errors occur, ensure you have reactivated your environment after the TensorRT installation, or verify that the `tensorrt_libs` folder is in your system PATH.