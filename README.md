# YOLOv8 ONNX and TensorRT Inference

This project demonstrates how to run YOLOv8 object detection on a webcam feed using different backends: PyTorch (standard), ONNX, and TensorRT for optimized performance on NVIDIA GPUs.

## 🚀 Setup

1. **Activate the Virtual Environment:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   *Note: Activation adds TensorRT libraries to your PATH automatically.*

2. **Install Dependencies:**
   The project requires `ultralytics`, `opencv-python`, `onnx`, `onnxruntime`, and `tensorrt`. Make sure PyTorch is installed with CUDA support for best performance.

## 📂 Project Structure

### 1. Model Preparation
- **`load_yolo.py`**: A simple script to verify the YOLOv8 model loads correctly and check for CUDA availability.
- **`export_onnx.py`**: Exports the standard `yolov8n.pt` model to `yolov8n.onnx` format.
- **`onnx_to_tensorrt.py`**: Converts the ONNX model into a TensorRT `.engine` file. This is customized for Windows systems where `trtexec` might be missing from the pip installation.

### 2. Webcam Inference
- **`webcam_inference_torch.py`**: Runs real-time detection using the standard PyTorch backend.
- **`webcam_inference_trt.py`**: Runs optimized real-time detection using the TensorRT engine.

## 📊 Performance and FPS

Both inference scripts include an FPS counter displayed in the top-left corner. 

### Understanding FPS Results:
- **Webcam Hardware Limit:** Most webcams are hardware-locked at 30 or 60 FPS. If both scripts show a steady 30.00 FPS, it means the hardware is the bottleneck, not the model.
- **TensorRT Optimization:** Even if the visible FPS is capped by the camera, TensorRT provides significant benefits:
    - **Lower Latency:** Faster "frame-to-detection" time.
    - **Reduced Resource Usage:** Lower GPU utilization and power consumption for the same workload.
- **Raw Speed:** To see the true potential of TensorRT without camera limits, you would need to run a benchmark script on a high-speed video file or a batch of images.

## 🛠 Troubleshooting

- **GPU not detected:** Run `python -c "import torch; print(torch.cuda.is_available())"`. If `False`, ensure you installed the CUDA-specific PyTorch build (`+cu124` or similar).
- **Missing TensorRT DLLs:** If `nvinfer.dll` errors occur, ensure you have reactivated your environment after the TensorRT installation, or verify that the `tensorrt_libs` folder is in your system PATH.