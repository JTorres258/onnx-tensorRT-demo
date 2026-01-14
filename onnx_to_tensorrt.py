import tensorrt as trt
import os
import sys
import torch

class RandomCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Random data calibrator for INT8 quantization.
    Note: For production accuracy, use a real dataset.
    """
    def __init__(self, input_shape, cache_file):
        super().__init__()
        self.input_shape = input_shape
        self.cache_file = cache_file
        # Allocate GPU memory for calibration data using torch
        self.device_input = torch.empty(input_shape, device='cuda', dtype=torch.float32)
        self.batch_count = 0
        self.max_batches = 10

    def get_batch_size(self):
        return self.input_shape[0]

    def get_batch(self, names):
        if self.batch_count < self.max_batches:
            self.device_input.uniform_(0, 1) # Fill with random data
            self.batch_count += 1
            return [self.device_input.data_ptr()]
        else:
            return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"Reading calibration cache from {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        print(f"Writing calibration cache to {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def build_engine(onnx_file_path, engine_file_path, precision='fp32'):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    config = builder.create_builder_config()
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB
    except AttributeError:
        pass

    # Precision settings
    if precision == 'fp16':
        if not builder.platform_has_fast_fp16:
            print("Warning: This device does not support fast FP16. Performance may be degraded.")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == 'int8':
        if not builder.platform_has_fast_int8:
            print("Warning: This device does not support fast INT8. Performance may be degraded.")
        config.set_flag(trt.BuilderFlag.INT8)
        # Assuming YOLOv8 input shape (1, 3, 640, 640)
        config.int8_calibrator = RandomCalibrator((1, 3, 640, 640), "calib_cache.bin")

    if not os.path.exists(onnx_file_path):
        print(f"Error: ONNX file {onnx_file_path} not found.")
        return

    print(f"Parsing ONNX file: {onnx_file_path}")
    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            print(f"ERROR: Failed to parse the ONNX file for {precision}.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    print(f"Building TensorRT {precision.upper()} engine. This may take a few minutes...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine:
        print(f"Saving engine to {engine_file_path}")
        with open(engine_file_path, "wb") as f:
            f.write(serialized_engine)
        print(f"Conversion to {precision.upper()} complete!")
    else:
        print(f"Failed to build {precision.upper()} engine.")

if __name__ == "__main__":
    # Add DLL path for Windows
    if os.name == 'nt':
        try:
            venv_root = os.path.dirname(os.path.dirname(sys.executable))
            trt_libs = os.path.join(venv_root, 'Lib', 'site-packages', 'tensorrt_libs')
            if os.path.exists(trt_libs) and hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(trt_libs)
        except Exception:
            pass

    onnx_path = "yolov8n.onnx"
    
    precisions = ['fp32', 'fp16', 'int8']
    
    for p in precisions:
        engine_path = f"yolov8n_{p}.engine"
        if os.path.exists(engine_path):
             print(f"Engine {engine_path} already exists. Skipping...")
             continue
        build_engine(onnx_path, engine_path, precision=p)
