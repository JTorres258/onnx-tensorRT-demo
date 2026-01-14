import tensorrt as trt
import os
import sys

def build_engine(onnx_file_path, engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    
    # Define the network with the explicit batch flag (required for ONNX)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    config = builder.create_builder_config()
    # Try to set workspace size to 1GB (1 << 30 bytes)
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    except AttributeError:
        # Fallback for older TRT versions if necessary
        pass

    if not os.path.exists(onnx_file_path):
        print(f"Error: ONNX file {onnx_file_path} not found.")
        return

    print(f"Parsing ONNX file: {onnx_file_path}")
    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    print("Building TensorRT engine. This may take a few minutes...")
    # Build the serialized network (engine)
    try:
        serialized_engine = builder.build_serialized_network(network, config)
    except AttributeError:
         # Fallback for very old TRT versions (unlikely given pip install)
        print("Error: build_serialized_network method not found.")
        return

    if serialized_engine:
        print(f"Saving engine to {engine_file_path}")
        with open(engine_file_path, "wb") as f:
            f.write(serialized_engine)
        print("Conversion complete!")
    else:
        print("Failed to build engine.")

if __name__ == "__main__":
    onnx_path = "yolov8n.onnx"
    engine_path = "yolov8n.engine"
    
    # Try to add tensorrt_libs to DLL path if on Windows, just in case
    # This helps if the user hasn't reactivated the environment
    if os.name == 'nt':
        try:
            # Assuming standard venv structure
            venv_root = os.path.dirname(os.path.dirname(sys.executable))
            trt_libs = os.path.join(venv_root, 'Lib', 'site-packages', 'tensorrt_libs')
            if os.path.exists(trt_libs):
                os.add_dll_directory(trt_libs)
        except Exception as e:
            # Continue normally if this fails, relying on PATH
            pass

    build_engine(onnx_path, engine_path)
