from ultralytics import YOLO

def main():
    print("Loading YOLOv8 model for export...")
    # Load a pretrained YOLOv8n model
    model = YOLO("yolov8n.pt")
    
    # Export the model to ONNX format
    print("Exporting model to ONNX...")
    model.export(format="onnx")
    print("Export complete! 'yolov8n.onnx' should be in the current directory.")

if __name__ == "__main__":
    main()
