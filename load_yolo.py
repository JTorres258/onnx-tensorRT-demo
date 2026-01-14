from ultralytics import YOLO
import torch

def main():
    print("Loading YOLOv8 model...")
    # Load a pretrained YOLOv8n model
    model = YOLO("yolov8n.pt")
    
    # Display model information
    print("Model loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Model names: {model.names}")
    
    # Check if CUDA is available and move model to GPU if so
    if torch.cuda.is_available():
        print("CUDA is available. Moving model to GPU...")
        model.to('cuda')
        print("Model moved to GPU.")
    else:
        print("CUDA not available. Running on CPU.")

if __name__ == "__main__":
    main()
