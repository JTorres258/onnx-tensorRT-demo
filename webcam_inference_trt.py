import cv2
import time
import os
import sys
from ultralytics import YOLO

def main():
    # Helper to add tensorrt_libs to DLL path on Windows if needed
    if os.name == 'nt':
        try:
            # Assuming standard venv structure: .venv/Scripts/python.exe -> .venv/Lib/site-packages/tensorrt_libs
            venv_root = os.path.dirname(os.path.dirname(sys.executable))
            trt_libs = os.path.join(venv_root, 'Lib', 'site-packages', 'tensorrt_libs')
            if os.path.exists(trt_libs) and hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(trt_libs)
        except Exception:
            pass

    # Load the YOLOv8 TensorRT model
    print("Loading YOLOv8 TensorRT engine...")
    try:
        model = YOLO("yolov8n.engine")
    except Exception as e:
        print(f"Error loading engine: {e}")
        print("Ensure 'yolov8n.engine' exists and TensorRT is correctly installed.")
        return

    # Open the webcam (default camera index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam inference (TensorRT). Press 'q' to exit.")

    prev_time = time.time()
    count_frame = 0
    fps = 0.0

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Run inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Calculate FPS
        diff_time = time.time() - prev_time
        if diff_time >= 1:
            fps = count_frame / diff_time
            prev_time = time.time()
            count_frame = 0
        else:
            count_frame += 1

        # Display FPS on frame
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Webcam Inference (TensorRT)", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
