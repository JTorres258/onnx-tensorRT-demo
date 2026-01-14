import cv2
import time
from ultralytics import YOLO

def main():
    # Load the YOLOv8 model
    # It will use the yolov8n.pt file if present in the current directory, or download it.
    model = YOLO("yolov8n.pt")

    # Open the webcam (default camera index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam inference. Press 'q' to exit.")

    prev_time = 0

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
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # Display FPS on frame (in my case, in the range of 15-20 fps)
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Webcam Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
