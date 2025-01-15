import cv2
import time
import torch
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('ppe.pt')  #  'ppe.pt' model path

# Determine device (e.g., CUDA or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

# Open video file or capture device
video_source = 0  # Set 0 for webcam, or replace with video file path
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("Error: Unable to open video source.")
    exit()

# Initialize variables
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Finished processing video.")
        break

    # Process frame with YOLO model
    results = model(frame)

    # Display results on frame
    annotated_frame = results[0].plot()

    # Calculate FPS
    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    # Overlay FPS and device information
    text = f"FPS: {fps:.2f} | Device: {device_name}"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_width, text_height = text_size

    # Draw black rectangle as background
    top_left_corner = (frame.shape[1] - text_width - 10, 10)
    bottom_right_corner = (frame.shape[1] - 10, 10 + text_height + 10)
    cv2.rectangle(annotated_frame, top_left_corner, bottom_right_corner, (0, 0, 0), -1)

    # Put text on the rectangle
    text_origin = (frame.shape[1] - text_width - 5, 30)
    cv2.putText(
        annotated_frame,
        text,
        text_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Show the frame
    cv2.imshow("YOLO Output", annotated_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Calculate final FPS
end_time = time.time()
total_elapsed_time = end_time - start_time
final_fps = frame_count / total_elapsed_time if total_elapsed_time > 0 else 0

print(f"Processed {frame_count} frames in {total_elapsed_time:.2f} seconds.")
print(f"Final Average FPS: {final_fps:.2f}")
