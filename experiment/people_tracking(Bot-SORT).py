import cv2
import time
from ultralytics import YOLO
import torch
import numpy as np

# Function to generate unique color for each track ID
def generate_color(track_id):
    np.random.seed(int(track_id))  # Seed with track_id to ensure unique color
    return tuple(np.random.randint(0, 255, 3).tolist())  # Random color (R, G, B)

# Load YOLOv8m model and move it to CUDA (GPU)
model = YOLO('yolov8m.pt')
model.cuda()  # Move the model to CUDA (GPU)
# Get CUDA device name
device_name = torch.cuda.get_device_name(0)

# Open video capture (use your camera or video file)
cap = cv2.VideoCapture('people.mp4')  # 0 for the default webcam, or replace with video file path

# Create a named window for full screen
cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# FPS calculation variables
prev_frame_time = 0
confidence_threshold = 0.5  # Confidence score threshold for detecting persons

# Font settings for text
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
thickness = 1
bg_padding = 5  # Padding around the text

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get the current time to calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # Perform YOLOv8 detection and tracking
    results = model.track(frame, classes=[0], persist=True)  # `persist=True` to maintain track IDs

    for result in results:
        for box in result.boxes:
            # Check if the detected class is 'person' (class_id = 0)
            if box.cls[0] == 0:  # 'person' class
                conf_score = box.conf[0].item()  # Get confidence score
                if conf_score >= confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract bounding box coordinates
                    track_id = int(box.id.item())  # Get track ID
                    
                    # Generate unique color for track ID
                    color = generate_color(track_id)

                    # Draw bounding box and person ID
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"Person ID: {track_id}", (x1, y1 - 10),
                                font, 0.9, color, 2)

    # Display FPS and device name
    fps_text = f"FPS: {fps:.2f}"
    device_text = f"Device: {device_name}"
    fps_size = cv2.getTextSize(fps_text, font, font_scale, thickness)[0]
    device_size = cv2.getTextSize(device_text, font, font_scale, thickness)[0]
    fps_pos = (10, fps_size[1] + 10)
    device_pos = (10, fps_pos[1] + device_size[1] + 5)

    cv2.rectangle(frame,
                  (fps_pos[0] - bg_padding, fps_pos[1] - fps_size[1] - bg_padding),
                  (fps_pos[0] + fps_size[0] + bg_padding, fps_pos[1] + bg_padding),
                  (0, 0, 0), -1)
    cv2.rectangle(frame,
                  (device_pos[0] - bg_padding, device_pos[1] - device_size[1] - bg_padding),
                  (device_pos[0] + device_size[0] + bg_padding, device_pos[1] + bg_padding),
                  (0, 0, 0), -1)

    cv2.putText(frame, fps_text, fps_pos, font, font_scale, (255, 255, 0), thickness)
    cv2.putText(frame, device_text, device_pos, font, font_scale, (255, 0, 255), thickness)

    # Display approach
    approach_text = f"Bot-SORT (default)"
 # Calculate size of each text part
    text1_size = cv2.getTextSize(approach_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]

    # Calculate total width and position for top center alignment
    total_width = text1_size[0] + 5  # 5 is the space between the texts
    x_center = (frame.shape[1] - total_width) // 2  # Centering on the x-axis

    # For Bot-SORT (blue color)
    cv2.putText(frame, approach_text, (x_center, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
        # Display the result
    cv2.imshow("Tracking", frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
