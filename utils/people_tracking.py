import cv2
import time
from ultralytics import YOLO
import torch
import numpy as np
from scipy.spatial.distance import cdist
from torchreid import models
from torchreid.data_preprocessing import preprocess

# Function to generate unique color for each track ID
def generate_color(track_id):
    np.random.seed(int(track_id))  # Seed with track_id to ensure unique color
    return tuple(np.random.randint(0, 255, 3).tolist())  # Random color (R, G, B)

# Initialize ReID model
def initialize_reid_model():
    model = models.build_model(name='osnet_x1_0', num_classes=1000, pretrained=True)  # OSNet ReID model
    model.eval().cuda()
    return model

# Function to extract embeddings for ReID using pretrained model
def extract_embedding(frame, box, model, transform):
    x1, y1, x2, y2 = map(int, box)
    cropped_person = frame[y1:y2, x1:x2]
    resized = cv2.resize(cropped_person, (256, 128))  # Resize for ReID model input
    input_tensor = transform(resized).unsqueeze(0).cuda()  # Preprocess image and convert to tensor
    embedding = model(input_tensor).cpu().detach().numpy()
    return embedding

# Initialize YOLOv8 model and ReID model
model = YOLO('yolov8m.pt')
model.cuda()
device_name = torch.cuda.get_device_name(0)

# Initialize ReID model
reid_model = initialize_reid_model()

# Open video capture
cap = cv2.VideoCapture(0)

# Full-screen display
cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# FPS calculation variables
prev_frame_time = 0
confidence_threshold = 0.5

# Font settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
thickness = 1
bg_padding = 5

# ReID tracking variables
tracks = {}
next_track_id = 0

# Image transform for ReID
from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # Perform YOLOv8 detection and tracking
    results = model.track(frame, classes=[0], persist=True)

    current_tracks = []  # Temporary storage for current frame's tracks

    for result in results:
        for box in result.boxes:
            # Check if detected class is 'person'
            if box.cls[0] == 0:
                conf_score = box.conf[0].item()
                if conf_score >= confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
                    new_embedding = extract_embedding(frame, (x1, y1, x2, y2), reid_model, transform)  # Extract embedding
                    
                    # Match with existing tracks using embedding similarity
                    matched_id = None
                    if tracks:
                        track_ids = list(tracks.keys())
                        existing_embeddings = np.vstack([tracks[tid]['embedding'] for tid in track_ids])
                        distances = cdist(new_embedding, existing_embeddings, metric='cosine')
                        min_distance_idx = np.argmin(distances)
                        if distances[0][min_distance_idx] < 0.5:  # Threshold for similarity
                            matched_id = track_ids[min_distance_idx]

                    # Assign new ID if no match found
                    if matched_id is None:
                        matched_id = next_track_id
                        tracks[matched_id] = {'embedding': new_embedding, 'bbox': (x1, y1, x2, y2)}
                        next_track_id += 1
                    else:
                        # Update track with new information
                        tracks[matched_id]['embedding'] = new_embedding
                        tracks[matched_id]['bbox'] = (x1, y1, x2, y2)

                    # Draw bounding box and person ID
                    color = generate_color(matched_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"Person ID: {matched_id}", (x1, y1 - 10),
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

    # Display the result
    cv2.imshow("Tracking", frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
