import cv2
import time
from ultralytics import YOLO
import torch
import numpy as np
from scipy.spatial.distance import cdist
from torchreid import models
from torchvision import transforms

# Function to generate unique color for each track ID
def generate_color(track_id):
    np.random.seed(int(track_id))
    return tuple(np.random.randint(0, 255, 3).tolist())

# Initialize ReID model
def initialize_reid_model():
    model = models.build_model(name='osnet_x1_0', num_classes=1000, pretrained=True)
    model.eval().cuda()
    return model

# Function to extract embeddings for ReID
def extract_embedding(frame, box, model, transform):
    x1, y1, x2, y2 = map(int, box)
    cropped_person = frame[y1:y2, x1:x2]
    resized = cv2.resize(cropped_person, (128, 256))
    input_tensor = transform(resized).unsqueeze(0).cuda()
    with torch.no_grad():
        embedding = model(input_tensor).cpu().numpy()
    return embedding

# Initialize YOLOv8 and ReID model
yolo_model = YOLO('yolov8m.pt').cuda()
reid_model = initialize_reid_model()

# Preprocessing for ReID
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Video capture and full-screen display
cap = cv2.VideoCapture('people.mp4')
cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Tracking variables
tracks = {}
next_track_id = 1
max_age = 30  # Max frames to keep a track without updates

# FPS calculation
prev_frame_time = 0
confidence_threshold = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform YOLO detection
    results = yolo_model(frame)
    detections = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, confidence, class]

    updated_tracks = {}

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if cls != 0 or conf < confidence_threshold:
            continue

        new_embedding = extract_embedding(frame, (x1, y1, x2, y2), reid_model, transform)

        # Match with existing tracks
        matched_id = None
        if tracks:
            track_ids = list(tracks.keys())
            existing_embeddings = np.vstack([tracks[tid]['embedding'] for tid in track_ids])
            distances = cdist(new_embedding, existing_embeddings, metric='cosine')
            min_distance_idx = np.argmin(distances)
            if distances[0][min_distance_idx] < 0.4:  # Adjust threshold as needed
                matched_id = track_ids[min_distance_idx]

        # Assign new ID if no match found
        if matched_id is None:
            matched_id = next_track_id
            next_track_id += 1

        # Update track information
        if matched_id not in updated_tracks:
            updated_tracks[matched_id] = {
                'embedding': new_embedding,
                'bbox': (x1, y1, x2, y2),
                'age': 0
            }
        else:
            # Use moving average for embedding
            updated_tracks[matched_id]['embedding'] = (
                0.8 * updated_tracks[matched_id]['embedding'] + 0.2 * new_embedding
            )
            updated_tracks[matched_id]['bbox'] = (x1, y1, x2, y2)
            updated_tracks[matched_id]['age'] = 0

        # Draw bounding box and ID
        color = generate_color(matched_id)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"ID: {matched_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Remove old tracks
    for track_id, data in tracks.items():
        if track_id not in updated_tracks:
            data['age'] += 1
            if data['age'] <= max_age:
                updated_tracks[track_id] = data

    tracks = updated_tracks

    # Display approach
    approach_text1 = "Bot-SORT"
    approach_text2 = "+ OSNet (re-ID)"

    # Calculate size of each text part
    text1_size = cv2.getTextSize(approach_text1, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
    text2_size = cv2.getTextSize(approach_text2, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]

    # Calculate total width and position for top center alignment
    total_width = text1_size[0] + text2_size[0] + 5  # 5 is the space between the texts
    x_center = (frame.shape[1] - total_width) // 2  # Centering on the x-axis

    # For Bot-SORT (yellow color)
    cv2.putText(frame, approach_text1, (x_center, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

    # For + OSNet (re-ID) (red color)
    cv2.putText(frame, approach_text2, (x_center + text1_size[0] + 5, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)


    # Show the frame
    cv2.imshow("Tracking", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
