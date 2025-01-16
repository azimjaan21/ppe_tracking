import cv2
import numpy as np
from ultralytics import YOLO

# Load models
person_model = YOLO("yolov8m.pt")  # For detecting person
ppe_model = YOLO("ppe.pt")         # For detecting helmet, vest, head

# Load icons
helmet_icon = cv2.imread("nohelmet.png", cv2.IMREAD_UNCHANGED)  # PNG with transparency
vest_icon = cv2.imread("vest.png", cv2.IMREAD_UNCHANGED)

# Validate icons
if helmet_icon is None or helmet_icon.shape[2] != 4:
    raise ValueError("Helmet icon must be a valid PNG image with an alpha channel.")
if vest_icon is None or vest_icon.shape[2] != 4:
    raise ValueError("Vest icon must be a valid PNG image with an alpha channel.")

def overlay_icon(image, icon, position):
    """Overlay an icon with transparency at a given position."""
    icon_h, icon_w, _ = icon.shape
    x, y = position

    # Handle out-of-bound coordinates
    if y < 0: y = 0
    if x + icon_w > image.shape[1]: x = image.shape[1] - icon_w
    if y + icon_h > image.shape[0]: y = image.shape[0] - icon_h

    # Region of interest
    roi = image[y:y+icon_h, x:x+icon_w]

    # Separate the alpha channel
    icon_rgb = icon[:, :, :3]
    icon_alpha = icon[:, :, 3] / 255.0

    # Blend the icon with the ROI
    for c in range(3):
        roi[:, :, c] = (icon_alpha * icon_rgb[:, :, c] + (1 - icon_alpha) * roi[:, :, c])

    image[y:y+icon_h, x:x+icon_w] = roi

def process_frame(frame):
    """
    Process a single frame for person and PPE detection.
    :param frame: Input image frame
    :return: Annotated frame
    """
    # Run person detection
    person_results = person_model(frame)
    person_boxes = [
        (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        for box in person_results[0].boxes.xyxy.cpu().numpy()
    ]

    # Run PPE detection
    ppe_results = ppe_model(frame)
    ppe_detections = []
    for box in ppe_results[0].boxes.xyxy.cpu().numpy():
        if len(box) >= 6:
            ppe_detections.append((
                ppe_model.names[int(box[5])],  # Label
                float(box[4]),                # Confidence
                int(box[0]), int(box[1]),     # Bounding box
                int(box[2]), int(box[3])
            ))

    # Map each person to their PPE status
    for (x_min, y_min, x_max, y_max) in person_boxes:
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # Check PPE items within this person's bounding box
        missing_items = []
        has_helmet = has_vest = False
        for label, confidence, px_min, py_min, px_max, py_max in ppe_detections:
            if px_min >= x_min and py_min >= y_min and px_max <= x_max and py_max <= y_max:
                if label == "helmet":
                    has_helmet = True
                elif label == "vest":
                    has_vest = True

        if not has_helmet:
            missing_items.append("helmet")
        if not has_vest:
            missing_items.append("vest")

        # Overlay icons for missing items
        icon_x, icon_y = x_min, y_min - 50  # Position above the bounding box
        for item in missing_items:
            if item == "helmet":
                overlay_icon(frame, helmet_icon, (icon_x, icon_y))
                icon_x += helmet_icon.shape[1] + 5  # Add spacing
            elif item == "vest":
                overlay_icon(frame, vest_icon, (icon_x, icon_y))

    return frame

# Example usage
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open camera. Please check your camera connection.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame
    annotated_frame = process_frame(frame)

    # Display the frame
    cv2.imshow("PPE Monitoring", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
