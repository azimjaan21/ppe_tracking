from ultralytics import YOLO

device = "cuda"  #  "cuda"
model = YOLO("yolov8m.pt").to(device)

# Perform person detection and tracking
results = model.track(
    source=0,                      # Webcam source
    show=True,                     # Display results
    tracker="bytetrack.yaml",      # Use ByteTrack with ReID
    classes=[0],                   # Track only 'person' class
    save=True,                     # Save results
    project="output",              # Output folder
    name="cuda_tracking",          # Subfolder name
)

print("Tracking completed with CUDA. Results saved in the 'output' folder.")
