from ppe_violation_tracker import PPEViolationTracker

def main():
    # Initialize the PPE violation tracking system
    detector = PPEViolationTracker(
    ppe_model_path='ppe.pt',        # Your PPE detection model
    person_model_path='yolov8m.pt', # Person detection model (n*)(m)
    violation_threshold=5.0,         # 5 seconds threshold
    confidence=0.4, #conf
    
)
    # # For WebCam
    detector.run_on_video(0)

if __name__ == "__main__":
    main()

    