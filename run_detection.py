from ppe_violation_tracker import PPEViolationTracker

def main():
    # Initialize the PPE violation tracking system
    detector = PPEViolationTracker(
    ppe_model_path='ppe.pt',        # Your PPE detection model
    person_model_path='yolov8m.pt', # Person detection model (n*)(m)
    violation_threshold=5.0,         # 5 seconds threshold
    confidence=0.5, #75
    
)
    
    # # For Vide output
    # detector.run_on_video('frame.png', 'output/frame.png')

     # Process a single image
    input_image_path = "this.jpg"      # Path to your input image
    output_image_path = "output/this.jpg"   # Path to save the processed image
    detector.run_on_image(input_image_path, output_image_path)
    
    # For WebCam
    # detector.run_on_video(0)

if __name__ == "__main__":
    main()

    