from ppe_violation_tracker import PPEViolationTracker

def main():
    # Initialize the PPE violation tracking system
    detector = PPEViolationTracker(
    ppe_model_path='ppe.pt',        # Your PPE detection model
    person_model_path='yolov8m.pt', # Person detection model (n*)(m)
    violation_threshold=5.0,         # 5 seconds threshold
    confidence=0.49, #75
    
)
    
    # # For output
    #detector.run_on_video('cctv1.mp4', 'output/cctv1.mp4')

    #  # Process a single image
    # input_image_path = "1.jpg"      # Path to your input image
    # output_image_path = "output.jpg"   # Path to save the processed image
    # detector.run_on_image(input_image_path, output_image_path)
    
    # For WebCam
    detector.run_on_video(0)

if __name__ == "__main__":
    main()

    


        #         def run_on_image(self, source: str, output_path: str = None):
        # """Run PPE violation detection on a single image."""
        # try:
        #     # Read the input image
        #     frame = cv2.imread(source)
        #     if frame is None:
        #         raise ValueError(f"Failed to read image from {source}")
            
        #     # Process the image
        #     processed_frame, violation_status = self.process_frame(frame)
            
        #     # Save the processed image if an output path is specified
        #     if output_path:
        #         cv2.imwrite(output_path, processed_frame)
        #         print(f"Processed image saved to {output_path}")
            
        #     # Display the processed image
        #     cv2.imshow('PPE Violation Detection', processed_frame)
        #     cv2.waitKey(0)  # Wait for a key press to close the window
        #     cv2.destroyAllWindows()
        
        # except Exception as e:
        #     print(f"Error processing image: {str(e)}")



            #     # Create a resizable window and maximize it
            # cv2.namedWindow('PPE Violation Detection', cv2.WINDOW_NORMAL)
            # cv2.setWindowProperty('PPE Violation Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # whileTRYE
