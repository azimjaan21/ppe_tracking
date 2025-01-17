import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict, deque
from typing import Dict, List, Tuple
import time

class PPEViolationTracker:
    def __init__(self, 
                 ppe_model_path: str = 'ppe.pt',
                 person_model_path: str = 'yolov8m.pt',
                 violation_threshold: float = 5.0,  #seconds
                 confidence: float = 0.1, #conf classes
                 track_history_length: int = 50):  # Length of tracking path
        """Initialize dual-model tracking system with enhanced visualization"""
        self.ppe_model = YOLO(ppe_model_path).to('cuda')
        self.person_model = YOLO(person_model_path).to('cuda')
        self.confidence = confidence
        self.violation_threshold = violation_threshold
        self.track_history_length = track_history_length
        
        # Track violation timestamps for each worker
        self.violation_tracks = defaultdict(lambda: {
            'start_time': None,
            'duration': 0.0,
            'is_violating': False,
            'track_history': deque(maxlen=track_history_length),  # Store tracking path
            'status_change_time': None  # Track time of last status change
        })
        
        # PPE class colors
        self.ppe_colors = {
            0: (255, 0, 0),    # Helmet - Blue
            1: (0, 255, 0),    # Vest - Green
            2: (0, 0, 255)     # Head - Red
        }


    def _draw_corners(self, frame: np.ndarray, box: Tuple[float, float, float, float], color: Tuple[int, int, int], thickness: int = 2):
            """Draw 90-degree corner markers around a bounding box."""
            x, y, w, h = box
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            corner_len = 15

            # Top-left corner
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)

            # Top-right corner
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)

            # Bottom-left corner
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)

            # Bottom-right corner
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

    def _draw_icons(self, frame: np.ndarray, box: Tuple[float, float, float, float], no_helmet: bool, no_vest: bool):
        """Draw icons above the top-right corner of the bounding box."""
        x, y, w, h = box
        x2, y1 = int(x + w / 2), int(y - h / 2)

        offset = 10
        icon_size = 33  # Icon size (width and height in pixels)
        icon_gap = 5
        current_x = x2 + offset

        # Helper to overlay PNG icons onto the frame
        def overlay_icon(frame, icon_path, position):
            """Overlay a PNG icon with transparency at a given position."""
            icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)  # Load icon with alpha channel
            if icon is None:
                print(f"Error: Unable to load icon: {icon_path}")
                return
            
            # Resize icon to desired size
            icon = cv2.resize(icon, (icon_size, icon_size), interpolation=cv2.INTER_AREA)

            x, y = position
            h, w, _ = icon.shape
            alpha_icon = icon[:, :, 3] / 255.0  # Extract alpha channel and normalize
            alpha_frame = 1.0 - alpha_icon

            # Blend the icon with the frame for each color channel
            for c in range(0, 3):
                frame[y:y+h, x:x+w, c] = (
                    alpha_icon * icon[:, :, c] + alpha_frame * frame[y:y+h, x:x+w, c]
                )

        # Draw "No Helmet" icon if applicable
        if no_helmet:
            overlay_icon(frame, 'no_helmet.png', (current_x, y1 - icon_size))
            current_x += icon_size + icon_gap

        # Draw "No Vest" icon if applicable
        if no_vest:
            overlay_icon(frame, 'no_vest.png', (current_x, y1 - icon_size))



    def _check_ppe_in_person(self, person_box: np.ndarray, frame: np.ndarray) -> Tuple[bool, List]:
        """Check for PPE items within a person's bounding box"""
        x, y, w, h = person_box
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        
        # Extract person ROI
        person_roi = frame[max(0, y1):min(frame.shape[0], y2),
                         max(0, x1):min(frame.shape[1], x2)]
        
        if person_roi.size == 0:
            return False, []
            
        # Run PPE detection on ROI
        ppe_results = self.ppe_model(person_roi)[0]
        
        has_head = False
        has_helmet = False
        ppe_detections = []
        
        # Convert ROI detections to full frame coordinates
        if ppe_results.boxes is not None:
            for det in ppe_results.boxes.data:
                x1_det, y1_det, x2_det, y2_det, conf, cls = det
                
                # Convert coordinates to full frame
                x1_full = x1_det + x1
                y1_full = y1_det + y1
                x2_full = x2_det + x1
                y2_full = y2_det + y1
                
                cls = int(cls)
                if cls == 2:  # head
                    has_head = True
                elif cls == 0:  # helmet
                    has_helmet = True
                
                # Store detection for visualization
                ppe_detections.append({
                    'box': (x1_full, y1_full, x2_full, y2_full),
                    'class': cls,
                    'conf': conf
                })
                
        return (has_head and not has_helmet), ppe_detections
    
    def _update_violation_status(self, track_id: int, has_violation: bool, 
                               current_time: float, person_center: Tuple[float, float]) -> bool:
        """Update violation status and tracking history"""
        track_info = self.violation_tracks[track_id]
        
        # Update tracking path
        track_info['track_history'].append(person_center)
        
        # Handle timing logic
        if track_info['status_change_time'] is None or current_time - track_info['status_change_time'] >= 8.0:
            if has_violation:
                if track_info['start_time'] is None:
                    track_info['start_time'] = current_time
                    track_info['duration'] = 0.0
                else:
                    track_info['duration'] = current_time - track_info['start_time']
                
                track_info['is_violating'] = track_info['duration'] >= self.violation_threshold
            else:
                track_info['start_time'] = None
                track_info['duration'] = 0.0
                track_info['is_violating'] = False
            
            track_info['status_change_time'] = current_time  # Update status change time
        
        return track_info['is_violating']
    
    def _draw_tracking_path(self, frame: np.ndarray, track_id: int, color: Tuple[int, int, int]):
        """Draw tracking path for a worker"""
        track_info = self.violation_tracks[track_id]
        points = list(track_info['track_history'])
        
        # Draw tracking line
        if len(points) > 1:
            for i in range(len(points) - 1):
                pt1 = tuple(map(int, points[i]))
                pt2 = tuple(map(int, points[i + 1]))
                cv2.line(frame, pt1, pt2, color, 2)

            # Add a pink point at the start of the path (last point)
        if len(points) > 0:
            end_point = tuple(map(int, points[-1]))  # Las point (end of path)
            pink_color = (230, 0, 255)  # Pink in BGR
            cv2.circle(frame, end_point, 5, pink_color, -1)  # Draw the circle with a radius of 5

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[int, bool]]:
            """Process a single frame and detect PPE violations."""
            current_time = time.time()
            violation_status = {}

            try:
                # First, detect and track persons
                person_results = self.person_model.track(frame, persist=True, conf=self.confidence)[0]
                
                if person_results.boxes is not None and hasattr(person_results.boxes, 'id'):
                    boxes = person_results.boxes.xywh.cpu()
                    track_ids = person_results.boxes.id.int().cpu().tolist()
                    classes = person_results.boxes.cls.cpu().tolist()
                    
                    # Process each tracked person
                    for track_id, cls, box in zip(track_ids, classes, boxes):
                        # Only process if it's a person (class 0 in COCO dataset)
                        if cls == 0:
                            x, y, w, h = box
                            person_center = (x, y)
                            
                            # Check for PPE violations
                            has_violation, ppe_detections = self._check_ppe_in_person(box, frame)
                            
                            # Update violation status
                            is_violating = self._update_violation_status(
                                track_id, has_violation, current_time, person_center)
                            violation_status[track_id] = is_violating
                            
                            # Draw person corners and icons
                            color = (0, 0, 255) if is_violating else (0, 255, 0)
                            self._draw_corners(frame, box, color)
                            
                            no_helmet = not any(det['class'] == 0 for det in ppe_detections)
                            no_vest = not any(det['class'] == 1 for det in ppe_detections)
                            self._draw_icons(frame, box, no_helmet, no_vest)

                            # Draw tracking path
                            self._draw_tracking_path(frame, track_id, (0, 255, 255))

                # Draw detection info
                num_workers = len(violation_status)
                num_violations = sum(violation_status.values())
                cv2.putText(frame, f"Workers: {num_workers}, Violations: {num_violations}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                        
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                
            return frame, violation_status
    
    ##video running process webcam
    import cv2
    import torch
    import time
    from typing import Tuple, Dict

    def run_on_video(self, source: str, output_path: str = None):
        """Run PPE violation detection on video source with FPS counter and device info"""
        try:
            if isinstance(source, str) and source.isdigit():
                source = int(source)
            cap = cv2.VideoCapture(source)
            
            if not cap.isOpened():
                raise ValueError(f"Failed to open video source: {source}")
            
            if output_path:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                out = cv2.VideoWriter(output_path, 
                                    cv2.VideoWriter_fourcc(*'mp4v'),
                                    fps, (width, height))
            
            # Set up a named window for full-screen display
            cv2.namedWindow('PPE Violation Detection', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('PPE Violation Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
            # Get device information
            device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            
            # Initialize FPS calculation variables
            prev_time = time.time()
            fps_counter = 0
            fps_display = 0
            fps_update_interval = 0.5  # Update FPS every 0.5 seconds
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Calculate FPS
                fps_counter += 1
                current_time = time.time()
                elapsed_time = current_time - prev_time
                
                # Update FPS display every update_interval seconds
                if elapsed_time > fps_update_interval:
                    fps_display = round(fps_counter / elapsed_time, 1)
                    fps_counter = 7
                    prev_time = current_time
                
                processed_frame, violation_status = self.process_frame(frame)
                
                # Print violation alerts
                for worker_id, is_violating in violation_status.items():
                    if is_violating:
                        duration = self.violation_tracks[worker_id]['duration']
                        print(f"ALERT: Worker {worker_id} has no helmet for {duration:.1f} seconds!")
                
                # Add FPS and device info to frame
                frame_h, frame_w = processed_frame.shape[:2]
                fps_text = f"FPS: {fps_display}"
                device_text = f"Device: {device}"
                
                # Get text sizes for positioning
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                fps_size = cv2.getTextSize(fps_text, font, font_scale, thickness)[0]
                device_size = cv2.getTextSize(device_text, font, font_scale, thickness)[0]
                
                # Calculate positions (bottom-right corner with padding)
                padding = 10
                fps_pos = (frame_w - fps_size[0] - padding, frame_h - fps_size[1] - padding - device_size[1] - 5)
                device_pos = (frame_w - device_size[0] - padding, frame_h - padding)
                
                # Draw background rectangles
                bg_padding = 5
                cv2.rectangle(processed_frame, 
                            (fps_pos[0] - bg_padding, fps_pos[1] - fps_size[1] - bg_padding),
                            (fps_pos[0] + fps_size[0] + bg_padding, fps_pos[1] + bg_padding),
                            (0, 0, 0), -1)
                cv2.rectangle(processed_frame,
                            (device_pos[0] - bg_padding, device_pos[1] - device_size[1] - bg_padding),
                            (device_pos[0] + device_size[0] + bg_padding, device_pos[1] + bg_padding),
                            (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(processed_frame, fps_text, fps_pos, font, font_scale, (255, 255, 255), thickness)
                cv2.putText(processed_frame, device_text, device_pos, font, font_scale, (255, 255, 255), thickness)
                
                
                cv2.imshow('PPE Violation Detection', processed_frame)
                
                if output_path:
                    out.write(processed_frame)
                    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Error running video detection: {str(e)}")
            
        finally:
            if 'cap' in locals():
                cap.release()
            if 'out' in locals():
                out.release()
            cv2.destroyAllWindows()

    def run_on_image(self, image_path: str, output_path: str = None):
        """Run PPE violation detection on a single image"""
        try:
            # Read the input image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to read image from path: {image_path}")

            # Process the image
            processed_image, violation_status = self.process_frame(image)

            # Print violation alerts
            for worker_id, is_violating in violation_status.items():
                if is_violating:
                    duration = self.violation_tracks[worker_id]['duration']
                    print(f"ALERT: Worker {worker_id} has no helmet for {duration:.1f} seconds!")

            # Save or display the image
            if output_path:
                cv2.imwrite(output_path, processed_image)
                print(f"Processed image saved to: {output_path}")
            else:
                cv2.imshow('PPE Violation Detection', processed_image)
                cv2.waitKey(0)

        except Exception as e:
            print(f"Error processing image: {str(e)}")

        finally:
            cv2.destroyAllWindows()
