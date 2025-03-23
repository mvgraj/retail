import torch
import cv2
import time
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Define paths
video_path = r'C:\code\emb\codew\inputs\Cleaning_acc_section_NVR_2_NVR_2_20250221220849_20250221221420_846094331.mp4'
output_path = r'C:\code\emb\codew\loitering\Cash_5_NVR_1_NVR_1_20250222154635_20250222155252_846899149.mp4'  # Save as .mp4

# Open the video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video Writer Setup
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Change to 'XVID' for AVI
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Tracking Variables
person_positions = {}
next_person_id = 0

# Function to detect loitering
def detect_loitering(person_id, x, y, threshold_time=10):
    current_time = time.time()
    
    if person_id not in person_positions:
        person_positions[person_id] = {
            'last_position': (x, y),
            'time_entered': current_time
        }

    # Calculate time spent in the region
    time_spent = current_time - person_positions[person_id]['time_entered']

    # Update tracking position
    person_positions[person_id]['last_position'] = (x, y)

    if time_spent > threshold_time:
        return True, int(time_spent)  # Return loitering status & time spent
    return False, int(time_spent)

# Confidence Threshold
confidence_threshold = 0.5

# Process Video Frames
start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error.")
        break

    # Perform object detection
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    # Track detections
    current_frame_detections = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if cls == 0 and conf > confidence_threshold:  # Only track 'person'
            current_frame_detections.append([x1, y1, x2, y2, conf, cls])

    # Assign IDs & Track Movement
    for det in current_frame_detections:
        x1, y1, x2, y2, conf, cls = det
        matched = False

        for person_id, data in person_positions.items():
            last_x, last_y = data['last_position']
            distance = np.sqrt((last_x - (x1 + x2) / 2) ** 2 + (last_y - (y1 + y2) / 2) ** 2)
            if distance < 50:
                person_positions[person_id]['last_position'] = ((x1 + x2) / 2, (y1 + y2) / 2)
                matched = True
                break

        if not matched:
            person_id = next_person_id
            next_person_id += 1
            person_positions[person_id] = {'last_position': ((x1 + x2) / 2, (y1 + y2) / 2), 'time_entered': time.time()}

        # Draw bounding box & ID
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {person_id}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Detect loitering with time spent
        loitering, time_spent = detect_loitering(person_id, (x1 + x2) / 2, (y1 + y2) / 2)
        
        if loitering:
            alert_text = f"Loitering: {time_spent}s"
            cv2.putText(frame, alert_text, (int(x1), int(y1)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            print(f"Loitering detected for person ID {person_id} - Time: {time_spent} seconds")

    # Save Frame to Video
    out.write(frame)

    # Show Video (Optional)
    cv2.imshow("Detection & Tracking", frame)

    # Stop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Timeout
    if time.time() - start_time > 60:
        print("Timeout reached.")
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processed video saved to {output_path}")
