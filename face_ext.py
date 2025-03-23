import cv2
import os
from ultralytics import YOLO

# Load YOLO face detection model
model = YOLO("yolov8n-face.pt")

# Path to input video
video_path = r"C:\code\emb\codew\new_in\Shoplifting (1).mp4"
cap = cv2.VideoCapture(video_path)

# Create directory to save extracted faces
save_path = r"C:\code\emb\codew\new_in\extracted_faces"
os.makedirs(save_path, exist_ok=True)

frame_count = 0  # Track frame number

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Run YOLO face detection
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()

            if conf > 0.5:  # Confidence filter
                face_crop = frame[y1:y2, x1:x2]

                # Save the cropped face
                face_filename = os.path.join(save_path, f"face_{frame_count}.jpg")
                cv2.imwrite(face_filename, face_crop)

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
