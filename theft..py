import cv2
import torch
from ultralytics import YOLO
import numpy as np
import os
import time

# Load models
pose_model = YOLO("yolo11n-pose.pt")  # Pose estimation model

# Directories
input_dir = r"C:\code\emb\codew\new_in"
output_dir = r"C:\code\emb\out_maybe_2"
screenshot_dir = os.path.join(output_dir, "screenshots")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(screenshot_dir, exist_ok=True)

# Processing settings
skip_frames = 3
hand_stay_time_chest = 3.0      # seconds before suspicion (chest region)
hand_stay_time_waist = 3.0      # Increased time for waist region
crop_padding = 50               # Padding for crop area to include more details

# COCO Pose Keypoints
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
NOSE = 0  # Used for front/back detection

# Tracking dictionaries
hand_timers = {}
previous_hand_positions = {}

# Person tracking
person_tracker = {}  # {person_id: (x1, y1, x2, y2, last_seen_time)}
next_person_id = 1   # Counter for assigning unique IDs

# Intersection-over-Union (IoU) for person tracking
def calculate_iou(box1, box2):
    """
    Calculate Intersection-over-Union (IoU) between two bounding boxes.
    """
    x1_intersect = max(box1[0], box2[0])
    y1_intersect = max(box1[1], box2[1])
    x2_intersect = min(box1[2], box2[2])
    y2_intersect = min(box1[3], box2[3])
    intersection_area = max(0, x2_intersect - x1_intersect) * max(0, y2_intersect - y1_intersect)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersection_area / float(box1_area + box2_area - intersection_area) if (box1_area + box2_area - intersection_area) > 0 else 0
    return iou

# Function to draw skeleton
def draw_skeleton(frame, keypoints, color=(0, 255, 0), thickness=2):
    """
    Draw a skeleton on the frame based on keypoints.
    """
    # Define connections between keypoints (COCO format)
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head and shoulders
        (5, 6), (5, 7), (6, 8),          # Torso and arms
        (7, 9), (8, 10),                 # Arms
        (5, 11), (6, 12),                # Torso to legs
        (11, 12),                        # Hips
        (11, 13), (12, 14),              # Legs
        (13, 15), (14, 16)               # Ankles
    ]
    # Draw keypoints
    for keypoint in keypoints:
        if keypoint[0] > 0 and keypoint[1] > 0:  # Only draw visible keypoints
            x, y = int(keypoint[0]), int(keypoint[1])
            cv2.circle(frame, (x, y), radius=5, color=color, thickness=-1)
    # Draw connections
    for connection in connections:
        idx1, idx2 = connection
        if keypoints[idx1][0] > 0 and keypoints[idx1][1] > 0 and keypoints[idx2][0] > 0 and keypoints[idx2][1] > 0:
            pt1 = tuple(map(int, keypoints[idx1][:2]))
            pt2 = tuple(map(int, keypoints[idx2][:2]))
            cv2.line(frame, pt1, pt2, color, thickness)

# Process videos
for file_name in os.listdir(input_dir):
    file_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, f"processed_{file_name}")
    if not file_name.lower().endswith((".mp4", ".avi", ".mov")):
        continue

    cap = cv2.VideoCapture(file_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_rate = int(cap.get(5))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames != 0:
            frame_count += 1
            continue
        frame_count += 1

        # Detect people and pose keypoints
        pose_results = pose_model(frame)

        # Temporary dictionary to store current detections for tracking
        current_detections = {}
        for result in pose_results:
            keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else []
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []

            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[i])
                width, height = x2 - x1, y2 - y1

                # Assign unique ID to each person using IoU-based tracking
                matched_person_id = None
                for person_id, (px1, py1, px2, py2, last_seen) in person_tracker.items():
                    iou = calculate_iou([x1, y1, x2, y2], [px1, py1, px2, py2])
                    if iou > 0.5:  # Threshold for matching
                        matched_person_id = person_id
                        break
                if matched_person_id is None:
                    matched_person_id = next_person_id
                    next_person_id += 1

                # Update person tracker
                person_tracker[matched_person_id] = (x1, y1, x2, y2, time.time())
                current_detections[matched_person_id] = (x1, y1, x2, y2)

                # Chest & Waist Bounding Boxes
                chest_box = [x1 + int(0.1 * width), y1, x2 - int(0.1 * width), y1 + int(0.4 * height)]
                left_waist_box = [x1, y1 + int(0.5 * height), x1 + int(0.5 * width), y2]
                right_waist_box = [x1 + int(0.5 * width), y1 + int(0.5 * height), x2, y2]

                # Draw Detection Zones
                cv2.rectangle(frame, tuple(chest_box[:2]), tuple(chest_box[2:]), (0, 255, 255), 2)
                cv2.rectangle(frame, tuple(left_waist_box[:2]), tuple(left_waist_box[2:]), (255, 0, 0), 2)
                cv2.rectangle(frame, tuple(right_waist_box[:2]), tuple(right_waist_box[2:]), (0, 0, 255), 2)

                if len(keypoints) <= i:
                    continue

                person_keypoints = keypoints[i]

                # Draw skeleton
                draw_skeleton(frame, person_keypoints)

                # Display person ID
                cv2.putText(frame, f"ID: {matched_person_id}", (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                left_wrist = person_keypoints[LEFT_WRIST] if person_keypoints[LEFT_WRIST][0] > 0 else None
                right_wrist = person_keypoints[RIGHT_WRIST] if person_keypoints[RIGHT_WRIST][0] > 0 else None

                for wrist, label in [(left_wrist, "left"), (right_wrist, "right")]:
                    if wrist is None:
                        continue

                    wrist_x, wrist_y = int(wrist[0]), int(wrist[1])
                    wrist_box = [wrist_x - 10, wrist_y - 10, wrist_x + 10, wrist_y + 10]
                    cv2.rectangle(frame, (wrist_box[0], wrist_box[1]), (wrist_box[2], wrist_box[3]), (0, 255, 0), 2)

                    def is_intersecting(box1, box2):
                        return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])

                    in_chest = is_intersecting(wrist_box, chest_box)
                    in_left_waist = is_intersecting(wrist_box, left_waist_box)
                    in_right_waist = is_intersecting(wrist_box, right_waist_box)

                    # Reset timers when hands leave the region
                    if label in previous_hand_positions:
                        prev_in_chest, prev_in_left_waist, prev_in_right_waist = previous_hand_positions[label]
                        if not (in_chest or in_left_waist or in_right_waist):
                            hand_timers.pop(label, None)  # Reset timer if hand leaves all regions

                    # Suspicious behavior detection
                    if in_chest or in_left_waist or in_right_waist:
                        hand_stay_time = hand_stay_time_waist if in_left_waist or in_right_waist else hand_stay_time_chest
                        if label not in hand_timers:
                            hand_timers[label] = time.time()
                        elif time.time() - hand_timers[label] > hand_stay_time:
                            cv2.putText(frame, "⚠️ Shoplifter!", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            # Crop the suspect with padding
                            x1_crop = max(0, x1 - crop_padding)
                            y1_crop = max(0, y1 - crop_padding)
                            x2_crop = min(frame_width, x2 + crop_padding)
                            y2_crop = min(frame_height, y2 + crop_padding)
                            suspect_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                            crop_filename = os.path.join(screenshot_dir, f"crop_{file_name}_{frame_count}.jpg")
                            cv2.imwrite(crop_filename, suspect_crop)
                        previous_hand_positions[label] = (in_chest, in_left_waist, in_right_waist)
                    else:
                        hand_timers.pop(label, None)

        # Remove stale person tracks
        for person_id in list(person_tracker.keys()):
            if time.time() - person_tracker[person_id][4] > 2:  # If person not seen for 2 seconds
                del person_tracker[person_id]

        out.write(frame)
        cv2.imshow("Theft Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()

cv2.destroyAllWindows()
print("✅ Processing complete!")