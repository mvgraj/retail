import cv2
import os
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

# Load YOLO face detection model
yolo_model = YOLO(r"C:\code\emb\codew\yolov8n-face.pt")

# Path to input video
video_path = r"C:\code\emb\codew\new_in\Shoplifting (1).mp4"
cap = cv2.VideoCapture(video_path)

# Create directory to save extracted faces and embeddings
save_path = r"C:\code\emb\codew\new_in\extracted_faces1234"
os.makedirs(save_path, exist_ok=True)

frame_count = 0  # Track frame number
face_id_counter = 0  # Unique face ID counter
face_embeddings = {}  # Dictionary to store face IDs and their embeddings

def expand_bounding_box(x1, y1, x2, y2, frame_shape, padding=0.2):
    """
    Expands the bounding box by a percentage (default 20%) to capture the full face.
    """
    width = x2 - x1
    height = y2 - y1
    x1 = max(0, int(x1 - width * padding))
    y1 = max(0, int(y1 - height * padding))
    x2 = min(frame_shape[1], int(x2 + width * padding))
    y2 = min(frame_shape[0], int(y2 + height * padding))
    return x1, y1, x2, y2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Run YOLO face detection
    results = yolo_model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()

            if conf > 0.5:  # Confidence filter
                # Expand the bounding box to capture the full face
                x1, y1, x2, y2 = expand_bounding_box(x1, y1, x2, y2, frame.shape)

                # Crop the face region
                face_crop = frame[y1:y2, x1:x2]

                try:
                    # Generate embedding using DeepFace
                    embedding_list = DeepFace.represent(face_crop, model_name="Facenet", enforce_detection=False)
                    
                    # Ensure embedding_list is a list and extract the first embedding
                    if isinstance(embedding_list, list) and len(embedding_list) > 0:
                        embedding_dict = embedding_list[0]  # Take the first face
                        embedding = np.array(embedding_dict["embedding"])  # Extract embedding vector
                    else:
                        print("No valid embedding found for the face.")
                        continue
                except ValueError:
                    # Skip if face cannot be detected by DeepFace
                    print("Face detection failed with DeepFace.")
                    continue

                # Compare with existing embeddings to check if it's the same face
                match_found = False
                current_face_id = None
                best_similarity = -1  # To track the highest similarity score
                for face_id, stored_embedding in face_embeddings.items():
                    similarity = cosine_similarity([embedding], [stored_embedding])[0][0]
                    if similarity > 0.6:  # Stricter threshold for cosine similarity
                        match_found = True
                        current_face_id = face_id
                        best_similarity = similarity
                        break

                if not match_found:
                    # Assign a new face ID
                    current_face_id = face_id_counter
                    face_id_counter += 1  # Increment the counter for the next face
                    face_embeddings[current_face_id] = embedding  # Store the new embedding
                    print(f"New face detected with ID {current_face_id}")

                    # Save the cropped face and embedding
                    face_filename = os.path.join(save_path, f"face_{current_face_id}.jpg")
                    embedding_filename = os.path.join(save_path, f"embedding_{current_face_id}.npy")

                    cv2.imwrite(face_filename, face_crop)  # Save the face image
                    np.save(embedding_filename, embedding)  # Save the embedding
                else:
                    print(f"Match found with face ID {current_face_id}, similarity: {best_similarity:.2f}")
                    continue  # Skip saving if the face is already detected

                # Draw bounding box and label with face ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Face ID: {current_face_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()