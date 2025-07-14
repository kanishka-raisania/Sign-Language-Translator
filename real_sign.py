import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib

# Load model and label encoder
model = tf.keras.models.load_model("final_model.keras")
label_encoder = joblib.load("label_encoder.pkl")

# MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def extract_keypoints(results):
    keypoints = []
    
    # Pose landmarks (33 points)
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    
    # Face landmarks (468 points)
    if results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    
    # Left hand landmarks (21 points)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    
    # Right hand landmarks (21 points)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    
    print(f"Extracted Keypoints: {len(keypoints)}")  # Debug: Check number of extracted keypoints
    return keypoints

def normalize_keypoints(keypoints):
    coords = np.array(keypoints).reshape(-1, 3)
    print(f"Before Normalization: {coords[:5]}")  # Debug: Print first 5 keypoints before normalization
    coords -= coords[0]  # Normalize by subtracting the first landmark (usually nose or root of pose)
    max_val = np.max(np.abs(coords))
    if max_val != 0:
        coords /= max_val
    print(f"After Normalization: {coords[:5]}")  # Debug: Print first 5 keypoints after normalization
    return coords.flatten()

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    if results.pose_landmarks or results.face_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
        # Draw landmarks on the frame
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_draw.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Extract keypoints
        keypoints = extract_keypoints(results)
        if keypoints:
            norm_kp = normalize_keypoints(keypoints).reshape(1, -1)  # Flatten and make it a 2D array for prediction
            pred = model.predict(norm_kp, verbose=0)
            print(f"Prediction Scores: {pred}")  # Debug: Print the model's prediction scores
            label = label_encoder.inverse_transform([np.argmax(pred)])[0]
            confidence = np.max(pred)

            # Display prediction
            cv2.putText(frame, f'{label} ({confidence:.2f})', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
