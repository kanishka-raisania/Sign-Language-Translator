import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib

# Load trained model and preprocessor
model = tf.keras.models.load_model('sign_language_model.h5')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# MediaPipe Holistic setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    keypoints = []

    # Pose
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            keypoints += [lm.x, lm.y]
    else:
        keypoints += [0] * 66  # 33 x (x, y)

    # Left hand
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            keypoints += [lm.x, lm.y]
    else:
        keypoints += [0] * 42  # 21 x (x, y)

    # Right hand
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            keypoints += [lm.x, lm.y]
    else:
        keypoints += [0] * 42

    return keypoints

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip and convert color
        image = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Make detections
        results = holistic.process(image_rgb)

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Extract keypoints and predict
        keypoints = extract_keypoints(results)

        if np.count_nonzero(keypoints) > 0:
            X = scaler.transform([keypoints])
            prediction = model.predict(X)
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
            confidence = np.max(prediction)

            if confidence > 0.8:
                cv2.putText(image, f'{predicted_label} ({confidence:.2f})', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('Sign Language Translator', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

