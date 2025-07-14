# Final GUI-Based Real-Time Sign Language Translator using trained model

import customtkinter as ctk
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib

# Load trained model and label encoder
model = tf.keras.models.load_model('sign_model.h5')
label_encoder = joblib.load('label_encoder.pkl')

# Mediapipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Extract keypoints from holistic results
def extract_keypoints(results):
    keypoints = []
    # Pose
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            keypoints += [lm.x, lm.y]
    else:
        keypoints += [0] * 66

    # Left Hand
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            keypoints += [lm.x, lm.y]
    else:
        keypoints += [0] * 42

    # Right Hand
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            keypoints += [lm.x, lm.y]
    else:
        keypoints += [0] * 42

    return np.array(keypoints)

# Create the GUI
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
window = ctk.CTk()
window.geometry("1100x900")
window.title("Sign Language Translator")

# Fonts
title_font = ctk.CTkFont(family='Consolas', size=26, weight='bold')
label_font = ctk.CTkFont(family='Consolas', size=200, weight='bold')

# Title
title = ctk.CTkLabel(window, text="Sign Language Translator", font=title_font, text_color="white")
title.pack(pady=10)

# Video Display
video_frame = ctk.CTkFrame(window, corner_radius=10)
video_frame.pack(pady=10)
video_label = ctk.CTkLabel(video_frame, text="")
video_label.pack()

# Detected Sign Display
result_label = ctk.CTkLabel(window, text="", font=label_font, text_color="white", fg_color="#2B2B2B")
result_label.pack(pady=10, fill=ctk.X, padx=20)

# Initialize video
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Prediction loop
def detect():
    ret, frame = cap.read()
    if not ret:
        return

    image = cv2.flip(frame, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        results = holistic.process(rgb_image)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        keypoints = extract_keypoints(results)

        if np.count_nonzero(keypoints) > 0:
            input_data = keypoints / np.max(keypoints)
            prediction = model.predict(np.expand_dims(input_data, axis=0))[0]
            confidence = np.max(prediction)
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

            if confidence > 0.7:
                result_label.configure(text=f"{predicted_label}")

    # Display on GUI
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    img = cv2.resize(img, (640, 480))
    img_pil = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    video_label.configure(image=img_tk)
    video_label.image = img_tk
    video_label.after(10, detect)

# Start Button
start_btn = ctk.CTkButton(window, text="Start Camera", command=detect)
start_btn.pack(pady=10)

# Start GUI
window.mainloop()

# Release camera after window closed
cap.release()
cv2.destroyAllWindows()
