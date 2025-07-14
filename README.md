# 🤟 Sign Language Translator

**Translate Indian Sign Language (ISL) to text using real-time camera input, gesture detection, and machine learning.**  
📷🤖📝 Built with Python, Mediapipe, TensorFlow, and CustomTkinter for an interactive GUI.

---
![SharedScreenshot](https://github.com/user-attachments/assets/0fd6decf-a1e7-4ecd-a096-450f726d8b3d)
![Screenshot 2025-07-14 095917](https://github.com/user-attachments/assets/7e737a00-a281-41eb-bea7-76653a016e96)
---

## 🧩 Problem Statement

Develop an innovative system that utilizes camera technology in web and mobile applications to translate sign language gestures into text. The primary goal is to enhance communication accessibility for the **Deaf and Hard of Hearing** community by providing a **real-time sign language-to-text** translation solution. 🌐🤟📱

---

## 🚀 Key Features

- 🎥 **Real-Time Gesture Recognition**  
  Tracks hand gestures using a live webcam feed via **Mediapipe**.

- 🧠 **ML-Powered Text Translation**  
  Trained using **TensorFlow** to recognize and interpret a wide range of signs.

- 🖼️ **Accessible & Custom GUI**  
  Built using **CustomTkinter** for modern styling and user-friendly experience.

- 🌍 **Multi-language Sign Support**  
  Easily expandable to include signs from other sign languages globally.

- ⚙️ **Customizable Settings**  
  Users can adjust camera, gestures, and output settings for personalized use.

---

## 🌟 Solution Overview

We solved the problem by implementing a comprehensive system that integrates:

- **MediaPipe** for accurate and fast **hand keypoint tracking**
- **TensorFlow** to train and run a **gesture classification model**
- **CustomTkinter** to design a smooth, intuitive **GUI interface**

---

### 🧠 How It Works

1. **Data Collection**  
   - Custom dataset created by capturing hand keypoints using a live webcam.
   - Labels saved per gesture.

2. **Model Training**  
   - A deep learning model is trained using **hand landmark sequences** to learn gestures.

3. **Live Prediction**  
   - During runtime, Mediapipe tracks gestures, and the model classifies them.
   - Translated text is displayed on the screen via GUI.

---

## 🧪 Supported Signs

We trained the system to recognize commonly used ISL gestures like:
- Alphabets: A–Z
- Words: Hello, Yes, No, Thank You, Good, Bad, I Love You, etc.

<img width="267" height="189" alt="image" src="https://github.com/user-attachments/assets/b6b3bd0a-bfc3-4958-a872-906ff87d8144" />

---
## Installation ⚙️
```bash
git clone https://github.com/kanishka-raisania/Sign-Language-Translator.git
cd Sign-Language-Translator
pip install -r requirements.txt
python app.py

## Contributing 🤝
Contributions are welcome! Fork the repo, make changes, and open a pull request.

## Author 👩‍💻
**Kanishka Raisania**  
B.Tech CSE @ NSUT  
