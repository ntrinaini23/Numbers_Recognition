import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Paths
MODEL_PATH = "gesture_recognition_model.h5"
GESTURE_LABELS = {0: 'aeroplane', 1: 'brother', 2: 'father', 3: 'gun', 4: 'help', 
                  5: 'kidnap', 6: 'me', 7: 'mother', 8: 'sister', 9: 'water'}

# Load the model
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(image):
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            keypoints = []
            for hand_landmarks in results.multi_hand_landmarks:
                keypoints.extend([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            if len(keypoints) == 21:
                keypoints.extend([[0, 0, 0]] * 21)  # Pad if one hand detected
            return np.array(keypoints).flatten()
        return np.zeros(21 * 3 * 2)  # No hands detected

# Open webcam feed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the camera.")
else:
    print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from the camera.")
        break
    
    frame = cv2.flip(frame, 1)
    keypoints = extract_keypoints(frame).reshape(1, -1)
    prediction = model.predict(keypoints)
    
    gesture = GESTURE_LABELS[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    # Display gesture
    cv2.putText(frame, f"{gesture} ({confidence*100:.2f}%)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Draw hand landmarks
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow("Gesture Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
