import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('hand_gesture_model.h5')

# Initialize mediapipe hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Start webcam capture
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Flip the frame for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the image to RGB (Mediapipe uses RGB)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image with Mediapipe to extract hand landmarks
    results = hands.process(image_rgb)

    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            keypoints = []

            # Extract the keypoints (21 landmarks * 3 coordinates)
            for lm in landmarks.landmark:
                keypoints.append([lm.x, lm.y, lm.z])

            keypoints = np.array(keypoints).flatten()  # Flatten to a 63-dimensional array
            keypoints = keypoints.reshape(1, 63, 1)  # Reshape for CNN input

            # Predict the digit using the trained model
            prediction = model.predict(keypoints)
            predicted_label = np.argmax(prediction)

            # Display the predicted label on the screen
            cv2.putText(frame, f'Predicted Label: {predicted_label}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw landmarks on the hand (optional)
            mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
