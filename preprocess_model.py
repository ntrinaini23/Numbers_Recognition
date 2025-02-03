import os
import cv2
import numpy as np
import mediapipe as mp

# Paths
DATA_DIR = "gesture_videos"
PROCESSED_DATA_DIR = "processed_data"
if not os.path.exists(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)

# Gesture labels
GESTURE_LABELS = {'aeroplane': 0, 'brother': 1, 'father': 2, 'gun': 3, 'help': 4, 
                  'kidnap': 5, 'me': 6, 'mother': 7, 'sister': 8, 'water': 9}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []
    
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process the frame with MediaPipe
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Extract keypoints
            if results.multi_hand_landmarks:
                keypoints = []
                for hand_landmarks in results.multi_hand_landmarks:
                    keypoints.extend([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                if len(keypoints) == 21:
                    keypoints.extend([[0, 0, 0]] * 21)  # Pad if one hand detected
                keypoints_list.append(np.array(keypoints).flatten())
            else:
                keypoints_list.append(np.zeros(21 * 3 * 2))  # No hands detected
            
    cap.release()
    return keypoints_list

# Process videos
keypoints, labels = [], []
for gesture, label in GESTURE_LABELS.items():
    folder_path = os.path.join(DATA_DIR, gesture)
    if not os.path.exists(folder_path):
        continue
    
    for video_name in os.listdir(folder_path):
        video_path = os.path.join(folder_path, video_name)
        video_keypoints = extract_keypoints_from_video(video_path)
        keypoints.extend(video_keypoints)
        labels.extend([label] * len(video_keypoints))

# Save processed data
np.save(os.path.join(PROCESSED_DATA_DIR, "keypoints.npy"), keypoints)
np.save(os.path.join(PROCESSED_DATA_DIR, "labels.npy"), labels)
print(f"Preprocessing complete. Saved {len(keypoints)} keypoints and {len(labels)} labels.")
