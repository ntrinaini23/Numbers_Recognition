import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# --- STEP 1: Extract Keypoints from Videos ---
def extract_keypoints_from_videos(video_folder):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    keypoints = []
    labels = []
    video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]

    for idx, video_file in enumerate(video_files):
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)

        print(f"Processing video: {video_file}")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB for MediaPipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Flatten the hand landmarks into a single feature vector
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                    keypoints.append(landmarks)
                    labels.append(idx)  # Use video index as label (e.g., 0, 1, 2, ...)

        cap.release()

    hands.close()
    return np.array(keypoints), np.array(labels)

# Path to your folder containing videos
video_folder_path = "signs"
data, labels = extract_keypoints_from_videos(video_folder_path)

print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

# --- STEP 2: Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# --- STEP 3: Train a Random Forest Classifier ---
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print("Model training completed!")

# --- STEP 4: Evaluate the Model ---
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# --- STEP 5: Save the Model for Later Use ---
import joblib
model_path = "gesture_model.pkl"
joblib.dump(clf, model_path)
print(f"Model saved to {model_path}")
