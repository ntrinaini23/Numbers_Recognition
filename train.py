import mediapipe as mp
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten

# Initialize mediapipe hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Dataset path
dataset_path = 'dataset'  # Ensure that the dataset is in the same directory as this script

# Prepare data (X = features, y = labels)
X = []
y = []

# Loop through each folder (0-9)
for label in range(10):
    folder_path = os.path.join(dataset_path, str(label))
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        # Load the image
        image = cv2.imread(file_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image with Mediapipe to extract landmarks
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                keypoints = []
                for lm in landmarks.landmark:
                    keypoints.append([lm.x, lm.y, lm.z])
                keypoints = np.array(keypoints).flatten()  # Flatten to 63 values (21 points * 3 coordinates)
                
                # Append the keypoints and label to the dataset
                X.append(keypoints)
                y.append(label)

# Convert to numpy arrays for easy handling
X = np.array(X)
y = np.array(y)

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data to be in the correct format for CNN (num_samples, 63, 1)
X_train = X_train.reshape(-1, 63, 1)
X_test = X_test.reshape(-1, 63, 1)

# Build the CNN model
model = Sequential()
model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(63, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))  # 10 output units for digits 0-9

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save('hand_gesture_model.h5')
print("Model trained and saved as 'hand_gesture_model.h5'")
