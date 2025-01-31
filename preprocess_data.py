import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_images(dataset_path, img_size=(224, 224)):
    data = []
    labels = []
    for label in range(10):  # 0-9 for sign language digits
        folder_path = os.path.join(dataset_path, str(label))
        print(f"Checking folder: {folder_path}")  # Debugging paths
        if not os.path.exists(folder_path):
            print(f"Error: Folder {folder_path} not found!")
            continue

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            if os.path.isfile(img_path):  # Ensure it's a file
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Error reading image: {img_path}")
                    continue
                img = cv2.resize(img, img_size)
                img = img / 255.0  # Normalize the image
                data.append(img)
                labels.append(label)

    # Convert to numpy arrays
    data = np.array(data).reshape(-1, img_size[0], img_size[1], 1)
    labels = np.array(labels)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
