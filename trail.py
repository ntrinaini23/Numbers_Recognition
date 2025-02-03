import cv2
print(cv2.__version__)
cap = cv2.VideoCapture(1)
print(cap.isOpened())
