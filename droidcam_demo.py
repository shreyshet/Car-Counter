import cv2
import matplotlib.pyplot as plt
import numpy as np

cap = cv2.VideoCapture('http://192.168.4.35:4747/video')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('gray', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
