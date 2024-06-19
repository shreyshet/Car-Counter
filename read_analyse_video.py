import cv2
import time
import numpy as np

cap = cv2.VideoCapture('http://192.168.4.39:4747/video')#'sample_vid_2.mp4')
new_frame_time  = 0
prev_frame_time = 0

def get_cropdim(cap):
    ## CROP Image
    ret, image = cap.read()
    # Select ROI
    # set window flags
    cv2.namedWindow("select the area", cv2.WINDOW_NORMAL)
    r = cv2.selectROI("select the area", image)
    x = r[0]
    y = r[1]
    w = r[2]
    h = r[3]

    # Crop image
    cropped_image = image[int(r[1]):int(r[1] + r[3]),
                    int(r[0]):int(r[0] + r[2])]
    # Display cropped image
    cv2.imshow("Cropped image", cropped_image)
    cv2.waitKey(0)
    return x,y,w,h
##

def preprocess_image(img_raw,x = 0,y = 0, w = 1092,h = 1080):
    if img_raw is None:
        raise ValueError("Image not found or unable to read the image")
    img = img_raw[y:y + h, x:x + w]
    return img

x,y,w,h = get_cropdim(cap)

while True:
    ret, frame = cap.read()
    cropframe = preprocess_image(frame,x,y,w,h)
    gray = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

cap.release()
cv2.destroyAllWindows()
