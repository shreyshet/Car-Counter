

import numpy as np
import cv2, os, glob

from random import randint
import tensorflow as tf
import time

import matplotlib.pyplot as plt

import subprocess

from sort import Sort

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#md = "ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8\ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8\saved_model"
#md = "centernet_mobilenetv2fpn_512x512_coco17_kpts\centernet_mobilenetv2_fpn_kpts\saved_model"
md = "centernet_resnet50_v1_fpn_512x512_coco17_tpu-8\centernet_resnet50_v1_fpn_512x512_coco17_tpu-8\saved_model"
# "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8\ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8\saved_model"
# best - "centernet_resnet50_v1_fpn_512x512_coco17_tpu-8\centernet_resnet50_v1_fpn_512x512_coco17_tpu-8\saved_model"
# "ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8/saved_model"
model = tf.saved_model.load(md)

vid = "sample_vid_2.mp4"
url = 'http://192.168.4.35:4747/video'




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

cap = cv2.VideoCapture(url)

# so, convert them from float to integer.
x,y,fw,fh = get_cropdim(cap)
#w = 700
#h = 512

size = (fw, fh)

# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.
#result = cv2.VideoWriter('Test_out_rough.avi',
#                         cv2.VideoWriter_fourcc(*'MJPG'),
#                         30, size)

# cap = cv2.VideoCapture(0)

# Initialize SORT tracker
tracker = Sort(max_age=1, iou_threshold=.2)

# Initialize SORT tracker
trackerleft = Sort(max_age=5, iou_threshold=.2)

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

# skip frames
count = 0
# number of times the object crosses
crosscount = 0
# detected pos
curr_pos = 0
prev_pos = [0]
speed = 0

labels = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
          'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
          'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
          'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
          'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
          'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
          'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
          'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
          'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

class Vehicle:
    def __init__(self, name, age):
        self.name = name
        self.age = age

def preprocess_image(img_raw, x, y, w, h):
    #img = cv2.imread(image_path)
    if img_raw is None:
        raise ValueError("Image not found or unable to read the image")
        # crop_box should be (x, y, width, height)

    #img = cv2.resize(img_raw, (640, 640))
    #img = img / 255.0  # Normalize if required by the model

    #x, y, w, h = (x, 300, frame_width, frame_height)
    img = img_raw[y:y + h, x:x + w]
    return img

tplot = []
t = 0.0
speedplot = []
id_list = []
right_count = 0
left_count = 0
prevflag = 0

while True:
    # Skip frame
    count += 1
    if ((count % 1) != 0):
        continue
    # Read Image from camera: Break if no image is available
    ret, image_raw = cap.read()
    # print(ret)
    if ret is False:
        print("Done")
        break

    #Calculate run time processing speed:
    # time when we finish processing for previous frame
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)


    # Preprocess Image: Crop using the roi params
    image_np = preprocess_image(image_raw,x,y,fw,fh)

    # Convert Image as appropriate input tensor for model
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)

    # Detect objects using model
    detection = model(input_tensor)

    # Parse the detection results
    boxes = detection['detection_boxes'].numpy()
    classes = detection['detection_classes'].numpy().astype(int)
    if classes is None:
        print("no classes detected")
    scores = detection['detection_scores'].numpy()
    detections = []
    flag = 0
    for i in range(classes.shape[1]):
        class_id = int(classes[0, i])
        score = scores[0, i]
        if np.any(score > 0.5) and ((class_id == 3) or (class_id == 8)):
            # Filter out low-confidence detections and keep only CAR or TRUCK
            flag = 1
            h, w, _ = image_np.shape
            ymin, xmin, ymax, xmax = boxes[0, i]

            # Convert normalized coordinates to image coordinates
            xmin = int(xmin * w)
            xmax = int(xmax * w)
            ymin = int(ymin * h)
            ymax = int(ymax * h)

            # Get the class name from the labels list
            class_name = labels[class_id]
            random_color = (0, 255, 0)
            # EXAMPLE of 'detections' variable:
            # detections = [
            #    [100, 100, 150, 150, 0.9],  # Example detection format
            #    [200, 200, 250, 250, 0.8],  # Example detection format
            #    [300, 300, 350, 350, 0.7]  # Example detection format
            # ]
            detections.append([xmin,ymin,xmax,ymax,score])
            tracked_objects = tracker.update(np.array(detections))
            speed = [0]*len(tracked_objects)
            curr_pos = [0] * len(tracked_objects)

            for i in range(len(tracked_objects)):
                obj = tracked_objects[i]
                x1, y1, x2, y2, id = obj.astype(np.int32)
                curr_pos[i] = (x1 + x2) / 2
                if (x1 > 100) and (x2 < 1000):
                    speed[i] = ((curr_pos[i] - prev_pos[i]) / 71.426) * 30  # mps
                    if speed[i] > 0:
                        print('Moving Right')
                        if (x1+x2)/2 > w/2:
                            right_count = right_count + 1
                    elif speed[i] < 0:
                        print('Moving Left')
                        if (x1+x2)/2 < w/2:
                            left_count = left_count + 1
                prev_pos = curr_pos

                '''
                print(speed)
                #t = t + 1/30
                t = cap.get(cv2.CAP_PROP_POS_MSEC)
                print(t)
                tplot.append(t)
                speedplot.append(speed)
                '''
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Class: {class_name} , Score: {score:.2f}, ID: {id:.2f}"
                cv2.putText(image_np, label, (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, random_color, 2)

    if flag == 0 and prevflag == 1:
        tracker.update(np.empty((0,5)))

    prevflag = flag
    # Display output
    cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
    #result.write(image_np)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        print("was here")
        break


'''
plt.plot(tplot,np.multiply(speedplot,2.23694))
plt.grid()
plt.show()
'''
cap.release()
#result.release()
cv2.destroyAllWindows()