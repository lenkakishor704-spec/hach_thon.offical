# import cv2
# import numpy as np


# # Load video file
# video = cv2.VideoCapture("kishor.mp4")

# CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
# 	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
# 	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
# 	"sofa", "train", "tvmonitor"]

# net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt','MobileNetSSD_deploy.caffemodel')

# if not video.isOpened():
#     print("❌ Could not open video file")
#     exit()

# while True:
#     ret, frame = video.read()
#     if not ret:  # End of video or error
#         print("⚠️ No more frames or failed to read video.")
#         break

#     # Correct way: pass (width, height) as a tuple
#     frame = cv2.resize(frame, (640, 480))
#     (h,w) = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), 0.007843,(300,300),127.5)
#     net.setInput(blob)
#     detection = net.forward()
#     for i in np.arange(0, detection.shape[2]):
#      confident = [0, 0,i,2]
#      if confident >0.5:
#       id = detection[0,0,i,1]
#       box = detection[0,0,i,3:7] * np.array([w,h,w,h])
#       (startx, starty, endx, endy) = box.astype("int")
#       cv2.rectangle(frame,(startx,starty),(endx,endy),(0,255,0))
#       print(detection)

#     cv2.imshow("Frame", frame)

#     # Press 'q' to quit
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break

# video.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
import random


R = random.randint(0,255)
G = random.randint(0,255)
B = random.randint(0,255)

# Load video file
video = cv2.VideoCapture("cat.mp4")

# Class labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

color = [(R, G, B) for i in CLASSES]

# Load model
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt','MobileNetSSD_deploy.caffemodel')

if not video.isOpened():
    print("❌ Could not open video file")
    exit()

while True:
    ret, frame = video.read()
    if not ret:  # End of video or error
        print("⚠️ No more frames or failed to read video.")
        break

    frame = cv2.resize(frame, (640, 480))
    (h, w) = frame.shape[:2]

    # Prepare blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), 0.007843,(300,300),127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]   # ✅ Correct way

        if confidence > 0.5:  # Confidence threshold
            idx = int(detections[0, 0, i, 1])  # Class ID
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            cv2.rectangle(frame, (startX-1, startY-40),(endX+1, startY-3), (0,255,0),4)

            # Draw bounding box
            label = f"{CLASSES[idx]}: {confidence:.2f}"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 2)
            cv2.putText(frame, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imshow("Frame", frame)

    # Press 'q' to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import random
# import os

# # === Check files exist ===
# for f in ["yolov3.cfg", "yolov3.weights", "coco.names", "birds.mp4"]:
#     if not os.path.exists(f):
#         print(f"❌ Missing file: {f}")
#         exit()

# # Load YOLO model (cfg first, then weights)
# net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")

# # Load COCO class labels (80 classes)
# with open("coco.names", "r") as f:
#     CLASSES = [line.strip() for line in f.readlines()]

# # Random colors for each class
# colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# # Set backend and target to OpenCV (CPU)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# # Load video
# video = cv2.VideoCapture("birds.mp4")
# if not video.isOpened():
#     print("❌ Could not open video file")
#     exit()

# # Get YOLO output layer names
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# while True:
#     ret, frame = video.read()
#     if not ret:
#         print("⚠️ No more frames or failed to read video.")
#         break

#     frame = cv2.resize(frame, (640, 480))
#     height, width = frame.shape[:2]

#     # Create blob
#     blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
#     net.setInput(blob)
#     outputs = net.forward(output_layers)

#     boxes, confidences, class_ids = [], [], []

#     # Process YOLO outputs
#     for output in outputs:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]

#             if confidence > 0.5:  # confidence threshold
#                 center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)

#                 boxes.append([x, y, int(w), int(h)])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     # Apply Non-Max Suppression to remove duplicates
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

#     if len(indexes) > 0:
#         for i in indexes.flatten():
#             x, y, w, h = boxes[i]
#             color = colors[class_ids[i]]
#             label = f"{CLASSES[class_ids[i]]}: {confidences[i]:.2f}"

#             cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#             cv2.putText(frame, label, (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     cv2.imshow("YOLO Detection", frame)

#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break

# video.release()
# cv2.destroyAllWindows()
