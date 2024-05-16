import cv2
from ultralytics import YOLO

# model = YOLO('./yolov8_hdb_saved_model/yolov8_hdb_float32.tflite')
model = YOLO('yolov8_hdb.pt')
source = './assets/table2.jpg'
results = model(source) 

for result in results:
    boxes = result.boxes  
    masks = result.masks  
    keypoints = result.keypoints
    probs = result.probs  
    obb = result.obb  
    result.show()  
