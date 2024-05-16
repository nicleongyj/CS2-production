from ultralytics import YOLO
import cv2
import math

# Start webcam
# stream_url = "rtsp://cs2projs:cs2projs@192.168.0.166/stream2"
stream_url = "rtsp://CS2Lab:CS2Lab@192.168.0.182:554/stream1"
cap = cv2.VideoCapture(stream_url)
cap.set(3, 640)
cap.set(4, 640)

# Load your retrained YOLOv8 model
model = YOLO("yolov8_hdb.pt")
# model = YOLO("yolov8n.pt")

# Custom class names
classNames = ["Bicycle", "Chair", "Box", "Table", "Plastic bag", "Flowerpot", 
              "Luggage and bags", "Umbrella", "Shopping trolley", "Person"]

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # Process detection results
    for r in results:
        print(r.boxes.xyxy)
            # Extract bounding box coordinates
            # x1, y1, x2, y2 = [int(coord) for coord in box[:4]]

            # # Draw bounding box
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # # Confidence
            # confidence = round(float(box[4]), 2)
            # print("Confidence:", confidence)

            # # Class   
            # class_index = int(box[5])
            # class_name = classNames[class_index]
            # print("Class:", class_name)

            # # Display class name
            # org = (x1, y1)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # fontScale = 1
            # color = (255, 0, 0)
            # thickness = 2
            # cv2.putText(img, class_name, org, font, fontScale, color, thickness)

    # Display webcam feed
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
