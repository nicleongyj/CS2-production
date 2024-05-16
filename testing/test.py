import tensorflow as tf

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./saved_model/yolov8n_float32.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

from PIL import Image
import numpy as np

# Load an image from file
image = Image.open("./assets/umbrella.jpg")

# Resize the image to match the input shape expected by the model
input_shape = input_details[0]['shape']
image = image.resize((input_shape[1], input_shape[2]))

# Convert the image to a numpy array and normalize it
input_data = np.expand_dims(image, axis=0)
input_data = np.float32(input_data) / 255.0  # normalize to [0,1] if the model requires it

interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

output_data = np.squeeze(output_data)  # Remove the batch dimension, resulting in shape (14, 8400)

# Split the output into bounding boxes, confidence scores, and class scores
num_classes = 10  # Change this based on your specific model's number of classes

boxes = output_data[:4, :]  # Shape (4, 8400)
confidences = output_data[4, :]  # Shape (8400,)
class_probs = output_data[5:5+num_classes, :]  # Shape (9, 8400)


confidence_threshold = 0.5  # Adjust based on your needs
detections = []

for i in range(8400):
    if confidences[i] > confidence_threshold:
        box = boxes[:, i]
        class_id = np.argmax(class_probs[:, i])
        class_prob = class_probs[class_id, i]
        
        detection = {
            'box': box,
            'confidence': confidences[i],
            'class_id': class_id,
            'class_prob': class_prob
        }
        detections.append(detection)

# Optional: Apply non-max suppression (NMS) if needed
def non_max_suppression(detections, iou_threshold=0.5):
    # Implementation of NMS (this is a simplified example, many libraries offer optimized NMS functions)
    boxes = np.array([det['box'] for det in detections])
    confidences = np.array([det['confidence'] for det in detections])
    indices = np.argsort(-confidences)
    
    keep = []
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        remaining = indices[1:]
        
        ious = compute_iou(boxes[current], boxes[remaining])
        indices = remaining[ious < iou_threshold]
    
    return [detections[i] for i in keep]

def compute_iou(box1, boxes):
    # Compute Intersection over Union (IoU) between box1 and an array of boxes
    # This is a simplified version for illustrative purposes
    x1 = np.maximum(box1[0], boxes[:, 0])
    y1 = np.maximum(box1[1], boxes[:, 1])
    x2 = np.minimum(box1[2], boxes[:, 2])
    y2 = np.minimum(box1[3], boxes[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - intersection
    
    iou = intersection / union
    return iou

# Apply NMS
final_detections = non_max_suppression(detections)

# Print final detections
for det in final_detections:
    print(f"Class ID: {det['class_id']}, Confidence: {det['confidence']}, Box: {det['box']}")
