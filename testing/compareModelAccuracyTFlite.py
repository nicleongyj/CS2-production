import tensorflow as tf
import cv2
import numpy as np
from PIL import Image, ImageDraw
import json
import os
import time

def load_annotations(annotations_path):
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    return annotations

def preprocess_image(image_path, input_size):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(input_size)
    image_np = np.array(image, dtype=np.float32)
    image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension
    return image_np

def load_and_preprocess_image(image_path, input_details):
    input_shape = input_details[0]['shape'][1:3]
    return preprocess_image(image_path, input_shape)

def run_inference(interpreter, image_path):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image = load_and_preprocess_image(image_path, input_details)
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    
    output_data = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    return output_data

def calculate_iou(bbox1, bbox2):
    x, y, w, h = bbox2
    bbox2_xyxy = [x, y, x + w, y + h]
    
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2_xyxy

    x1 = max(xmin1, xmin2)
    y1 = max(ymin1, ymin2)
    x2 = min(xmax1, xmax2)
    y2 = min(ymax1, ymax2)

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
    box2_area = (xmax2 - xmin2) * (ymax2 - ymin2)
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area if union_area != 0 else 0
    return iou

def evaluate_detections(annotations, detections, image_id, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    gt_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]
    matched_gt = set()

    for detection in detections:
        max_iou = 0
        matched_gt_ann = None
        for gt_ann in gt_annotations:
            iou = calculate_iou(detection[:4], gt_ann['bbox'])
            if iou > max_iou:
                max_iou = iou
                matched_gt_ann = gt_ann

        if max_iou >= iou_threshold:
            true_positives += 1
            matched_gt.add(matched_gt_ann['id'])
        else:
            false_positives += 1

    false_negatives = len(gt_annotations) - len(matched_gt)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

def visualize(image_path, detections, annotations):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    for detection in detections:
        draw.rectangle(detection[:4], outline="green")
    
    for ann in annotations:
        bbox = ann['bbox']
        x, y, w, h = bbox
        x1, y1, x2, y2 = x, y, x + w, y + h
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    
    image.show()

def test_tflite_model(model_path, annotations_path, image_dir):
    annotations = load_annotations(annotations_path)
    precisions = []
    recalls = []
    f1_scores = []
    processing_times = []
    count = 0

    interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=4)
    interpreter.allocate_tensors()

    for image_info in annotations['images']:
        count += 1
        image_path = os.path.join(image_dir, image_info['file_name'])

        time_start = time.time()
        output_data = run_inference(interpreter, image_path)
        print(output_data[0].shape)
        processing_time = time.time() - time_start

        detections = []
        for i in range(output_data[0].shape[1]):
            bbox = output_data[0][0, i, :4]  # Adjust index based on model output
            confidence = output_data[0][0, i, 4]
            class_id = output_data[0][0, i, 5]
            if confidence > 0.5:  # Confidence threshold
                detections.append(list(bbox) + [class_id, confidence])

        precision, recall, f1_score = evaluate_detections(annotations, detections, image_info['id'])
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
        processing_times.append(processing_time)

        print(f"Image ID: {image_info['id']} - Precision: {precision}, Recall: {recall}", f"F1 Score: {f1_score}")
        # visualize(image_path, detections, [ann for ann in annotations['annotations'] if ann['image_id'] == image_info['id']])
        break

    mean_precision = np.mean(precisions) if precisions else 0
    mean_recall = np.mean(recalls) if recalls else 0
    mean_f1_score = np.mean(f1_scores) if f1_scores else 0
    mean_processing_time = np.mean(processing_times) if processing_times else 0

    print(f"Total images processed: {count}")
    print(f"Mean Precision: {mean_precision}, Mean Recall: {mean_recall}", f"Mean F1 Score: {mean_f1_score}", f"Mean Processing Time: {mean_processing_time}")

# Paths to your annotations and image directory
annotations_path = 'COCO/annotations/instances_val2017.json'
image_dir = 'datasets/images/val/val2017'

filtered_annotations_path = "COCO/annotations/sampled_annotations.json"
filtered_image_dir = "datasets/images/val/filtered_val2017"

tflite_model_path = "saved_model/yolov8_hdb_float32.tflite"

# Run the TFLite model test
test_tflite_model(tflite_model_path, filtered_annotations_path, filtered_image_dir)

