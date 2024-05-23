import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os
import json
from PIL import Image, ImageDraw
import time

def load_annotations(annotations_path):
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    return annotations

def predict_image(model, image_path):
    image = load_and_preprocess_image(image_path)
    with torch.no_grad():
        detections = model(image)
    
    return detections

def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((640, 640)), 
        transforms.ToTensor(), 
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def calculate_iou(bbox1, bbox2):
    # Convert annotations from xywh to xyxy format
    x, y, w, h = bbox2
    bbox2_xyxy = [x, y, x + w, y + h]
    
    # Extract coordinates from bounding boxes
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2_xyxy

    # Intersection coordinates
    x1 = max(xmin1, xmin2)
    y1 = max(ymin1, ymin2)
    x2 = min(xmax1, xmax2)
    y2 = min(ymax1, ymax2)

    # Intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Union area
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
    
    # Detections in green
    for detection in detections:
        draw.rectangle(detection[:4], outline="green")
    
    # Annotations in red
    for ann in annotations:
        bbox = ann['bbox']
        x, y, w, h = bbox
        x1, y1, x2, y2 = x, y, x + w, y + h
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.rectangle(bbox, outline="red")
    
    image.show()


def test_yolo_model(model, annotations_path, image_dir):
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    precisions = []
    recalls = []    
    f1_scores = []
    processing_times = []
    count = 0 

    with torch.no_grad():
        for image_info in annotations['images']:
            count+=1
            image_path = os.path.join(image_dir, image_info['file_name'])
            
            time_start = time.time()
            results = model(image_path)
            processing_time = time.time() - time_start
            
            detections = []
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  
                for box in boxes:
                    detections.append(box.tolist())  

            print(detections)
            precision, recall, f1_score = evaluate_detections(annotations, detections, image_info['id'])
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)
            processing_times.append(processing_time)

            print(f"Image ID: {image_info['id']} - Precision: {precision}, Recall: {recall}", f"F1 Score: {f1_score}")
            break
            # visualize(image_path, detections, [ann for ann in annotations['annotations'] if ann['image_id'] == image_info['id']])

    mean_precision = np.mean(precisions) if precisions else 0
    mean_recall = np.mean(recalls) if recalls else 0
    mean_f1_score = np.mean(f1_scores) if f1_scores else 0
    mean_processing_time = np.mean(processing_times) if processing_times else 0
    print(f"Total images processed: {count}")
    print(f"Mean Precision: {mean_precision}, Mean Recall: {mean_recall}", f"Mean F1 Score: {mean_f1_score}", f"Mean Processing Time: {mean_processing_time}")



annotations_path = 'COCO/annotations/instances_val2017.json'
image_dir = 'datasets/images/val/val2017'

filtered_annotations_path = "COCO/annotations/sampled_annotations.json"
filtered_image_dir = "datasets/images/val/filtered_val2017"

nano_model_path = "yolov8n.pt"
pt_model_path = "yolov8_hdb.pt"
tflite_model_path = "saved_model/yolov8_hdb_float32.tflite"

model = YOLO(tflite_model_path)
# test_yolo_model(model, annotations_path, image_dir)
test_yolo_model(model, filtered_annotations_path, filtered_image_dir)




############################## Results ###############################
# Precision: Accuracy of positive predicitions
# Recall: Ability to correctly identify positive instances
# F1 score: Overall performance
#
# Complete dataset
# HDB_PT : Mean Precision: 0.403, Mean Recall: 0.152 Mean F1 Score: 0.200 Mean Processing Time: 0.057
# 
# Sampled dataset
# NANO   : Mean Precision: 0.468, Mean Recall: 0.731 Mean F1 Score: 0.525 Mean Processing Time: 0.0566
# HDB_PT : Mean Precision: 0.361, Mean Recall: 0.256 Mean F1 Score: 0.267 Mean Processing Time: 0.0527
# TFlite : Mean Precision: 0.351, Mean Recall: 0.257 Mean F1 Score: 0.264 Mean Processing Time: 0.139
######################################################################