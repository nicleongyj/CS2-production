import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
from ultralytics import YOLO
import os
import time

def load_annotations(label_path, img_width, img_height, target_width=640, target_height=640):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    annotations = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        # Convert bbox format and normalize
        xmin = (x_center - width / 2) * img_width
        ymin = (y_center - height / 2) * img_height
        xmax = (x_center + width / 2) * img_width
        ymax = (y_center + height / 2) * img_height
        
        # Scale to target dimensions
        scale_x = target_width / img_width
        scale_y = target_height / img_height
        xmin *= scale_x
        ymin *= scale_y
        xmax *= scale_x
        ymax *= scale_y

        annotations.append([class_id, xmin, ymin, xmax, ymax])
    return annotations


def calculate_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2[1:]

    x_inter1 = max(x1, x3)
    y_inter1 = max(y1, y3)
    x_inter2 = min(x2, x4)
    y_inter2 = min(y2, y4)

    inter_area = max(0, x_inter2 - x_inter1) * max(0, y_inter2 - y_inter1)
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x4 - x3) * (y4 - y3)
    iou = inter_area / (bbox1_area + bbox2_area - inter_area) if bbox1_area + bbox2_area - inter_area != 0 else 0
    return iou

def visualize_detections(image_path, detections, annotations):
    image = Image.open(image_path)
    image = image.resize((640,640))
    draw = ImageDraw.Draw(image)
    
    # Draw ground truth annotations in red
    for ann in annotations:
        class_id, xmin, ymin, xmax, ymax = ann
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
    
    # Draw detections in green
    for detection in detections:
        xmin, ymin, xmax, ymax = detection
        draw.rectangle([xmin, ymin, xmax, ymax], outline="green", width=2)
    
    image.show()

def evaluate_detections(gt_annotations, detections, img_width, img_height, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    matched_gt = set()

    for detection in detections:
        max_iou = 0
        matched_gt_ann = None
        for gt_ann in gt_annotations:
            iou = calculate_iou(detection, gt_ann)
            if iou > max_iou:
                max_iou = iou
                matched_gt_ann = gt_ann

        if max_iou >= iou_threshold:
            true_positives += 1
            matched_gt.add(tuple(matched_gt_ann))
        else:
            false_positives += 1

    false_negatives = len(gt_annotations) - len(matched_gt)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

def test_yolo_model(model, image_dir, label_dir):
    precisions = []
    recalls = []
    f1_scores = []
    processing_times = []
    count = 0

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

    with torch.no_grad():
        for image_file in image_files:
            count += 1
            image_path = os.path.join(image_dir, image_file)
            label_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + '.txt')

            bnw_images = ['./datasets/images/open_images/images/83ad1edb154a2106.jpg', 
                          './datasets/images/open_images/images/ae27395fbd328ee1.jpg',
                          './datasets/images/open_images/images/eb99cee1f01b08ff.jpg',
                          './datasets/images/open_images/images/d0ffc5d3a209d1f0.jpg',
                          './datasets/images/open_images/images/209782692eed237b.jpg',
                          './datasets/images/open_images/images/9a18f1f319f9d79a.jpg',
                          './datasets/images/open_images/images/7f8ef92c55a63b32.jpg',
                          './datasets/images/open_images/images/e216dd92c7f1e961.jpg',
                          './datasets/images/open_images/images/bfd453a9345ee0fc.jpg',
                          './datasets/images/open_images/images/1a04c113301b09f8.jpg',
                          './datasets/images/open_images/images/153c8636fb302e54.jpg']
            
            print(image_path)
            # if image_path in bnw_images:
            #     continue

            if not os.path.exists(label_path):
                continue

            image = Image.open(image_path)
            img_width, img_height = image.size

            gt_annotations = load_annotations(label_path, img_height=img_height, img_width=img_width)


            transform = transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
            ])
            image_resized = transform(image).unsqueeze(0)

            time_start = time.time()
            try:
                results = model(image_resized)
                processing_time = time.time() - time_start

                detections = []
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    for box in boxes:
                        detections.append(box.tolist())

                precision, recall, f1_score = evaluate_detections(gt_annotations, detections, img_width, img_height)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1_score)
                processing_times.append(processing_time)

                print(f"Image: {image_file} - Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")
                # visualize_detections(image_path, detections, gt_annotations)
            except ValueError as e:
                if "set tensor" in str(e):
                    print(f"Skipping {image_path} due to black and white image.")
                    continue
                else:
                    raise e

    mean_precision = np.mean(precisions) if precisions else 0
    mean_recall = np.mean(recalls) if recalls else 0
    mean_f1_score = np.mean(f1_scores) if f1_scores else 0
    mean_processing_time = np.mean(processing_times) if processing_times else 0
    print(f"Total images processed: {count}")
    print(f"Mean Precision: {mean_precision}, Mean Recall: {mean_recall}, Mean F1 Score: {mean_f1_score}, Mean Processing Time: {mean_processing_time}")

# Update these paths with the paths to your own dataset
image_dir = './datasets/images/open_images/images'
label_dir = './datasets/images/open_images/labels'

model = YOLO('saved_model/yolov8_hdb_float32.tflite')
test_yolo_model(model, image_dir, label_dir)


############################## Results ###############################
# Precision: Accuracy of positive predicitions
# Recall: Ability to correctly identify positive instances
# F1 score: Overall performance
#
# Sampled dataset
# NANO   : Mean Precision: 0.468, Mean Recall: 0.731 Mean F1 Score: 0.525 Mean Processing Time: 0.0566
# HDB_PT : Mean Precision: 0.653, Mean Recall: 0.630, Mean F1 Score: 0.602, Mean Processing Time: 0.0657
# TFlite : Mean Precision: 0.653, Mean Recall: 0.630, Mean F1 Score: 0.602, Mean Processing Time: 0.132
######################################################################