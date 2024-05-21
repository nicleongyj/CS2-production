from pycocotools.coco import COCO
import json

# Class names to filter
class_names = ["bicycle", "chair", "box", "table", "plastic bag", "flowerpot", 
               "luggage and bags", "umbrella", "shopping trolley", "person", "luggage", 'bags','person','suitcase','couch']

# Map class names to COCO category IDs
def get_category_ids(coco, class_names):
    category_ids = []
    for category in coco.loadCats(coco.getCatIds()):
        if category['name'] in class_names:
            category_ids.append(category['id'])
    return category_ids

# # # Load COCO annotations
coco = COCO('COCO/annotations/instances_val2017.json')

# # Get category IDs for the specified classes
category_ids = get_category_ids(coco, class_names)
category_ids = [46]

image_ids = set()
for cat_id in category_ids:
    image_ids.update(coco.getImgIds(catIds=[46]))

image_ids = list(image_ids)
print("Total number of test images:", len(image_ids))

# Filter annotations
filtered_annotations = {
    'images': [],
    'annotations': [],
    'categories': [cat for cat in coco.loadCats(coco.getCatIds()) if cat['id'] in category_ids]
}
filtered_annotations = {
    'images': [],
    'annotations': [],
    'categories': [cat for cat in coco.loadCats(coco.getCatIds()) if cat['id'] in category_ids]
}

# Add filtered images
for img_id in image_ids:
    image_info = coco.loadImgs(img_id)[0]
    filtered_annotations['images'].append(image_info)

# Add filtered annotations
for ann in coco.loadAnns(coco.getAnnIds(imgIds=image_ids, catIds=category_ids)):
    if ann['category_id'] in category_ids:
        filtered_annotations['annotations'].append(ann)

filtered_annotations_path = "COCO/annotations/filtered_annotations.json"
with open(filtered_annotations_path, 'w') as f:
    json.dump(filtered_annotations, f)




import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os
import json
from PIL import Image, ImageDraw

# Load filtered annotations
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
        transforms.Resize((640, 640)),  # Resize the image to (640, 640)
        transforms.ToTensor(),  # Convert PIL image to torch tensor
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Calculate IoU (Intersection over Union)
def calculate_iou(bbox1, bbox2):
    # Extract coordinates from bounding boxes
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2

    # Calculate intersection coordinates
    x1 = max(xmin1, xmin2)
    y1 = max(ymin1, ymin2)
    x2 = min(xmax1, xmax2)
    y2 = min(ymax1, ymax2)

    # Calculate intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union area
    box1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
    box2_area = (xmax2 - xmin2) * (ymax2 - ymin2)
    union_area = box1_area + box2_area - intersection_area
    # Calculate IoU
    iou = intersection_area / union_area if union_area != 0 else 0

    return iou

# Evaluate detected objects against ground truth annotations
def evaluate_detections(annotations, detections, image_id, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    gt_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]
    matched_gt = set()

    for ann in gt_annotations:
        print(ann['bbox'])
    print("\n")
    for detection in detections:
        print(detection)

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

    return precision, recall

def visualize(image_path, detections, annotations):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Draw detections in green
    for detection in detections:
        draw.rectangle(detection[:4], outline="green")
    
    # Draw annotations in red
    for ann in annotations:
        bbox = ann['bbox']
        draw.rectangle(bbox, outline="red")
    
    image.show()


# Main function to test the PyTorch YOLO model
def test_yolo_model(model, annotations_path, image_dir):
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    precisions = []
    recalls = []    

    with torch.no_grad():
        for image_info in annotations['images']:
            print(image_info['id'])
            image_path = os.path.join(image_dir, image_info['file_name'])
            results = model(image_path)

            # Extract bounding boxes from results
            detections = []
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # Convert to numpy array
                for box in boxes:
                    detections.append(box.tolist())  # Convert each box to a list
            precision, recall = evaluate_detections(annotations, detections, image_info['id'])
            precisions.append(precision)
            recalls.append(recall)

            print(f"Image ID: {image_info['id']} - Precision: {precision}, Recall: {recall}")
            visualize(image_path, detections, [ann for ann in annotations['annotations'] if ann['image_id'] == image_info['id']])
            break

    # Calculate mean precision and recall
    mean_precision = np.mean(precisions) if precisions else 0
    mean_recall = np.mean(recalls) if recalls else 0
    print(f"Mean Precision: {mean_precision}, Mean Recall: {mean_recall}")

annotations_path = 'COCO/annotations/instances_val2017.json'
image_dir = 'datasets/images/val/val2017'
model_path = "yolov8_hdb.pt"
model = YOLO(model_path)
test_yolo_model(model, annotations_path, image_dir)
