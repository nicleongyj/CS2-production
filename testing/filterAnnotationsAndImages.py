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
