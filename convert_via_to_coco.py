import json
import os
import shutil
import random
from pathlib import Path

# Define paths
VIA_JSON_FILE = "/Users/anugunj/Downloads/AI Datasets/AN_workSpace/TrainDataset_US24_640/train/US_24.json"  # Change this to the actual path
IMAGE_FOLDER = "/Users/anugunj/Downloads/AI Datasets/AN_workSpace/TrainDataset_US24_640/train"   # Folder containing the images
OUTPUT_FOLDER = "data"


def create_output_folders():
    """Creates necessary folders for COCO dataset."""
    os.makedirs(f"{OUTPUT_FOLDER}/train", exist_ok=True)
    os.makedirs(f"{OUTPUT_FOLDER}/val", exist_ok=True)
    os.makedirs(f"{OUTPUT_FOLDER}/annotations", exist_ok=True)


def load_via_json(via_json_path):
    """Loads VIA annotation JSON file."""
    with open(via_json_path, "r") as f:
        return json.load(f)


def get_category_mapping(via_categories):
    """Creates category mapping from VIA categories."""
    return {k: i + 1 for i, (k, v) in enumerate(via_categories.items())}


def initialize_coco_structure(category_mapping):
    """Initializes COCO data structure."""
    categories = [{"id": v, "name": k, "supercategory": "interface"} for k, v in category_mapping.items()]
    return {"train": {"images": [], "annotations": [], "categories": categories},
            "val": {"images": [], "annotations": [], "categories": categories}}


def split_data(image_list, train_ratio=0.8):
    """Splits the image list into training and validation sets."""
    random.shuffle(image_list)
    train_split = int(train_ratio * len(image_list))
    return image_list[:train_split], image_list[train_split:]


def process_images(via_img_metadata, category_mapping, train_images, val_images):
    """Processes images and annotations and organizes them into COCO format."""
    coco_data = initialize_coco_structure(category_mapping)
    annotation_id = 1

    for idx, image_id in enumerate(via_img_metadata.keys()):
        img_metadata = via_img_metadata[image_id]
        filename = img_metadata["filename"]
        img_path = os.path.join(IMAGE_FOLDER, filename)

        if not os.path.exists(img_path):
            print(f"Warning: Image {filename} not found. Skipping.")
            continue

        subset = "train" if image_id in train_images else "val"

        coco_image = {"id": idx + 1, "file_name": filename, "width": 640, "height": 640}
        coco_data[subset]["images"].append(coco_image)

        shutil.copy(img_path, f"{OUTPUT_FOLDER}/{subset}/{filename}")

        for region in img_metadata["regions"]:
            shape = region["shape_attributes"]
            category = region["region_attributes"]["type"]
            category_id = category_mapping.get(category, 0)

            if shape["name"] == "rect":
                x, y, w, h = shape["x"], shape["y"], shape["width"], shape["height"]
                coco_annotation = {
                    "id": annotation_id,
                    "image_id": coco_image["id"],
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                }
                coco_data[subset]["annotations"].append(coco_annotation)
                annotation_id += 1

    return coco_data


def save_coco_annotations(coco_data):
    """Saves COCO-formatted annotations to JSON files."""
    with open(f"{OUTPUT_FOLDER}/annotations/train.json", "w") as f:
        json.dump(coco_data["train"], f, indent=4)

    with open(f"{OUTPUT_FOLDER}/annotations/val.json", "w") as f:
        json.dump(coco_data["val"], f, indent=4)


def main():
    """Main function to convert VIA annotations to COCO format."""
    create_output_folders()

    via_data = load_via_json(VIA_JSON_FILE)
    via_img_metadata = via_data["_via_img_metadata"]
    via_categories = via_data["_via_attributes"]["region"]["type"]["options"]

    category_mapping = get_category_mapping(via_categories)
    print("Category Mapping:", category_mapping)

    image_list = list(via_img_metadata.keys())
    train_images, val_images = split_data(image_list)

    coco_data = process_images(via_img_metadata, category_mapping, train_images, val_images)

    save_coco_annotations(coco_data)

    print("COCO conversion completed successfully!")


if __name__ == "__main__":
    main()
