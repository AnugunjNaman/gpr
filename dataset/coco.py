import os
import torchvision
import torchvision.transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_folder, processor, transforms, train=True):
        """
        Custom COCO dataset class for object detection.

        Args:
            img_folder (str): Path to the folder containing images.
            ann_folder (str): Path to the folder containing annotations.
            processor (callable): Function to process images and annotations.
            transforms (callable): Image and target transformations.
            train (bool): Whether to use the training or validation split.
        """
        ann_file = os.path.join(ann_folder, "train.json" if train else "val.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.processor = processor
        self._transforms = transforms

    def __getitem__(self, idx):
        """
        Loads an image and its annotations, applies transformations, and formats for DETR.

        Args:
            idx (int): Index of the image.

        Returns:
            tuple: Processed image tensor and target annotations.
        """
        # Load image and target annotations
        img, target = super(CocoDetection, self).__getitem__(idx)

        # Ensure target annotations are properly formatted
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}

        if self._transforms:
            img, target = self._transforms(img, target)

        # Convert to the format required by DETR
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")

        # Remove batch dimension
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]  # Remove batch dimension

        return pixel_values, target


def make_coco_transforms(image_set):
    """
    Defines image transformations for training and validation.
    
    Args:
        image_set (str): Either "train" or "val".

    Returns:
        callable: A function that applies the necessary transformations to both image and target.
    """
    to_tensor = T.ToTensor()  # Convert image to tensor but do NOT normalize

    def transform_function(img, target):
        """Applies transformations to both image and annotations."""
        img = to_tensor(img)  # Convert to tensor, keeping values in [0,1]
        return img, target  # Keep target unchanged

    if image_set in ["train", "val"]:
        return transform_function  # Return a function instead of Compose()

    raise ValueError(f"Unknown dataset split: {image_set}")

