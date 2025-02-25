import os

import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ObjectDetectionDataset(Dataset):
    def __init__(self, image_folder, annotation_folder, transform=None):
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load corresponding annotation
        annotation_filename = image_filename.replace(".jpg", ".txt")
        annotation_path = os.path.join(self.annotation_folder, annotation_filename)

        boxes = []
        labels = []

        if os.path.exists(annotation_path):
            with open(annotation_path, "r") as f:
                for line in f.readlines():
                    data = line.strip().split()
                    class_id = int(data[0])
                    x_center, y_center, width, height = map(float, data[1:])

                    # Convert from normalized to absolute pixel values
                    x_center *= image.shape[1]
                    y_center *= image.shape[0]
                    width *= image.shape[1]
                    height *= image.shape[0]

                    x_min = x_center - width / 2
                    y_min = y_center - height / 2
                    x_max = x_center + width / 2
                    y_max = y_center + height / 2

                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id)

        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # Apply image transformations (if any)
        if self.transform:
            image = self.transform(image)

        return image, {"boxes": boxes, "labels": labels}


def collate_fn(batch):
    images, targets = zip(*batch)  # Unpack batch
    images = torch.stack(
        images, dim=0
    )  # Stack images into a single tensor (B, C, H, W)
    return images, targets  # Targets remain as a list (since they are variable-length)


# Function to visualize an image with bounding boxes
def visualize_sample(image, targets):
    image = image.permute(1, 2, 0).numpy()  # Convert to (H, W, C) for visualization
    plt.imshow(image)

    for box, label in zip(targets["boxes"], targets["labels"]):
        x_min, y_min, x_max, y_max = box
        plt.gca().add_patch(
            plt.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                fill=False,
                edgecolor="red",
                linewidth=2,
            )
        )
        plt.text(
            x_min,
            y_min,
            str(label.item()),
            color="white",
            fontsize=12,
            bbox=dict(facecolor="red", alpha=0.5),
        )

    plt.show()


if __name__ == "__main__":

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),  # Converts to [0, 1] range
        ]
    )

    train_dataset = ObjectDetectionDataset(
        "data/train", "data/annotations/train", transform=transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
    )

    for batch in train_loader:
        print(f"Batch size: {len(batch[0])}")
        print(f"Images batch shape: {batch[0].shape}")  # Expecting (3, 640, 640)
        print(f"Targets: {batch[1]}")
        break

    sample_batch = next(iter(train_loader))
    images, targets = sample_batch
    visualize_sample(images[0], targets[0])
