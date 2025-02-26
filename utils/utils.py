import torch
import torch.nn.functional as F

import torch.distributed as dist


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def collate_fn(batch):
        """
        Custom collate function to handle batches of pixel values and labels.
        Manually pad the pixel values to ensure consistent size.
        """
        pixel_values = [item[0] for item in batch]  # List of tensors
        labels = [item[1] for item in batch]       # List of labels

        # Find the maximum height and width in the batch
        max_height = max([img.shape[1] for img in pixel_values])
        max_width = max([img.shape[2] for img in pixel_values])

        # Pad all images to the maximum height and width
        padded_images = []
        pixel_masks = []
        for img in pixel_values:
            _, h, w = img.shape
            padded_img = F.pad(img, (0, max_width - w, 0, max_height - h), value=0)  # Pad with zeros
            padded_images.append(padded_img)

            # Create a mask for the padded regions
            mask = torch.zeros((max_height, max_width), dtype=torch.bool)
            mask[:h, :w] = 1
            pixel_masks.append(mask)

        # Stack padded images and masks
        pixel_values = torch.stack(padded_images)
        pixel_masks = torch.stack(pixel_masks)

        # Create the batch dictionary
        batch = {
            'pixel_values': pixel_values,  # Tensor of shape [batch_size, 3, max_height, max_width]
            'pixel_mask': pixel_masks,     # Tensor of shape [batch_size, max_height, max_width]
            'labels': labels,              # List of label dictionaries
        }
        return batch
