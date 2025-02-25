import torch
from torchvision.ops.boxes import box_area
import torch.nn.functional as F
import torch.distributed as dist


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # print("Boxes1:", boxes1)
    # print("Boxes2:", boxes2)
    
    # assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    # assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

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
