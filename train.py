import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from loader import ObjectDetectionDataset, collate_fn
from loss import HungarianMatcher, SetCriterion
from models.detr import DETR

def get_device():
    """Returns the available device (CUDA or CPU)."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def get_transforms():
    """Returns the image transformations."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # Converts to [0, 1] range
    ])

def get_dataloaders(batch_size):
    """Loads training and validation datasets and returns corresponding dataloaders."""
    transform = get_transforms()
    
    train_dataset = ObjectDetectionDataset(
        "data/train", "data/annotations/train", transform=transform
    )
    val_dataset = ObjectDetectionDataset(
        "data/validate", "data/annotations/validate", transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader


def get_model(num_classes, num_queries, device):
    """Initializes and returns the DETR model."""
    model = DETR(num_classes=num_classes, num_queries=num_queries)
    model.to(device)
    return model


def get_loss_criterion(num_classes):
    """Returns the loss criterion for training."""
    matcher = HungarianMatcher(num_classes=num_classes, cost_class=1, cost_bbox=5, cost_giou=2)
    weight_dict = {"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2}
    return SetCriterion(num_classes, matcher, weight_dict, eos_coef=0.1, losses=["labels", "boxes"])

def compute_metrics(pred_logits, pred_boxes, targets, num_classes, thresholds=torch.arange(0, 1.05, 0.05).tolist()):

    device = pred_logits.device
    batch_size, num_queries, _ = pred_logits.shape

    # Convert logits to probabilities (assuming softmax was used)
    pred_probs = F.softmax(pred_logits, dim=-1)  # Shape: [batch_size, num_queries, num_classes + 1]

    # Remove background class (last index)
    pred_probs = pred_probs[..., :-1]  # Shape: [batch_size, num_queries, num_classes]
    pred_classes = torch.argmax(pred_probs, dim=-1)  # Shape: [batch_size, num_queries]

    # Flatten all predictions and targets for processing
    all_preds = []
    all_targets = []
    
    for batch_idx in range(batch_size):
        pred_labels = pred_classes[batch_idx]  # [num_queries]
        pred_confidences = pred_probs[batch_idx].max(dim=-1)[0]  # Highest confidence per query
        pred_bboxes = pred_boxes[batch_idx]  # [num_queries, 4]
        
        target_labels = targets[batch_idx]["labels"].to(device)  # [N]
        target_bboxes = targets[batch_idx]["boxes"].to(device)  # [N, 4]

        # Ignore background labels (num_classes) in ground truth
        foreground_mask = target_labels < num_classes  # Exclude background (last class)
        target_labels = target_labels[foreground_mask]
        target_bboxes = target_bboxes[foreground_mask]

        all_preds.append((pred_labels, pred_confidences, pred_bboxes))
        all_targets.append((target_labels, target_bboxes))

    # Store max F1-score and accuracy per class
    max_f1_per_class = {c: 0.0 for c in range(num_classes)}
    accuracy_per_class = {c: 0.0 for c in range(num_classes)}

    for threshold in thresholds:
        tp_per_class = {c: 0 for c in range(num_classes)}
        fp_per_class = {c: 0 for c in range(num_classes)}
        fn_per_class = {c: 0 for c in range(num_classes)}
        total_per_class = {c: 0 for c in range(num_classes)}

        for (pred_labels, pred_confidences, pred_bboxes), (target_labels, target_bboxes) in zip(all_preds, all_targets):
            # Filter predictions by confidence threshold and ignore background class
            valid_preds = pred_confidences > threshold
            pred_labels = pred_labels[valid_preds]
            pred_bboxes = pred_bboxes[valid_preds]

            matched_preds = set()
            matched_targets = set()

            # Match predictions with ground truth using IoU
            for t_idx, target_label in enumerate(target_labels):
                total_per_class[target_label.item()] += 1

                best_iou = 0
                best_p_idx = -1

                for p_idx, pred_label in enumerate(pred_labels):
                    if pred_label == target_label:
                        iou = compute_iou(pred_bboxes[p_idx], target_bboxes[t_idx])
                        if iou > best_iou:
                            best_iou = iou
                            best_p_idx = p_idx

                # Consider it a True Positive if IoU > 0.5
                if best_iou > 0.5:
                    tp_per_class[target_label.item()] += 1
                    matched_preds.add(best_p_idx)
                    matched_targets.add(t_idx)
                else:
                    fn_per_class[target_label.item()] += 1

            # False Positives: Predictions that did not match any ground truth
            for p_idx, pred_label in enumerate(pred_labels):
                if p_idx not in matched_preds and pred_label < num_classes:
                    fp_per_class[pred_label.item()] += 1

        # Compute metrics for each class
        for c in range(num_classes):
            tp = tp_per_class[c]
            fp = fp_per_class[c]
            fn = fn_per_class[c]
            total = total_per_class[c]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = tp / total if total > 0 else 0.0

            max_f1_per_class[c] = max(max_f1_per_class[c], f1_score)
            accuracy_per_class[c] = accuracy

    return {"max_f1_per_class": max_f1_per_class, "accuracy_per_class": accuracy_per_class}


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Trains the model for one epoch."""
    model.train()
    criterion.train()
    running_loss = 0.0

    progress_bar = tqdm(train_loader, desc="Training", leave=True, position=0)

    for batch in progress_bar:
        pixel_values = batch[0].to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in batch[1]]
        
        optimizer.zero_grad()
        outputs = model(pixel_values)

        # Check for NaN values
        if torch.isnan(outputs["pred_logits"]).any() or torch.isnan(outputs["pred_boxes"]).any():
            print("NaN detected in model outputs, skipping batch.")
            continue

        loss_dict = criterion(outputs, targets)
        losses = sum(loss_dict.values())

        if torch.isnan(losses):
            print("NaN detected in losses, skipping batch.")
            continue

        losses.backward()
        optimizer.step()
        running_loss += losses.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Training Loss: {avg_loss:.4f}")
    return avg_loss


def validate(model, val_loader, criterion, device, num_classes):
    """Validates the model and returns the validation loss and aggregated metrics."""
    model.eval()
    criterion.eval()
    val_loss = 0.0

    # Initialize accumulators for metrics
    max_f1_per_class = {c: 0.0 for c in range(num_classes)}
    total_accuracy_per_class = {c: 0.0 for c in range(num_classes)}
    batch_count = 0

    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch[0].to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in batch[1]]
            outputs = model(pixel_values)

            pred_logits = outputs["pred_logits"]
            pred_boxes = outputs["pred_boxes"]

            # Compute metrics for current batch
            metrics = compute_metrics(pred_logits, pred_boxes, targets, num_classes)

            # Aggregate max F1-score per class (keep the max seen across all batches)
            for c in range(num_classes):
                max_f1_per_class[c] = max(max_f1_per_class[c], metrics["max_f1_per_class"][c])

                # Sum accuracy per class for averaging later
                total_accuracy_per_class[c] += metrics["accuracy_per_class"][c]

            batch_count += 1

            # Compute validation loss
            loss_dict = criterion(outputs, targets)
            losses = sum(loss_dict.values())
            val_loss += losses.item()

    # Compute the average accuracy per class
    avg_accuracy_per_class = {c: total_accuracy_per_class[c] / batch_count for c in range(num_classes)}

    # Aggregate metrics dictionary
    aggregated_metrics = {
        "max_f1_per_class": max_f1_per_class,
        "avg_accuracy_per_class": avg_accuracy_per_class
    }

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Aggregated Metrics: {aggregated_metrics}")

    return avg_val_loss, aggregated_metrics

def save_checkpoint(model, epoch, best_model, best_val_loss):
    """Saves the model checkpoint."""
    if best_model:
        torch.save(best_model, "detr_model_best.pth")
        print(f"Best model saved with validation loss: {best_val_loss:.4f}")

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"detr_model_{epoch+1}.pth")


def main():
    # Configuration
    num_classes = 4  # Number of object classes
    num_queries = 4   # Number of queries in DETR
    batch_size = 4
    epochs = 10
    lr = 1e-4
    weight_decay = 1e-3

    num_classes += 1  # +1 for background class
    device = get_device()

    # Initialize model, optimizer, scheduler, loss function, and data loaders
    model = get_model(num_classes, num_queries, device)
    train_loader, val_loader = get_dataloaders(batch_size)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.1)
    criterion = get_loss_criterion(num_classes)

    best_val_loss = float("inf")
    best_model = None

    # Training loop
    print("Starting training...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        # Validation every 2 epochs or on first epoch
        if (epoch + 1) % 2 == 0 or epoch == 0:
            val_loss, val_metrics = validate(model, val_loader, criterion, device)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()

            save_checkpoint(model, epoch, best_model, best_val_loss)

    # Save final model
    torch.save(model.state_dict(), "detr_model.pth")
    print("Training complete. Model saved as 'detr_model.pth'.")


if __name__ == "__main__":
    main()
