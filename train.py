import torch
from torch.utils.data import DataLoader
from models.detr import DETR
from models.loss import HungarianMatcher, SetCriterion
from transformers import DetrImageProcessor
from utils.utils import collate_fn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from dataset.coco import CocoDetection, make_coco_transforms


def initialize_feature_extractor():
    """Initializes the feature extractor for DETR."""
    return DetrImageProcessor.from_pretrained(
        'facebook/detr-resnet-50',
        do_rescale=False
    )


def initialize_dataloader(batch_size, feature_extractor):
    """Creates the train and validation data loaders."""
    train_dataset = CocoDetection(
        img_folder='data/train',
        ann_folder='data/annotations',
        processor=feature_extractor,
        transforms=make_coco_transforms('train'),
        train=True
    )
    val_dataset = CocoDetection(
        img_folder='data/val',
        ann_folder='data/annotations',
        processor=feature_extractor,
        transforms=make_coco_transforms('val'),
        train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    num_classes = 4 + 1  # +1 for the background class
    num_queries = 4
    batch_size = 4
    epochs = 500
    lr = 1e-4
    weight_decay = 1e-3

    # Initialize feature extractor and dataloaders
    feature_extractor = initialize_feature_extractor()
    train_loader, val_loader = initialize_dataloader(batch_size, feature_extractor)

    # Initialize model
    model = DETR(num_classes=num_classes, num_queries=num_queries)
    model.to(device)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.1)

    # Loss function
    matcher = HungarianMatcher(num_classes=num_classes, cost_class=1, cost_bbox=5, cost_giou=2)
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
    criterion = SetCriterion(num_classes, matcher, weight_dict, eos_coef=0.1, losses=['labels', 'boxes'])

    print("Starting training...")
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True, position=0)

        for batch in progress_bar:
            pixel_values = batch['pixel_values'].to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in batch['labels']]
            
            # Replace 'class_labels' with 'labels'
            for target in targets:
                target['labels'] = target.pop('class_labels')

            optimizer.zero_grad()
            outputs = model(pixel_values)
            print(pixel_values, targets, outputs)
            exit()

            # Debugging: Check for NaNs
            if torch.isnan(outputs['pred_logits']).any() or torch.isnan(outputs['pred_boxes']).any():
                print("Warning: NaN detected in model outputs. Skipping batch.")
                continue

            loss_dict = criterion(outputs, targets)
            losses = sum(loss_dict.values())

            if torch.isnan(losses):
                print("Warning: NaN detected in loss computation. Skipping batch.")
                continue

            losses.backward()
            optimizer.step()

            running_loss += losses.item()

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.4f}")

        # Validation and checkpoint saving
        if (epoch + 1) % 2 == 0 or epoch == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    pixel_values = batch['pixel_values'].to(device)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in batch['labels']]

                    for target in targets:
                        target['labels'] = target.pop('class_labels')

                    outputs = model(pixel_values)
                    loss_dict = criterion(outputs, targets)
                    val_loss += sum(loss_dict.values()).item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch [{epoch + 1}/{epochs}] - Validation Loss: {avg_val_loss:.4f}")

            # Save best model checkpoint
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), 'detr_model_best.pth')
                print(f"Best model saved with val loss: {best_val_loss:.4f}")

            # Save model checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), f'detr_model_{epoch+1}.pth')

    # Save the final model
    torch.save(model.state_dict(), 'detr_model.pth')
    print("Training complete. Final model saved as 'detr_model.pth'.")


if __name__ == '__main__':
    main()
