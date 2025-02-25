# train.py

import torch
from torch.utils.data import DataLoader
from models.detr import DETR
from loss import HungarianMatcher
from loss import SetCriterion
from transformers import  DetrFeatureExtractor
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from tqdm import tqdm
from loader import ObjectDetectionDataset, collate_fn


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    num_classes = 7
    num_queries = 4  
    batch_size = 4
    epochs = 500
    lr = 1e-4
    # weight_decay = 1e-4
    weight_decay = 1e-3

    # Initialize the feature extractor
    feature_extractor = DetrFeatureExtractor.from_pretrained(
        'facebook/detr-resnet-50',
        size=800,       # Ensure this is an integer
        max_size=1333   # Ensure this is an integer
        # max_size = 800
    )


    num_classes = 1+num_classes  # +1 for the background class
    # Model
    model = DETR(num_classes=num_classes, num_queries=num_queries, pretrained_weights_path='resnet50-0676ba61.pth')
    # model.load_state_dict(torch.load('detr_model_19.pth'))
    model.to(device)

    train_dataset = CocoDetection(
        img_folder='data/BoneFractureData/train2017/images',
        ann_folder='data/BoneFractureData/annotations',
        processor=feature_extractor,
        transforms=make_coco_transforms('train'),
        train=True
    )
    val_dataset = CocoDetection(
        img_folder='data/BoneFractureData/val2017/images',
        ann_folder='data/BoneFractureData/annotations',
        processor=feature_extractor,
        transforms=make_coco_transforms('val'),
        train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')

    # Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.1)

    # Loss
    matcher = HungarianMatcher(num_classes=num_classes, cost_class=1, cost_bbox=5, cost_giou=2)
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
    criterion = SetCriterion(num_classes, matcher, weight_dict, eos_coef=0.1, losses=['labels', 'boxes'])

    #  freeze the backbone during training
    # for param in model.backbone.parameters():
    #     param.requires_grad = False

    print("Starting training...")
    best_val_loss = float('inf')
    best_model = None
    # Training Loop
    for epoch in range(epochs):
        model.train()
        criterion.train()
        running_loss = 0.0

        # Initialize progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True, position=0)

        # for images, targets in train_loader:
        # for images, targets in progress_bar:
        for i,batch in enumerate(progress_bar):
            # targets = [pad_targets(t, num_queries=num_queries, num_classes=num_classes) for t in targets]
            pixel_values = batch['pixel_values'].to(device)  # Extract pixel values
            targets = [{k: v.to(device) for k, v in t.items()} for t in batch['labels']]  # Process labels
            # replace 'class_labels' with 'labels' 
            for target in targets:
                target['labels'] = target.pop('class_labels')

            optimizer.zero_grad()
            outputs = model(pixel_values)

            # Debug: Check for NaN in outputs
            if torch.isnan(outputs['pred_logits']).any() or torch.isnan(outputs['pred_boxes']).any():
                print("NaN detected in model outputs")
                continue  # Skip this batch
            
            loss_dict = criterion(outputs, targets)
            # print("Loss Dict:", loss_dict)
            losses = sum(loss_dict[k] for k in loss_dict.keys())

            # Check for NaN in losses
            if torch.isnan(losses):
                print("NaN detected in losses")
                continue  # Skip this batch

            losses.backward()
            
            # Gradient clipping
            # clip_grad_norm_(model.parameters(), max_norm=0.1)

            optimizer.step()

            running_loss += losses.item()

        scheduler.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader)}")

        # validation and checkpoint saving
        if (epoch + 1) % 2 == 0 or epoch == 0:
            model.eval()
            criterion.eval()
            val_loss = 0.0
            with torch.no_grad():
                # for images, targets in val_loader:
                for i,batch in enumerate(val_loader):
                    # targets = [pad_targets(t, num_queries=num_queries, num_classes=num_classes) for t in targets]
                    pixel_values = batch['pixel_values'].to(device)  # Extract pixel values
                    targets = [{k: v.to(device) for k, v in t.items()} for t in batch['labels']]

                    for target in targets:
                        target['labels'] = target.pop('class_labels')

                    outputs = model(pixel_values)
                    loss_dict = criterion(outputs, targets)
                    losses = sum(loss_dict[k] for k in loss_dict.keys())
                    val_loss += losses.item()

            print(f"Validation Loss: {val_loss / len(val_loader)}")
            # save the best model
            if val_loss/len(val_loader) < best_val_loss:
                best_val_loss = val_loss/len(val_loader)
                best_model = model.state_dict()
                torch.save(best_model, 'detr_model_best.pth')
                print(f"Best model saved with val loss: {best_val_loss}")
            
            if ((epoch+1) % 10) == 0:
                torch.save(model.state_dict(), f'detr_model_{epoch+1}.pth')


    # Save the final model
    torch.save(model.state_dict(), 'detr_model.pth')


if __name__ == '__main__':
    main()