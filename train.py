import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from loader import ObjectDetectionDataset, collate_fn
from loss import HungarianMatcher, SetCriterion
from models.detr import DETR


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    num_classes = 4
    num_queries = 4
    batch_size = 4
    epochs = 500
    lr = 1e-4
    weight_decay = 1e-3

    
    num_classes = 1 + num_classes  # +1 for the background class
    
    model = DETR(
        num_classes=num_classes,
        num_queries=num_queries,
        pretrained_weights_path="resnet50-0676ba61.pth",
    )
    model.to(device)

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
    val_dataset = ObjectDetectionDataset(
        "data/validate", "data/annotations/validate", transform=transform
    )
    val_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.1)

    matcher = HungarianMatcher(
        num_classes=num_classes, cost_class=1, cost_bbox=5, cost_giou=2
    )
    weight_dict = {"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2}
    
    criterion = SetCriterion(
        num_classes, matcher, weight_dict, eos_coef=0.1, losses=["labels", "boxes"]
    )

    
    print("Starting training...")
    best_val_loss = float("inf")
    best_model = None
    # Training Loop
    for epoch in range(epochs):
        model.train()
        criterion.train()
        running_loss = 0.0

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}", leave=True, position=0
        )

        for i, batch in enumerate(progress_bar):
            pixel_values = batch[0].to(device)
            targets = batch[1]
            optimizer.zero_grad()
            outputs = model(pixel_values)

            if (
                torch.isnan(outputs["pred_logits"]).any()
                or torch.isnan(outputs["pred_boxes"]).any()
            ):
                print("NaN detected in model outputs")
                continue

            loss_dict = criterion(outputs, targets)
            losses = sum(loss_dict[k] for k in loss_dict.keys())

            if torch.isnan(losses):
                print("NaN detected in losses")
                continue

            losses.backward()
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
                for i, batch in enumerate(val_loader):
                    pixel_values = batch[0].to(device)
                    targets = batch[1]
                    outputs = model(pixel_values)
                    loss_dict = criterion(outputs, targets)
                    losses = sum(loss_dict[k] for k in loss_dict.keys())
                    val_loss += losses.item()

            print(f"Validation Loss: {val_loss / len(val_loader)}")
            if val_loss / len(val_loader) < best_val_loss:
                best_val_loss = val_loss / len(val_loader)
                best_model = model.state_dict()
                torch.save(best_model, "detr_model_best.pth")
                print(f"Best model saved with val loss: {best_val_loss}")

            if ((epoch + 1) % 10) == 0:
                torch.save(model.state_dict(), f"detr_model_{epoch+1}.pth")

    torch.save(model.state_dict(), "detr_model.pth")


if __name__ == "__main__":
    main()
