import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from architectures.AttentionUnet import AttentionUNet1024Classifier  # Modelul tău
from png_dataset import PngDataset, data_transforms

# ========== Configurație ==========
root_dir = r"D:\study\facultate\test_cuda\data"
json_path = r"D:\study\facultate\test_cuda\data\output9CanaleV3\dataset_distributie.json"

batch_size = 32
num_epochs = 30  # 30 epoci
num_folds = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model_path = "best_multi_task_model.pth"
results_txt = "training_results.txt"

# Early Stopping
early_stopping_patience = 5
early_stopping_counter = 0
best_val_loss = float("inf")

# Directory pentru rezultate
results_dir = r"D:\study\facultate\test_cuda\results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


# ========== Funcții Utilitare ==========
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        p_t = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        return focal_loss.mean()


def dice_coefficient(pred, target, smooth=1e-7):
    intersection = (pred * target).sum().item()
    dice = (2. * intersection + smooth) / (pred.sum().item() + target.sum().item() + smooth)
    return dice


def train_epoch(model, device, loader, optimizer, criterion_seg, criterion_cls):
    model.train()
    running_loss, running_seg_loss, running_cls_loss = 0.0, 0.0, 0.0
    for inputs, seg_target, cls_label in tqdm(loader, desc="Training", unit="batch"):
        inputs, seg_target, cls_label = inputs.to(device), seg_target.to(device), cls_label.to(device).float()
        optimizer.zero_grad()
        seg_logits, cls_logits = model(inputs)

        # Compute losses
        loss_seg = criterion_seg(seg_logits, seg_target.long())  # Dice Loss + CE
        loss_cls = criterion_cls(cls_logits.squeeze(1), cls_label)  # Focal Loss
        loss = loss_seg + loss_cls

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_seg_loss += loss_seg.item() * inputs.size(0)
        running_cls_loss += loss_cls.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, running_seg_loss / len(loader.dataset), running_cls_loss / len(loader.dataset)


def validate_epoch(model, device, loader, criterion_seg, criterion_cls):
    model.eval()
    running_loss, running_seg_loss, running_cls_loss = 0.0, 0.0, 0.0
    all_cls_preds, all_cls_labels = [], []
    dice_total = 0.0

    with torch.no_grad():
        for inputs, seg_target, cls_label in tqdm(loader, desc="Validating", unit="batch"):
            inputs, seg_target, cls_label = inputs.to(device), seg_target.to(device), cls_label.to(device).float()
            seg_logits, cls_logits = model(inputs)

            loss_seg = criterion_seg(seg_logits, seg_target.long())
            loss_cls = criterion_cls(cls_logits.squeeze(1), cls_label)
            loss = loss_seg + loss_cls

            running_loss += loss.item() * inputs.size(0)
            running_seg_loss += loss_seg.item() * inputs.size(0)
            running_cls_loss += loss_cls.item() * inputs.size(0)

            cls_preds = torch.sigmoid(cls_logits.squeeze(1))
            all_cls_preds.append(cls_preds.cpu())
            all_cls_labels.append(cls_label.cpu())

            seg_preds = torch.argmax(seg_logits, dim=1)
            for i in range(seg_target.size(0)):
                dice_total += dice_coefficient(seg_preds[i].float(), seg_target[i].float())

    epoch_loss = running_loss / len(loader.dataset)
    auroc = roc_auc_score(torch.cat(all_cls_labels).numpy(), torch.cat(all_cls_preds).numpy()) if all_cls_preds else float("nan")
    mean_dice = dice_total / len(loader.dataset)
    return epoch_loss, running_seg_loss / len(loader.dataset), running_cls_loss / len(loader.dataset), auroc, mean_dice


# ========== Antrenare cu K-Fold ==========
def train_model():
    global best_val_loss, early_stopping_counter

    with open(json_path, 'r') as f:
        data_dict = json.load(f)

    patients_data = {**data_dict["cancer"], **data_dict["non_cancer"]} if "cancer" in data_dict else data_dict
    if not patients_data:
        print("ERROR: Dataset JSON is gol!")
        sys.exit(1)

    keys = list(patients_data.keys())
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    best_overall_auroc = 0.0
    fold_index = 1

    for train_index, val_index in kf.split(keys):
        print(f"\n=== Fold {fold_index}/{num_folds} ===")
        train_keys = [keys[i] for i in train_index]
        val_keys = [keys[i] for i in val_index]

        train_dataset = PngDataset({k: patients_data[k] for k in train_keys}, root_dir, transform=data_transforms)
        val_dataset = PngDataset({k: patients_data[k] for k in val_keys}, root_dir, transform=data_transforms)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        model = AttentionUNet1024Classifier(n_channels=3, n_seg_classes=2, n_bin_classes=1).to(device)
        criterion_seg = nn.CrossEntropyLoss()
        criterion_cls = FocalLoss()
        optimizer = optim.RAdam(model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)

        best_auroc_fold = 0.0
        best_model_fold = None

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            train_loss, train_seg_loss, train_cls_loss = train_epoch(model, device, train_loader, optimizer,
                                                                     criterion_seg, criterion_cls)
            val_loss, val_seg_loss, val_cls_loss, auroc, mean_dice = validate_epoch(model, device, val_loader,
                                                                                    criterion_seg, criterion_cls)
            scheduler.step(val_loss)

            print(f"  Train Loss: {train_loss:.4f} (Seg: {train_seg_loss:.4f}, Cls: {train_cls_loss:.4f})")
            print(
                f"  Val Loss: {val_loss:.4f} (Seg: {val_seg_loss:.4f}, Cls: {val_cls_loss:.4f}) | AUROC: {auroc:.4f} | Dice: {mean_dice:.4f}")

            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping activat!")
                break

        if best_auroc_fold > best_overall_auroc:
            best_overall_auroc = best_auroc_fold
            torch.save(best_model_fold, best_model_path)
            print(f"Salvat modelul cu AUROC {best_overall_auroc:.4f}")

        fold_index += 1

    print(f"\nAntrenare finalizată! Cel mai bun model are AUROC: {best_overall_auroc:.4f}")


if __name__ == "__main__":
    train_model()
