import os
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from zoo.VNet import UNetSegClassifier1024
from dataloader import PngDataset, data_transforms
from zoo.AttentionUnet import AttentionUNet1024Classifier


def load_data(json_path):
    """Încarcă datele din fișierul JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def dice_coefficient(pred, target, smooth=1e-7):
    """
    Calculează coeficientul Dice pentru două mape binare (0/1).
    pred și target sunt tensori de tip float/binar (0/1).
    """
    intersection = (pred * target).sum().item()
    dice = (2. * intersection + smooth) / (pred.sum().item() + target.sum().item() + smooth)
    return dice


def train_epoch(model, device, loader, optimizer, criterion_cls, criterion_seg):
    """
    Antrenează modelul multi-task (segmentare + clasificare) pe un epoch.
    - model(data) -> (seg_logits, cls_logits)
    - data, mask, gs provin din DataLoader: (image, mask, GS)
    """
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_seg_loss = 0.0
    for inputs, seg_target, gs in tqdm(loader, desc="Training", unit="batch"):
        inputs = inputs.to(device)
        seg_target = seg_target.to(device)
        gs = gs.to(device).float()

        optimizer.zero_grad()
        seg_logits, cls_logits = model(inputs)
        cls_loss = criterion_cls(cls_logits.squeeze(1), gs)
        seg_loss = criterion_seg(seg_logits, seg_target)

        # Aici am pus weight=0 la seg_loss,
        # dar poți schimba la 1 dacă vrei să antrenezi și segmentarea
        # loss = cls_loss + 0 * seg_loss
        loss = cls_loss + 0.1 * seg_loss  # încearcă 0.1, 0.5, 1
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_cls_loss += cls_loss.item() * inputs.size(0)
        running_seg_loss += seg_loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_cls_loss = running_cls_loss / len(loader.dataset)
    epoch_seg_loss = running_seg_loss / len(loader.dataset)
    return epoch_loss, epoch_cls_loss, epoch_seg_loss


def validate_epoch(model, device, loader, criterion_cls, criterion_seg):
    """
    Validează modelul (segmentare + clasificare) și returnează:
      - Val Loss
      - Acuratețe, Precizie, Recall, Specificitate, F1, AUROC
      - Dice mediu (pe 2 clase: fundal și tumoare)
    """
    model.eval()
    running_loss = 0.0
    all_cls_preds = []
    all_cls_labels = []

    dice_scores_background = []
    dice_scores_tumor = []
    dice_scores_mean = []

    with torch.no_grad():
        for inputs, seg_target, gs in tqdm(loader, desc="Validating", unit="batch"):
            inputs = inputs.to(device)
            seg_target = seg_target.to(device)
            gs = gs.to(device).float()

            seg_logits, cls_logits = model(inputs)
            cls_loss = criterion_cls(cls_logits.squeeze(1), gs)
            seg_loss = criterion_seg(seg_logits, seg_target)

            # Aici punem weight=0 la seg_loss,
            # dar poți schimba la 1 dacă vrei să antrenezi și segmentarea
            loss = cls_loss + 0 * seg_loss

            running_loss += loss.item() * inputs.size(0)

            # Clasificare
            cls_preds = torch.sigmoid(cls_logits.squeeze(1))
            all_cls_preds.append(cls_preds.cpu())
            all_cls_labels.append(gs.cpu())

            # Segmentare (dacă vrei să monitorizezi dice, chiar dacă weight=0)
            seg_preds = torch.argmax(seg_logits, dim=1)  # [B, H, W]
            for i in range(seg_target.size(0)):
                target_img = seg_target[i]
                pred_img = seg_preds[i]
                # 0 = fundal, 1 = tumoare
                bg_target = (target_img == 0).float()
                bg_pred = (pred_img == 0).float()
                tumor_target = (target_img == 1).float()
                tumor_pred = (pred_img == 1).float()

                dice_bg = dice_coefficient(bg_pred, bg_target)
                dice_tumor = dice_coefficient(tumor_pred, tumor_target)
                dice_mean_img = (dice_bg + dice_tumor) / 2.0
                dice_scores_background.append(dice_bg)
                dice_scores_tumor.append(dice_tumor)
                dice_scores_mean.append(dice_mean_img)

    epoch_loss = running_loss / len(loader.dataset)
    all_cls_preds = torch.cat(all_cls_preds)
    all_cls_labels = torch.cat(all_cls_labels)

    # Clasificare
    cls_pred_labels = (all_cls_preds > 0.5).float()
    accuracy = (cls_pred_labels == all_cls_labels).float().mean().item()
    TP = ((cls_pred_labels == 1) & (all_cls_labels == 1)).sum().item()
    TN = ((cls_pred_labels == 0) & (all_cls_labels == 0)).sum().item()
    FP = ((cls_pred_labels == 1) & (all_cls_labels == 0)).sum().item()
    FN = ((cls_pred_labels == 0) & (all_cls_labels == 1)).sum().item()

    precision = TP / (TP + FP + 1e-7)
    recall = TP / (TP + FN + 1e-7)
    specificity = TN / (TN + FP + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    try:
        auroc = roc_auc_score(all_cls_labels.numpy(), all_cls_preds.numpy())
    except ValueError:
        auroc = float('nan')

    dice_bg_mean = np.mean(dice_scores_background)
    dice_tumor_mean = np.mean(dice_scores_tumor)
    dice_mean = np.mean(dice_scores_mean)

    return epoch_loss, accuracy, precision, recall, specificity, f1, auroc, dice_bg_mean, dice_tumor_mean, dice_mean


def run_kfold_training(json_path, root_dir, num_epochs=20, batch_size=16, k_folds=5):
    """
    K-Fold Cross Validation pentru antrenarea multi-task (segmentare + clasificare).
    -> Checkpoint se salvează după cel mai bun AUROC (nu după val_acc).
    """
    import shutil
    from sklearn.model_selection import KFold

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Încărcăm datele
    with open(json_path, 'r') as f:
        data_dict = json.load(f)
    keys = list(data_dict.keys())
    np.random.shuffle(keys)

    # Definim criteriile
    criterion_cls = torch.nn.BCEWithLogitsLoss()
    criterion_seg = torch.nn.CrossEntropyLoss()

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Vom stoca cele mai bune valori pt fold, definim structura
    fold_results = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(keys)):
        fold_num = fold + 1
        print(f"\n===== Fold {fold_num}/{k_folds} =====")

        fold_dir = os.path.join("results", f"fold_{fold_num}")
        if os.path.exists(fold_dir):
            shutil.rmtree(fold_dir)
        os.makedirs(fold_dir, exist_ok=True)

        train_keys = [keys[i] for i in train_idx]
        valid_keys = [keys[i] for i in valid_idx]

        train_data = {k: data_dict[k] for k in train_keys}
        valid_data = {k: data_dict[k] for k in valid_keys}

        # Salvăm datele de validare
        val_data_path = os.path.join(fold_dir, "val_data.json")
        with open(val_data_path, 'w') as f:
            json.dump(valid_data, f, indent=4)
        print(f"Validation data for fold {fold_num} saved in {val_data_path}")

        # Construim dataseturile
        train_dataset = PngDataset(train_data, root_dir, transform=data_transforms)
        valid_dataset = PngDataset(valid_data, root_dir, transform=data_transforms)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        # Model + optimizer
        model = UNetSegClassifier1024(1,2,1)  # segmentare 2 clase
        # NOTĂ: Dacă vrei clasificare binară -> n_bin_classes=1
        # Poate fi: model = AttentionUNet1024Classifier(1,2,1).to(device)
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        # Vom face checkpoint când AUROC e cel mai mare
        best_val_auroc = 0.0
        best_val_acc = 0.0
        best_val_recall = 0.0
        best_val_spec = 0.0

        # Pentru plot
        epoch_nums = []
        train_loss_list = []
        val_loss_list = []
        accuracy_list = []
        auroc_list = []

        for epoch in range(num_epochs):
            train_loss, train_cls_loss, train_seg_loss = train_epoch(
                model, device, train_loader, optimizer, criterion_cls, criterion_seg
            )
            (val_loss, val_acc, precision, recall, specificity, f1, auroc,
             dice_bg, dice_tumor, dice_mean) = validate_epoch(
                model, device, valid_loader, criterion_cls, criterion_seg
            )

            current_lr = scheduler.get_last_lr()[0]
            print(f"[Fold {fold_num}, Epoch {epoch + 1}/{num_epochs}]")
            print(f"  Train Loss: {train_loss:.4f} (Cls: {train_cls_loss:.4f}, Seg: {train_seg_loss:.4f})")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | AUROC: {auroc:.4f}")
            print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | Specificity: {specificity:.4f}")
            print(f"  Dice Mean: {dice_mean:.4f} | LR: {current_lr:.6f}")

            epoch_nums.append(epoch + 1)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            accuracy_list.append(val_acc)
            auroc_list.append(auroc)

            # Salvăm modelul dacă AUROC e mai bun
            if auroc > best_val_auroc:
                best_val_auroc = auroc
                best_val_acc = val_acc
                best_val_recall = recall
                best_val_spec = specificity
                checkpoint_path = os.path.join(fold_dir, "best_model.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  [Fold {fold_num}] Best model saved at {checkpoint_path} (AUROC improved)")

            scheduler.step()

        # Plotăm metricile
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_nums, train_loss_list, label="Train Loss")
        plt.plot(epoch_nums, val_loss_list, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Fold {fold_num} - Loss per Epoch")
        plt.legend()
        plt.grid(True)
        plot_loss_path = os.path.join(fold_dir, "loss_per_epoch.png")
        plt.savefig(plot_loss_path)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(epoch_nums, accuracy_list, label="Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Fold {fold_num} - Accuracy per Epoch")
        plt.legend()
        plt.grid(True)
        plot_acc_path = os.path.join(fold_dir, "accuracy_per_epoch.png")
        plt.savefig(plot_acc_path)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(epoch_nums, auroc_list, label="AUROC")
        plt.xlabel("Epoch")
        plt.ylabel("AUROC")
        plt.title(f"Fold {fold_num} - AUROC per Epoch")
        plt.legend()
        plt.grid(True)
        plot_auroc_path = os.path.join(fold_dir, "auroc_per_epoch.png")
        plt.savefig(plot_auroc_path)
        plt.close()

        fold_results.append({
            "fold": fold_num,
            "best_val_acc": best_val_acc,
            "best_val_auroc": best_val_auroc,
            "best_val_recall": best_val_recall,
            "best_val_spec": best_val_spec
        })

    avg_acc = np.mean([fr["best_val_acc"] for fr in fold_results])
    avg_auroc = np.mean([fr["best_val_auroc"] for fr in fold_results])
    avg_recall = np.mean([fr["best_val_recall"] for fr in fold_results])
    avg_spec = np.mean([fr["best_val_spec"] for fr in fold_results])

    print("\n===== K-Fold Cross Validation Results =====")
    for fr in fold_results:
        print(f"Fold {fr['fold']} -> ACC: {fr['best_val_acc']:.4f}, AUROC: {fr['best_val_auroc']:.4f}, "
              f"Recall: {fr['best_val_recall']:.4f}, Spec: {fr['best_val_spec']:.4f}")
    print(
        f"\nOverall -> Accuracy: {avg_acc:.4f} | AUROC: {avg_auroc:.4f} | Recall: {avg_recall:.4f} | Specificity: {avg_spec:.4f}")


if __name__ == "__main__":
    json_path = r"D:\study\facultate\test_cuda\data\1234.json"
    root_dir = r"D:\study\facultate\test_cuda"
    run_kfold_training(json_path, root_dir, num_epochs=14, batch_size=32, k_folds=5)
