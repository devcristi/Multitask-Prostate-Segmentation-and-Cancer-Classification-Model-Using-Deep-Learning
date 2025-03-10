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
import torchvision.transforms as transforms
from PIL import Image

from architectures.AttentionUnet import AttentionUNet1024Classifier  # Using your custom UNet implementation

# ========== Configuration ==========
# Ajustează aceste directoare după necesitate.
root_dir = r"D:\study\facultate\test_cuda\data"  # Directorul de bază. Când se face join cu căile din JSON se obțin căile complete.
json_path = r"D:\study\facultate\test_cuda\data\output9CanaleV3\dataset_distributie.json"

batch_size = 32
num_epochs = 20  # Număr mai mare de epoci pentru antrenare
num_folds = 5  # Cross Validation: numărul de fold-uri.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model_overall_path = "best_multi_task_model.pth"
results_txt = "training_results.txt"

# Directorul pentru salvarea imaginilor de comparație a segmentărilor
results_dir = r"D:\study\facultate\test_cuda\results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


# ========== Utility Functions ==========
def dice_coefficient(pred, target, smooth=1e-7):
    """
    Compute the Dice coefficient for a single image.
    Both pred and target should be binary masks.
    """
    intersection = (pred * target).sum().item()
    dice = (2. * intersection + smooth) / (pred.sum().item() + target.sum().item() + smooth)
    return dice


def train_epoch(model, device, loader, optimizer, criterion_seg, criterion_cls):
    model.train()
    running_loss = 0.0
    running_seg_loss = 0.0
    running_cls_loss = 0.0
    for inputs, seg_target, cls_label in tqdm(loader, desc="Training", unit="batch"):
        inputs = inputs.to(device)
        seg_target = seg_target.to(device)  # Shape: [B, H, W]
        cls_label = cls_label.to(device).float()  # Shape: [B]
        optimizer.zero_grad()
        seg_logits, cls_logits = model(inputs)
        loss_seg = criterion_seg(seg_logits, seg_target)
        loss_cls = criterion_cls(cls_logits.squeeze(1), cls_label)
        loss = loss_seg + loss_cls
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_seg_loss += loss_seg.item() * inputs.size(0)
        running_cls_loss += loss_cls.item() * inputs.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    epoch_seg_loss = running_seg_loss / len(loader.dataset)
    epoch_cls_loss = running_cls_loss / len(loader.dataset)
    return epoch_loss, epoch_seg_loss, epoch_cls_loss


def validate_epoch(model, device, loader, criterion_seg, criterion_cls):
    model.eval()
    running_loss = 0.0
    running_seg_loss = 0.0
    running_cls_loss = 0.0
    all_cls_preds = []
    all_cls_labels = []
    dice_total = 0.0
    with torch.no_grad():
        for inputs, seg_target, cls_label in tqdm(loader, desc="Validating", unit="batch"):
            inputs = inputs.to(device)
            seg_target = seg_target.to(device)
            cls_label = cls_label.to(device).float()
            seg_logits, cls_logits = model(inputs)
            loss_seg = criterion_seg(seg_logits, seg_target)
            loss_cls = criterion_cls(cls_logits.squeeze(1), cls_label)
            loss = loss_seg + loss_cls
            running_loss += loss.item() * inputs.size(0)
            running_seg_loss += loss_seg.item() * inputs.size(0)
            running_cls_loss += loss_cls.item() * inputs.size(0)
            cls_preds = torch.sigmoid(cls_logits.squeeze(1))
            all_cls_preds.append(cls_preds.cpu())
            all_cls_labels.append(cls_label.cpu())
            seg_preds = torch.argmax(seg_logits, dim=1)  # [B, H, W]
            for i in range(seg_target.size(0)):
                dice_total += dice_coefficient(seg_preds[i].float(), seg_target[i].float())
    epoch_loss = running_loss / len(loader.dataset)
    epoch_seg_loss = running_seg_loss / len(loader.dataset)
    epoch_cls_loss = running_cls_loss / len(loader.dataset)
    all_cls_preds = torch.cat(all_cls_preds)
    all_cls_labels = torch.cat(all_cls_labels)
    try:
        auroc = roc_auc_score(all_cls_labels.numpy(), all_cls_preds.numpy())
    except ValueError:
        auroc = float('nan')
    mean_dice = dice_total / len(loader.dataset)
    return epoch_loss, epoch_seg_loss, epoch_cls_loss, auroc, mean_dice


def plot_history(epochs, history, ylabel, title, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), history, marker='o', label=ylabel)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def save_segmentation_comparison(val_dataset, model, device, fold, epoch, results_dir):
    """
    Salvează o comparație între segmentarea originală și predicția modelului.
    Folosește primul eșantion din dataset-ul de validare.
    """
    model.eval()
    sample, gt_mask, _ = val_dataset[0]
    input_img = sample.unsqueeze(0).to(device)
    with torch.no_grad():
        seg_logits, _ = model(input_img)
    pred_mask = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy()

    # Pentru afișare, selectăm doar primul canal din imagine (deoarece sample are 9 canale)
    img_np = sample.cpu().permute(1, 2, 0).numpy()  # forma: (128, 128, 9)
    # Selectăm primul canal pentru vizualizare, ca imagine grayscale:
    img_to_show = img_np[:, :, 0]
    img_to_show = np.clip(img_to_show, 0, 1)

    gt_mask_np = gt_mask.cpu().numpy()

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_to_show, cmap="gray")
    plt.title("Original Image (Channel 0)")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(gt_mask_np, cmap="gray")
    plt.title("Ground Truth Segmentation")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("Predicted Segmentation")
    plt.axis("off")
    plt.tight_layout()

    save_path = os.path.join(results_dir, f"fold{fold}_epoch{epoch}_segmentation_comparison.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Segmentation comparison saved to {save_path}")


# ========== Dataset Classes ==========

# --- Noul dataset care încarcă .npy cu concatenare pe canale (rezultat: 128x128x9) ---
class NpyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict, root_dir, split="train", test_ratio=0.2, seed=42):
        """
        Dataset care:
          - Împarte cheile (pacienții) în train/test în funcție de test_ratio.
          - Pentru fiecare pacient, în __getitem__:
             * Încarcă imaginea .npy (128x128x9) din directorul 'output9CanaleV3/concatenare_canale'
             * Încărcă masca PNG (felia de mijloc), o redimensionează la 128x128 și o convertește în tensor Long.
             * Returnează (image_tensor, mask_tensor, gs)
        """
        valid_keys = [key for key, item in data_dict.items() if "slice" in item and "data" in item and "label" in item]
        valid_keys.sort()
        np.random.seed(seed)
        np.random.shuffle(valid_keys)
        test_size = int(len(valid_keys) * test_ratio)
        if split == "train":
            self.keys = valid_keys[test_size:]
        else:
            self.keys = valid_keys[:test_size]
        self.data_dict = data_dict
        self.root_dir = root_dir  # ex: D:\study\facultate\test_cuda\data
        self.resize_mask = transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        item = self.data_dict[key]

        # 1) Încarcă imaginea .npy (128,128,9) pentru pacient.
        # Presupunem că fișierul se numește: {key}_9channels.npy și se află în:
        # D:\study\facultate\test_cuda\data\output9CanaleV3\concatenare_canale\
        npy_path = os.path.join(self.root_dir, "output9CanaleV3", "concatenare_canale", f"{key}_9channels.npy")
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Nu am găsit fișierul: {npy_path}")
        img_9ch = np.load(npy_path)  # forma: (128, 128, 9)
        # Convertim la tensor PyTorch și permutăm la [C, H, W]
        image_tensor = torch.from_numpy(img_9ch).permute(2, 0, 1).float()

        # 2) Încarcă masca: se folosește felia de mijloc.
        slices = sorted(item["slice"])
        mid_slice = slices[len(slices) // 2]
        slice_index = item["slice"].index(mid_slice)
        mask_relative_path = item["label"][slice_index]
        mask_path = os.path.join(self.root_dir, mask_relative_path)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Nu am găsit fișierul pentru mască: {mask_path}")
        mask_pil = Image.open(mask_path).convert("L")
        mask_pil = self.resize_mask(mask_pil)
        mask_np = np.array(mask_pil)
        if mask_np.max() > 1:
            mask_np = (mask_np > 127).astype(np.int64)
        mask_tensor = torch.from_numpy(mask_np).long()

        # 3) Obține label-ul numeric GS
        gs = int(item.get("GS", 0))

        return image_tensor, mask_tensor, gs

# --- Clasa veche PngDataset (comentată) ---
"""
class PngDataset(Dataset):
    def __init__(self, data_dict, root_dir, split="train", test_ratio=0.2, transform=None, seed=42):
        valid_keys = [key for key, item in data_dict.items() if "slice" in item and "data" in item and "label" in item]
        valid_keys.sort()
        random.seed(seed)
        random.shuffle(valid_keys)
        test_size = int(len(valid_keys) * test_ratio)
        self.keys = valid_keys[test_size:] if split == "train" else valid_keys[:test_size]
        self.data_dict = data_dict
        self.root_dir = root_dir
        self.transform = transform if transform is not None else data_transforms

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        item = self.data_dict[key]
        slices = sorted(item["slice"])
        mid_slice = slices[len(slices) // 2]
        slice_index = item["slice"].index(mid_slice)

        img_types = ["T2W", "ADC", "HBV"]
        all_slices = []
        resize_transform = transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.BILINEAR)

        for img_type in img_types:
            slice_images = []
            for i in range(-1, 2):  # -1, 0, 1 slices
                try:
                    img_relative_path = item["data"][img_type][slice_index + i]
                except IndexError:
                    img_relative_path = item["data"][img_type][slice_index]
                img_path = os.path.join(self.root_dir, img_relative_path)
                image = Image.open(img_path).convert("L")
                image = resize_transform(image)
                image_tensor = transforms.ToTensor()(image)
                slice_images.append(image_tensor)
            all_slices.append(torch.cat(slice_images, dim=0))
        img_tensor = torch.cat(all_slices, dim=0)  # [9, 128, 128]
        mask_relative_path = item["label"][slice_index]
        mask_path = os.path.join(self.root_dir, mask_relative_path)
        mask = Image.open(mask_path).convert("L")
        mask = resize_transform(mask)
        _, mask_tensor = self.transform(mask, mask)
        gs = int(item.get("GS", 0))
        return img_tensor, mask_tensor, gs
"""


# ========== Training with K-Fold Cross Validation ==========
def train_model():
    # Încarcă JSON-ul datasetului.
    with open(json_path, 'r') as f:
        data_dict = json.load(f)
    if "cancer" in data_dict and "non_cancer" in data_dict:
        patients_data = {}
        patients_data.update(data_dict["cancer"])
        patients_data.update(data_dict["non_cancer"])
    else:
        patients_data = data_dict

    if not patients_data:
        print("ERROR: Dataset JSON is empty. Please check your file and required fields.")
        sys.exit(1)

    keys = list(patients_data.keys())
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    best_overall_auroc = 0.0
    best_overall_fold = -1
    best_fold_results = {}
    best_model_state = None
    fold_logs = []
    fold_auroc_per_epoch = [[] for _ in range(num_epochs)]  # Salvează AUROC per epocă per fold

    best_overall_auroc = 0.0
    best_epoch_overall = -1
    best_model_state = None

    fold_index = 1
    for train_index, val_index in kf.split(keys):
        print(f"\n=============== Fold {fold_index}/{num_folds} ===============")
        train_keys = [keys[i] for i in train_index]
        val_keys = [keys[i] for i in val_index]
        train_data = {k: patients_data[k] for k in train_keys}
        val_data = {k: patients_data[k] for k in val_keys}

        # Folosește noul dataset NpyDataset (pentru imagini din .npy)
        train_dataset = NpyDataset(train_data, root_dir, split="train", test_ratio=0.2, seed=42)
        val_dataset = NpyDataset(val_data, root_dir, split="val", test_ratio=0.2, seed=42)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        model = AttentionUNet1024Classifier(n_channels=9, n_seg_classes=3, n_bin_classes=1).to(device)
        criterion_seg = nn.CrossEntropyLoss()
        criterion_cls = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        for epoch in range(num_epochs):
            print(f"Epoch [{epoch + 1}/{num_epochs}]")

            train_loss, train_seg_loss, train_cls_loss = train_epoch(model, device, train_loader, optimizer,
                                                                     criterion_seg, criterion_cls)
            val_loss, val_seg_loss, val_cls_loss, auroc, mean_dice = validate_epoch(model, device, val_loader,
                                                                                    criterion_seg, criterion_cls)

            scheduler.step(val_loss)

            print(f"  Train Loss: {train_loss:.4f} (Seg: {train_seg_loss:.4f}, Cls: {train_cls_loss:.4f})")
            print(f"  Val Loss:   {val_loss:.4f} (Seg: {val_seg_loss:.4f}, Cls: {val_cls_loss:.4f})")
            print(f"  Val AUROC:  {auroc:.4f}, Val Dice: {mean_dice:.4f}")

            # Salvăm AUROC pentru acest fold și epocă
            fold_auroc_per_epoch[epoch].append(auroc)

            save_segmentation_comparison(val_dataset, model, device, fold_index, epoch + 1, results_dir)

        fold_index += 1

    # Calculăm media AUROC pe fiecare epocă după toate fold-urile
    average_auroc_per_epoch = [np.mean(epoch_aurocs) for epoch_aurocs in fold_auroc_per_epoch]

    # Alegem epoca cu cel mai bun AUROC mediu
    best_epoch_overall = np.argmax(average_auroc_per_epoch) + 1  # +1 pentru că epocile sunt 1-indexate
    best_overall_auroc = max(average_auroc_per_epoch)

    print(f"\nFinal decision: Best epoch overall is {best_epoch_overall} with AUROC: {best_overall_auroc:.4f}")
    torch.save(best_model_state, best_model_overall_path)

    # Salvăm informațiile de training într-un fișier text
    with open(results_txt, "w", encoding="utf-8") as f:
        f.write("=== K-Fold Cross Validation Training Summary ===\n")
        f.write(f"Total Folds: {num_folds}\n")
        f.write(f"Best Overall Fold: {best_overall_fold} with AUROC: {best_overall_auroc:.4f}\n\n")
        for fold in fold_logs:
            f.write(f"Fold {fold['fold']}:\n")
            f.write(f"  Best Val AUROC: {fold['best_val_auroc']:.4f}\n")
            f.write(f"  Final Train Loss: {fold['train_loss_history'][-1]:.4f}\n")
            f.write(f"  Final Val Loss:   {fold['val_loss_history'][-1]:.4f}\n")
            f.write(f"  Final Val Dice:   {fold['val_dice_history'][-1]:.4f}\n")
            f.write("\n")
        f.write("Model saved at: " + os.path.abspath(best_model_overall_path) + "\n")
        f.write("Plots saved as train_loss_foldX.png, val_loss_foldX.png, val_auroc_foldX.png, val_dice_foldX.png\n")
        f.write("Segmentation comparison images saved in: " + os.path.abspath(results_dir) + "\n")

    print("\nTraining complete!")
    print(f"Best overall model is from fold {best_overall_fold} with AUROC: {best_overall_auroc:.4f}")
    print("Model and training summary saved.")


if __name__ == "__main__":
    train_model()
