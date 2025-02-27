import os
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix

# Import modelul (se presupune că returnează două ieșiri: seg_logits și cls_logits)
from architectures.VNet import UNetSegClassifier1024
from dataloader import PngDataset  # folosim clasa de dataset deja definită


def minimal_data_transforms(image, mask):
    print("Aplic transformările minimale pe imagine și mască...")
    w, h = image.size
    new_w, new_h = int(w * 0.5), int(h * 0.5)
    left = (w - new_w) // 2
    upper = (h - new_h) // 2
    right = left + new_w
    lower = upper + new_h
    image = image.crop((left, upper, right, lower))
    mask = mask.crop((left, upper, right, lower))

    image = transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.BILINEAR)(image)
    mask = transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST)(mask)

    image_tensor = transforms.ToTensor()(image)
    mean = image_tensor.mean()
    std = image_tensor.std()
    image_tensor = (image_tensor - mean) / (std + 1e-8)

    mask_np = np.array(mask)
    if mask_np.max() > 1:
        mask_np = (mask_np > 127).astype(np.int64)
    mask_tensor = torch.from_numpy(mask_np).long()

    return image_tensor, mask_tensor


def test_model_minimal(json_path, root_dir, model_path, batch_size=1):
    print("Încep procesul de testare...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Folosesc dispozitivul: {device}")

    print("Încărc datele de testare...")
    with open(json_path, 'r') as f:
        test_data = json.load(f)

    print("Inițializez dataset-ul și DataLoader-ul...")
    test_dataset = PngDataset(data_dict=test_data, root_dir=root_dir, transform=minimal_data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Încărc modelul salvat...")
    model = UNetSegClassifier1024(1, 2, 1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_probs = []
    all_labels = []
    total = 0
    correct = 0

    print("Încep inferența pe datele de test...")
    with torch.no_grad():
        for i, (inputs, _, gs) in enumerate(test_loader):
            print(f"Procesare batch {i + 1}/{len(test_loader)}...")
            inputs = inputs.to(device)
            gs = gs.to(device).float()
            _, cls_logits = model(inputs)
            probs = torch.sigmoid(cls_logits.squeeze(1))
            all_probs.append(probs.cpu())
            all_labels.append(gs.cpu())
            pred_labels = (probs > 0.5).float()
            total += gs.size(0)
            correct += (pred_labels == gs).sum().item()

    print("Finalizat procesul de inferență. Calculăm metricele...")
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    pred_labels = (all_probs > 0.5).astype(np.float32)
    accuracy = correct / total

    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = float('nan')

    precision = precision_score(all_labels, pred_labels, zero_division=0)
    recall = recall_score(all_labels, pred_labels, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(all_labels, pred_labels).ravel()
    specificity = tn / (tn + fp + 1e-7)
    weighted_accuracy = 0.4 * recall + 0.6 * specificity

    print("Test Results:")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    Weighted Accuracy: {weighted_accuracy:.4f}")
    print(f"    AUROC: {auroc:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall (Sensitivity): {recall:.4f}")
    print(f"    Specificity: {specificity:.4f}")

    return {
        "accuracy": accuracy,
        "weighted_accuracy": weighted_accuracy,
        "auroc": auroc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity
    }


if __name__ == "__main__":
    json_path = r"D:\study\facultate\test_cuda\data\output_updated.json"
    root_dir = r"D:\study\facultate\test_cuda"
    model_path = r"D:\study\facultate\test_cuda\results\fold_5\best_model.pth"
    test_model_minimal(json_path, root_dir, model_path, batch_size=16)
