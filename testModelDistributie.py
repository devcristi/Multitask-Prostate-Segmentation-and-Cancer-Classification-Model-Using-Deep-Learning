import os
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix

# Importă propriul UNet definit în D:\study\facultate\test_cuda\architectures\UNet.py
from architectures.UNet import UNet
from png_dataset import PngDataset  # folosim clasa de dataset deja definită

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

def test_model_minimal(test_data, root_dir, model_path, batch_size=1):
    print("Încep procesul de testare...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Folosesc dispozitivul: {device}")

    print("Inițializez dataset-ul și DataLoader-ul...")
    test_dataset = PngDataset(data_dict=test_data, root_dir=root_dir, transform=minimal_data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Încărc modelul salvat...")
    # Inițializează UNet-ul propriu cu 1 canal de intrare și 1 canal de ieșire.
    model = UNet(in_channels=1, out_channels=1).to(device)
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

            # Rulează modelul: UNet-ul nostru returnează o ieșire segmentată.
            outputs = model(inputs)
            # Pentru scopuri de clasificare, agregăm ieșirea spațial pe fiecare imagine
            # (de exemplu, media valorilor pe dimensiunile spațiale) pentru a obține un singur logits per imagine.
            cls_logits = torch.mean(outputs, dim=[2, 3])
            probs = torch.sigmoid(cls_logits).squeeze(1)
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

def load_json(path):
    """
    Încarcă un fișier JSON de la calea specificată.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def merge_datasets(random_dataset, cancer_dataset):
    """
    Combină două seturi de date (sub formă de dicționare).
    Se presupune că cheile din fiecare dicționar sunt unice.
    """
    merged = {}
    merged.update(random_dataset)
    merged.update(cancer_dataset)
    return merged

if __name__ == "__main__":
    # Căile către fișierele dataseturilor
    # Random dataset (format JSON) generat de scriptul data_random.py
    random_dataset_path = r"D:\study\facultate\test_cuda\data\output9CanaleV3\data_random.json"
    cancer_dataset_path = r"D:\study\facultate\test_cuda\Bosma22a_segmentation_slices.json"

    # Directorul rădăcină din care sunt accesate imaginile
    root_dir = r"D:\study\facultate\test_cuda"
    # Calea către modelul antrenat (care a fost antrenat cu UNet-ul din architectures\UNet.py)
    model_path = r"D:\study\facultate\test_cuda\results\fold_5\best_model.pth"

    print("Încărc seturile de date...")
    random_dataset = load_json(random_dataset_path)
    cancer_dataset = load_json(cancer_dataset_path)

    print("Combin seturile de date...")
    combined_dataset = merge_datasets(random_dataset, cancer_dataset)

    results = test_model_minimal(combined_dataset, root_dir, model_path, batch_size=16)

    print("Rezultatele testării:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")