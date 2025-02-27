import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class JointTransform:
    """
    Aplică transformări comune (crop pe centru și rotație aleatorie)
    atât pe imagine cât și pe mască.
    """

    def __init__(self, crop_factor=0.5, rotation_degree=10):
        self.crop_factor = crop_factor
        self.rotation_degree = rotation_degree

    def __call__(self, image, mask):
        image, mask = self.center_crop(image, mask, self.crop_factor)
        angle = random.uniform(-self.rotation_degree, self.rotation_degree)
        image = TF.rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)
        return image, mask

    def center_crop(self, image, mask, crop_factor):
        w, h = image.size
        new_w, new_h = int(w * crop_factor), int(h * crop_factor)
        left = (w - new_w) // 2
        upper = (h - new_h) // 2
        right = left + new_w
        lower = upper + new_h
        return image.crop((left, upper, right, lower)), mask.crop((left, upper, right, lower))


def data_transforms(image, mask):
    """
    Aplica transformările:
      - Crop pe centru și rotație aleatorie (±10°)
      - Redimensionare la 128x128
      - Conversie la tensor; pentru imagine se aplică normalizare z-score;
        pentru mască se convertește la tipul Long.
    """
    # Transformări comune
    joint_transform = JointTransform(crop_factor=0.6, rotation_degree=10)
    image, mask = joint_transform(image, mask)

    # Redimensionare (folosim interpolare biliniară pentru imagine și nearest pentru mască)
    image = transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.BILINEAR)(image)
    mask = transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST)(mask)

    # Imagine: conversie la tensor și normalizare z-score
    image_tensor = transforms.ToTensor()(image)
    mean = image_tensor.mean()
    std = image_tensor.std()
    image_tensor = (image_tensor - mean) / (std + 1e-8)

    # Mască: conversie la numpy, eventual binarizare și la tensor (tip Long)
    mask_np = np.array(mask)
    if mask_np.max() > 1:
        mask_np = (mask_np > 127).astype(np.int64)
    mask_tensor = torch.from_numpy(mask_np).long()

    return image_tensor, mask_tensor


class PngDataset(Dataset):
    """
    Dataset pentru segmentare și clasificare.
    Fiecare intrare din JSON conține:
      - 'data': calea relativă către imagine,
      - 'label': calea relativă către mască,
      - 'GS': eticheta de clasificare.
    """

    def __init__(self, data_dict, root_dir, transform=None):
        """
        data_dict: dicționarul încărcat din JSON.
        root_dir: directorul de bază.
        transform: funcție care primește (image, mask) și returnează
                   (image_tensor, mask_tensor).
        """
        self.data_dict = data_dict
        self.root_dir = root_dir
        self.keys = list(data_dict.keys())
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        item = self.data_dict[key]

        # Construiește calea completă
        img_path = os.path.join(self.root_dir, item["data"])
        mask_path = os.path.join(self.root_dir, item["label"])

        # Eticheta de clasificare
        gs = int(item["GS"])

        # Încarcă imaginea și masca (convertite în grayscale)
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image_tensor, mask_tensor = self.transform(image, mask)
        else:
            image_tensor = transforms.ToTensor()(image)
            mean = image_tensor.mean()
            std = image_tensor.std()
            image_tensor = (image_tensor - mean) / (std + 1e-8)
            mask_np = np.array(mask)
            if mask_np.max() > 1:
                mask_np = (mask_np > 127).astype(np.int64)
            mask_tensor = torch.from_numpy(mask_np).long()

        return image_tensor, mask_tensor, gs
