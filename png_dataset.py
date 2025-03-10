import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


def data_transforms(image, mask):
    """
    Applies transformations to the image and mask.
    Resizes the image and mask to 256x256, converts the image to a tensor with normalized values (0-1)
    and converts the mask to a long tensor.
    """
    # Resize image and mask
    resize_img = transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.BILINEAR)
    resize_mask = transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST)
    image = resize_img(image)
    mask = resize_mask(mask)

    # Convert image to tensor
    image_tensor = transforms.ToTensor()(image)

    # Convert mask to numpy, then to a long tensor
    mask_np = np.array(mask)
    # If mask values are in [0, 255], convert them to binary (0 and 1)
    if mask_np.max() > 1:
        mask_np = (mask_np > 127).astype(np.int64)
    mask_tensor = torch.from_numpy(mask_np).long()

    return image_tensor, mask_tensor


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
                    img_relative_path = item["data"][img_type][slice_index]  # Duplicate if out of bounds
                img_path = os.path.join(self.root_dir, img_relative_path)
                image = Image.open(img_path).convert("L")
                image = resize_transform(image)  # Resize to 128x128
                image_tensor = transforms.ToTensor()(image)
                slice_images.append(image_tensor)
            all_slices.append(torch.cat(slice_images, dim=0))

        img_tensor = torch.cat(all_slices, dim=0)  # [9, 128, 128]

        mask_relative_path = item["label"][slice_index]
        mask_path = os.path.join(self.root_dir, mask_relative_path)
        mask = Image.open(mask_path).convert("L")
        mask = resize_transform(mask)  # Resize mask to 128x128
        _, mask_tensor = self.transform(mask, mask)

        gs = int(item.get("GS", 0))

        return img_tensor, mask_tensor, gs
