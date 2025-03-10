import os
import json
import torch
from torch.utils.data import DataLoader
from png_dataset import PngDataset, data_transforms


def main():
    # Set the root directory for your PNG data
    root_dir = r"D:\study\facultate\test_cuda\data\output9CanaleV3"

    # Assume that the JSON file containing your dataset information is located in the same directory
    json_path = os.path.join(root_dir, "data.json")

    # Check if the JSON file exists
    if not os.path.exists(json_path):
        print(f"JSON file not found at {json_path}")
        return

    # Load dataset details from JSON file
    with open(json_path, "r", encoding="utf-8") as f:
        data_dict = json.load(f)

    # Initialize the dataset with your defined transforms
    dataset = PngDataset(data_dict, root_dir, transform=data_transforms)

    # Create a DataLoader for batching and shuffling
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Iterate over the DataLoader and process batches
    for i, (images, masks, gs) in enumerate(dataloader):
        print(f"Batch {i + 1}:")
        print("Images shape:", images.shape)
        print("Masks shape:", masks.shape)
        print("Classification labels (GS):", gs)
        # Process each batch through your UNet model or any further processing steps as needed
        # For demonstration, we break after a few batches
        if i >= 2:
            break


if __name__ == '__main__':
    main()