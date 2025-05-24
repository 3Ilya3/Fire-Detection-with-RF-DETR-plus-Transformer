import os
import json
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class FireSmokeDataset(Dataset):
    def __init__(self, json_path, data_dir, transform=None, reverse_prob=0.5):

        with open(json_path, 'r') as f:
            self.annotations = json.load(f)
            
        self.data_dir = data_dir
        self.transform = transform
        self.reverse_prob = reverse_prob 

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get sequence info
        seq_info = self.annotations[idx]
        seq_name = seq_info["seq"]
        label = seq_info["label"] 

        # Load 30 frames
        seq_path = os.path.join(self.data_dir, seq_name)
        frames = []
        for i in range(30):
            frame_path = os.path.join(seq_path, f"frame_{i:03d}.jpg")
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frames.append(frame)

        # Reverse sequence 
        if self.reverse_prob > 0 and np.random.rand() < self.reverse_prob:
            frames = frames[::-1]

        # Apply transforms
        if self.transform:
            augmented_frames = [self.transform(image=frame)["image"] for frame in frames]
            frames = torch.stack(augmented_frames)  # [30, 3, 224, 224]
        else:
            # Default transform: just to tensor
            default_transform = A.Compose([ToTensorV2()])
            frames = torch.stack([default_transform(image=frame)["image"] for frame in frames])

        return frames, torch.tensor(label, dtype=torch.long)

# Define transforms
train_transform = A.Compose([
    A.RandomCrop(height=200, width=200, p=0.3),
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Create datasets
def get_dataloaders(working_dir, batch_size=4, num_workers=4):
    train_dataset = FireSmokeDataset(
        json_path=os.path.join(working_dir, "train", "annotations.json"),
        data_dir=os.path.join(working_dir, "train"),
        transform=train_transform
    )
    test_dataset = FireSmokeDataset(
        json_path=os.path.join(working_dir, "test", "annotations.json"),
        data_dir=os.path.join(working_dir, "test"),
        transform=test_transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print("Datasets created...")

    return train_loader, test_loader