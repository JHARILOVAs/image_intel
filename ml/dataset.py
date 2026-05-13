import os
import sys

from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class IntelDataset(Dataset):
    """Custom PyTorch Dataset for Intel Image Classification data."""

    def __init__(self, folder, transform=None):
        self.folder    = folder
        self.transform = transform
        self.classes   = sorted(os.listdir(folder))
        self.images    = []
        self.labels    = []

        for idx, cls in enumerate(self.classes):
            cls_dir = os.path.join(folder, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(cls_dir, fname))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


def get_dataloaders(batch_size=32, data_root=None):
    """
    Build train / val / test DataLoaders.

    Parameters
    ----------
    batch_size : int
    data_root  : str | None
        Root that contains archive/seg_train and archive/seg_test.
        Defaults to the project root (two levels above this file).
    """
    if data_root is None:
        # ml/dataset.py → ml/ → project_root
        data_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    train_dir = os.path.join(data_root, 'archive', 'seg_train', 'seg_train')
    test_dir  = os.path.join(data_root, 'archive', 'seg_test',  'seg_test')

    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f"Training data not found at: {train_dir}\n"
            "Download the Intel Image Classification dataset from Kaggle and place it at\n"
            "<project_root>/archive/seg_train/seg_train  (and seg_test).\n"
            "Or pass --data_dir <path> to main.py."
        )

    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_full   = IntelDataset(train_dir, transform=transform_train)
    test_dataset = IntelDataset(test_dir,  transform=transform_test)

    val_size      = int(0.15 * len(train_full))
    train_size    = len(train_full) - val_size
    train_dataset, val_dataset = random_split(train_full, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Classes : {train_full.classes}")
    print(f"Train   : {len(train_dataset)} | Val : {len(val_dataset)} | Test : {len(test_dataset)}")

    return train_loader, val_loader, test_loader
