import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from datasets import load_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


CLASS_NAMES = ["normal", "ischemic", "hemorrhagic"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def get_transforms(image_size: int, split: str) -> A.Compose:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


class Tekno21Dataset(Dataset):
    """
    BTX24/tekno21-brain-stroke-dataset-multi HuggingFace 데이터셋 래퍼.
    "train" split만 존재 → 내부적으로 train/val 분리.
    label 매핑: 0=normal, 1=ischemic, 2=hemorrhagic
    """

    def __init__(self, hf_dataset, indices: list, image_size: int, split: str):
        self.dataset = hf_dataset
        self.indices = indices
        self.transform = get_transforms(image_size, split)
        self.label_key = self._detect_label_key()
        self.image_key = self._detect_image_key()

    def _detect_label_key(self) -> str:
        for key in ["label", "labels", "category", "class"]:
            if key in self.dataset.features:
                return key
        raise ValueError(f"Label column을 찾을 수 없습니다. 컬럼: {list(self.dataset.features.keys())}")

    def _detect_image_key(self) -> str:
        for key in ["image", "img", "pixel_values"]:
            if key in self.dataset.features:
                return key
        raise ValueError(f"Image column을 찾을 수 없습니다. 컬럼: {list(self.dataset.features.keys())}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        item = self.dataset[real_idx]
        image = item[self.image_key]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image_np = np.array(image.convert("RGB"))

        tensor = self.transform(image=image_np)["image"]
        label = int(item[self.label_key])
        return tensor, label

    def get_labels(self) -> list:
        return [int(self.dataset[i][self.label_key]) for i in self.indices]

    def get_class_weights(self) -> torch.Tensor:
        labels = self.get_labels()
        counts = np.bincount(labels, minlength=3)
        weights = 1.0 / (counts + 1e-6)
        return torch.tensor(weights / weights.sum(), dtype=torch.float32)

    def get_sampler(self) -> WeightedRandomSampler:
        labels = self.get_labels()
        class_weights = self.get_class_weights().numpy()
        sample_weights = [class_weights[l] for l in labels]
        return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def build_classifier_dataloaders(image_size: int, batch_size: int,
                                  val_ratio: float = 0.15, cache_dir: str = None,
                                  seed: int = 42):
    print("  tekno21 데이터셋 로딩 (train split만 존재 → train/val 자동 분리)...")
    full_ds = load_dataset(
        "BTX24/tekno21-brain-stroke-dataset-multi",
        split="train",
        cache_dir=cache_dir,
    )

    label_key = next(k for k in ["label", "labels", "category", "class"]
                     if k in full_ds.features)
    all_labels = [int(full_ds[i][label_key]) for i in range(len(full_ds))]
    all_indices = list(range(len(full_ds)))

    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=val_ratio,
        stratify=all_labels,
        random_state=seed,
    )

    train_ds = Tekno21Dataset(full_ds, train_idx, image_size, "train")
    val_ds = Tekno21Dataset(full_ds, val_idx, image_size, "val")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=train_ds.get_sampler(),
        num_workers=2, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=False,
    )
    return train_loader, val_loader, train_ds.get_class_weights()
