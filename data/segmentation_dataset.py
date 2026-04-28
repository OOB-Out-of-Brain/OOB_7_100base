import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_seg_transforms(image_size: int, split: str) -> A.Compose:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    additional_targets = {"mask": "mask"}

    if split == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.4),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.5),
            A.ElasticTransform(alpha=120, sigma=6, alpha_affine=3, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            A.GaussNoise(var_limit=(5, 30), p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ], additional_targets=additional_targets)
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ], additional_targets=additional_targets)


class AISDataset(Dataset):
    """
    AISD 뇌 허혈성 뇌졸중 분할 데이터셋.
    https://github.com/GriffinLiang/AISD

    폴더 구조:
        data/raw/aisd/
            images/  (CT PNG 파일들)
            masks/   (분할 마스크 PNG 파일들, 병변=255, 배경=0)
    """

    def __init__(self, root: str, split: str = "train", image_size: int = 256,
                 val_ratio: float = 0.15):
        self.root = Path(root)
        self.image_size = image_size
        self.transform = get_seg_transforms(image_size, split)

        images_dir = self.root / "images"
        masks_dir = self.root / "masks"

        if not images_dir.exists():
            raise FileNotFoundError(
                f"AISD 이미지 폴더가 없습니다: {images_dir}\n"
                "scripts/download_data.py를 먼저 실행하세요."
            )

        all_files = sorted([
            f.stem for f in images_dir.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        ])

        n_val = max(1, int(len(all_files) * val_ratio))
        if split == "train":
            self.file_stems = all_files[n_val:]
        else:
            self.file_stems = all_files[:n_val]

        self.images_dir = images_dir
        self.masks_dir = masks_dir

    def __len__(self) -> int:
        return len(self.file_stems)

    def _find_file(self, directory: Path, stem: str) -> Path:
        for ext in [".png", ".PNG", ".jpg", ".jpeg"]:
            p = directory / (stem + ext)
            if p.exists():
                return p
        raise FileNotFoundError(f"{directory}/{stem}.*  파일을 찾을 수 없습니다.")

    def __getitem__(self, idx: int):
        stem = self.file_stems[idx]

        image_path = self._find_file(self.images_dir, stem)
        mask_path = self._find_file(self.masks_dir, stem)

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        mask = (mask > 127).astype(np.float32)

        transformed = self.transform(image=image, mask=mask)
        image_tensor = transformed["image"]
        mask_tensor = transformed["mask"].unsqueeze(0)

        return image_tensor, mask_tensor


def build_segmentation_dataloaders(aisd_root: str, image_size: int, batch_size: int,
                                    val_ratio: float = 0.15):
    train_ds = AISDataset(aisd_root, split="train", image_size=image_size, val_ratio=val_ratio)
    val_ds = AISDataset(aisd_root, split="val", image_size=image_size, val_ratio=val_ratio)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=False,
    )
    return train_loader, val_loader
