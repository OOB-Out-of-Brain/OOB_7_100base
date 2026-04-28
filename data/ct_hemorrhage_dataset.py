import pandas as pd
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

CT_CLASS_NAMES = ["normal", "hemorrhagic"]

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def _cls_transforms(image_size, split):
    if split == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


def _seg_transforms(image_size, split):
    extra = {"mask": "mask"}
    if split == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            A.GaussNoise(var_limit=(5, 30), p=0.3),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ], additional_targets=extra)
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ], additional_targets=extra)


class CTClassifierDataset(Dataset):
    def __init__(self, samples, image_size, split):
        self.samples = samples  # list of (Path, label_int)
        self.transform = _cls_transforms(image_size, split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        return self.transform(image=image)["image"], label

    def get_sampler(self):
        labels = [s[1] for s in self.samples]
        counts = np.bincount(labels, minlength=2)
        weights = 1.0 / (counts + 1e-6)
        sample_weights = [weights[l] for l in labels]
        return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


class CTSegDataset(Dataset):
    def __init__(self, samples, image_size, split):
        self.samples = samples  # list of (img_path, mask_path)
        self.transform = _seg_transforms(image_size, split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.float32)
        out = self.transform(image=image, mask=mask)
        return out["image"], out["mask"].unsqueeze(0)


def _collect_samples(data_root: str):
    root = Path(data_root)
    diag_csv = root / "hemorrhage_diagnosis.csv"
    patients_dir = root / "Patients_CT"

    df = pd.read_csv(diag_csv)
    df.columns = df.columns.str.strip()

    cls_samples, seg_samples = [], []
    for _, row in df.iterrows():
        pid_int = int(row["PatientNumber"])
        pid_str = str(pid_int).zfill(3)
        slice_num = int(row["SliceNumber"])
        label = 0 if int(row["No_Hemorrhage"]) == 1 else 1

        img_path = patients_dir / pid_str / "brain" / f"{slice_num}.jpg"
        if not img_path.exists():
            continue
        cls_samples.append((img_path, label, pid_int))

        mask_path = patients_dir / pid_str / "brain" / f"{slice_num}_HGE_Seg.jpg"
        if mask_path.exists():
            seg_samples.append((img_path, mask_path, pid_int))

    return cls_samples, seg_samples


def _patient_split(samples, val_ratio, seed):
    patient_ids = sorted({s[2] for s in samples})
    rng = np.random.RandomState(seed)
    rng.shuffle(patient_ids)
    n_val = max(1, int(len(patient_ids) * val_ratio))
    val_set = set(patient_ids[:n_val])
    train = [(p, x) for p, x, pid in samples if pid not in val_set]
    val   = [(p, x) for p, x, pid in samples if pid in val_set]
    return train, val


def build_ct_classifier_dataloaders(data_root, image_size, batch_size,
                                     val_ratio=0.2, seed=42):
    cls_samples, _ = _collect_samples(data_root)
    train_s, val_s = _patient_split(cls_samples, val_ratio, seed)

    train_ds = CTClassifierDataset(train_s, image_size, "train")
    val_ds   = CTClassifierDataset(val_s,   image_size, "val")

    labels = [s[1] for s in train_s]
    counts = np.bincount(labels, minlength=2)
    class_weights = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32)
    class_weights = class_weights / class_weights.sum()

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=train_ds.get_sampler(),
                              num_workers=2, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=False)
    return train_loader, val_loader, class_weights


def _collect_bhsd_seg(processed_dir: str):
    """BHSD 전처리된 (image_png, mask_png, patient_id) 리스트."""
    import csv
    root = Path(processed_dir)
    idx_csv = root / "index.csv"
    if not idx_csv.exists():
        return []
    samples = []
    with open(idx_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_p = root / row["image_path"]
            msk_p = root / row["mask_path"]
            stem = img_p.stem
            pid = stem.rsplit("_s", 1)[0]
            samples.append((img_p, msk_p, pid))
    return samples


def build_ct_seg_dataloaders(data_root, image_size, batch_size,
                              val_ratio=0.2, seed=42,
                              bhsd_processed_dir="./data/processed/bhsd"):
    _, seg_samples = _collect_samples(data_root)
    train_s, val_s = _patient_split(seg_samples, val_ratio, seed)

    # BHSD 세그멘터 샘플도 추가
    bhsd_samples = _collect_bhsd_seg(bhsd_processed_dir)
    if bhsd_samples:
        b_pids = sorted({s[2] for s in bhsd_samples})
        rng = np.random.RandomState(seed + 1)
        rng.shuffle(b_pids)
        n_val_b = max(1, int(len(b_pids) * val_ratio))
        val_pids = set(b_pids[:n_val_b])
        b_train = [(s[0], s[1]) for s in bhsd_samples if s[2] not in val_pids]
        b_val   = [(s[0], s[1]) for s in bhsd_samples if s[2] in val_pids]
        train_s = train_s + b_train
        val_s   = val_s + b_val
        print(f"  BHSD 세그 샘플 추가: train={len(b_train)}, val={len(b_val)}")

    print(f"  세그 합계: train={len(train_s)}, val={len(val_s)}")

    train_ds = CTSegDataset(train_s, image_size, "train")
    val_ds   = CTSegDataset(val_s,   image_size, "val")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=False)
    return train_loader, val_loader
