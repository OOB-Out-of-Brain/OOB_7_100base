"""
CT Hemorrhage + tekno21 데이터셋 결합 (2-class: normal / hemorrhagic).

tekno21 라벨 매핑 (실제):
  0 = Kanama (출혈) → hemorrhagic (1)
  1 = iskemi (허혈) → 제외
  2 = İnme Yok (정상) → normal (0)

CT Hemorrhage:
  No_Hemorrhage=1 → normal (0)
  No_Hemorrhage=0 → hemorrhagic (1)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from datasets import load_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


CLASS_NAMES = ["normal", "hemorrhagic"]
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def _transforms(image_size: int, split: str) -> A.Compose:
    if split == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.GaussNoise(p=0.3),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


class CombinedDataset(Dataset):
    """
    samples: list of tuples (source, ref, label)
      "ct"   : ref = Path(img)
      "tk"   : ref = HF index
      "bhsd" : ref = Path(img)   # 전처리된 PNG 슬라이스
    """

    def __init__(self, samples, hf_dataset, image_size, split):
        self.samples = samples
        self.hf = hf_dataset
        self.transform = _transforms(image_size, split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        source, ref, label = self.samples[idx]
        if source == "ct" or source == "bhsd":
            image = np.array(Image.open(ref).convert("RGB"))
        else:  # tekno21
            item = self.hf[ref]
            img = item["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            image = np.array(img.convert("RGB"))
        tensor = self.transform(image=image)["image"]
        return tensor, label

    def get_labels(self):
        return [s[2] for s in self.samples]

    def get_sampler(self):
        labels = self.get_labels()
        counts = np.bincount(labels, minlength=2)
        weights = 1.0 / (counts + 1e-6)
        sample_weights = [weights[l] for l in labels]
        return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def _collect_ct(data_root):
    root = Path(data_root)
    df = pd.read_csv(root / "hemorrhage_diagnosis.csv")
    df.columns = df.columns.str.strip()

    samples = []
    for _, row in df.iterrows():
        pid_int = int(row["PatientNumber"])
        pid_str = str(pid_int).zfill(3)
        slice_num = int(row["SliceNumber"])
        label = 0 if int(row["No_Hemorrhage"]) == 1 else 1
        img_path = root / "Patients_CT" / pid_str / "brain" / f"{slice_num}.jpg"
        if img_path.exists():
            samples.append(("ct", img_path, label, pid_int))
    return samples


def _collect_bhsd(processed_dir: str = "./data/processed/bhsd"):
    """BHSD 전처리된 슬라이스 index.csv 로드.
    모든 슬라이스는 hemorrhagic (label=1). 환자 ID로 split 가능하게 반환.
    반환: [(source, img_path, label, patient_id_str), ...]"""
    root = Path(processed_dir)
    idx_csv = root / "index.csv"
    if not idx_csv.exists():
        print(f"  ⚠️ BHSD index 없음: {idx_csv} (preprocess_bhsd.py 먼저 실행)")
        return []
    samples = []
    import csv
    with open(idx_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = root / row["image_path"]
            # 파일명에서 환자 ID 추출 (ID_xxx_ID_yyy_sNNN.png → ID_xxx_ID_yyy)
            stem = img_path.stem
            pid = stem.rsplit("_s", 1)[0]
            samples.append(("bhsd", img_path, 1, pid))  # 모두 hemorrhagic
    return samples


def _collect_tekno21(cache_dir):
    """tekno21 로드 후 (source, hf_idx, label, tekno21_idx) 리스트 반환.
    0=Kanama(출혈)→1, 2=İnme Yok(정상)→0, 1=iskemi는 제외."""
    ds = load_dataset(
        "BTX24/tekno21-brain-stroke-dataset-multi",
        split="train",
        cache_dir=cache_dir,
    )
    remap = {0: 1, 2: 0}  # Kanama→hemorrhagic, İnme Yok→normal
    samples = []
    for i in range(len(ds)):
        orig = int(ds[i]["label"])
        if orig not in remap:
            continue
        samples.append(("tk", i, remap[orig], -1))  # tekno21엔 환자ID 없어서 -1
    return ds, samples


def build_combined_dataloaders(ct_root, tekno21_cache, image_size, batch_size,
                                val_ratio=0.2, seed=42, num_workers=2,
                                bhsd_processed_dir="./data/processed/bhsd"):
    print("  CT Hemorrhage 로딩...")
    ct_all = _collect_ct(ct_root)
    # CT는 환자 단위 분리 (leakage 방지)
    ct_patients = sorted({s[3] for s in ct_all})
    rng = np.random.RandomState(seed)
    rng.shuffle(ct_patients)
    n_val_pat = max(1, int(len(ct_patients) * val_ratio))
    val_pat_set = set(ct_patients[:n_val_pat])
    ct_train = [(s[0], s[1], s[2]) for s in ct_all if s[3] not in val_pat_set]
    ct_val   = [(s[0], s[1], s[2]) for s in ct_all if s[3] in val_pat_set]

    print("  tekno21 로딩...")
    hf, tk_all = _collect_tekno21(tekno21_cache)
    # tekno21은 stratified split
    tk_labels = [s[2] for s in tk_all]
    tk_idx = list(range(len(tk_all)))
    tk_train_i, tk_val_i = train_test_split(
        tk_idx, test_size=val_ratio, stratify=tk_labels, random_state=seed
    )
    tk_train = [(tk_all[i][0], tk_all[i][1], tk_all[i][2]) for i in tk_train_i]
    tk_val   = [(tk_all[i][0], tk_all[i][1], tk_all[i][2]) for i in tk_val_i]

    print("  BHSD 로딩...")
    bhsd_all = _collect_bhsd(bhsd_processed_dir)
    # BHSD는 환자(볼륨) 단위 split
    bhsd_pids = sorted({s[3] for s in bhsd_all})
    rng_b = np.random.RandomState(seed + 1)
    rng_b.shuffle(bhsd_pids)
    n_val_b = max(1, int(len(bhsd_pids) * val_ratio))
    bhsd_val_set = set(bhsd_pids[:n_val_b])
    bhsd_train = [(s[0], s[1], s[2]) for s in bhsd_all if s[3] not in bhsd_val_set]
    bhsd_val   = [(s[0], s[1], s[2]) for s in bhsd_all if s[3] in bhsd_val_set]

    train_samples = ct_train + tk_train + bhsd_train
    val_samples   = ct_val + tk_val + bhsd_val

    # 통계 출력
    def _dist(ss):
        c = np.bincount([s[2] for s in ss], minlength=2)
        return f"normal={c[0]}, hemorrhagic={c[1]}"
    print(f"  학습 세트: {len(train_samples)}개 ({_dist(train_samples)})")
    print(f"  검증 세트: {len(val_samples)}개 ({_dist(val_samples)})")

    train_ds = CombinedDataset(train_samples, hf, image_size, "train")
    val_ds   = CombinedDataset(val_samples,   hf, image_size, "val")

    labels = train_ds.get_labels()
    counts = np.bincount(labels, minlength=2)
    class_weights = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32)
    class_weights = class_weights / class_weights.sum()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=train_ds.get_sampler(),
        num_workers=num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=False,
    )
    return train_loader, val_loader, class_weights
