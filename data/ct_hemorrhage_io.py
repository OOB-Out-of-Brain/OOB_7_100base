from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


@dataclass(frozen=True)
class NiftiSliceRef:
    path: Path
    slice_index: int


def diagnosis_csv_path(data_root: str | Path) -> Path:
    root = Path(data_root)
    csv_path = root / "hemorrhage_diagnosis_raw_ct.csv"
    if csv_path.exists():
        return csv_path

    legacy_csv_path = root / "hemorrhage_diagnosis.csv"
    if legacy_csv_path.exists():
        return legacy_csv_path

    raise FileNotFoundError(
        "CT Hemorrhage diagnosis CSV not found. Expected "
        f"{csv_path} or {legacy_csv_path}"
    )


def read_diagnosis(data_root: str | Path) -> pd.DataFrame:
    df = pd.read_csv(diagnosis_csv_path(data_root))
    df.columns = df.columns.str.strip()
    return df


def has_nifti_layout(data_root: str | Path) -> bool:
    root = Path(data_root)
    return (root / "ct_scans").is_dir()


def legacy_image_path(data_root: str | Path, patient_id: int, slice_number: int) -> Path:
    root = Path(data_root)
    pid_str = str(int(patient_id)).zfill(3)
    return root / "Patients_CT" / pid_str / "brain" / f"{int(slice_number)}.jpg"


def legacy_mask_path(data_root: str | Path, patient_id: int, slice_number: int) -> Path:
    root = Path(data_root)
    pid_str = str(int(patient_id)).zfill(3)
    return root / "Patients_CT" / pid_str / "brain" / f"{int(slice_number)}_HGE_Seg.jpg"


def nifti_image_ref(data_root: str | Path, patient_id: int, slice_index: int) -> NiftiSliceRef:
    root = Path(data_root)
    pid_str = str(int(patient_id)).zfill(3)
    return NiftiSliceRef(root / "ct_scans" / f"{pid_str}.nii", int(slice_index))


def nifti_mask_ref(data_root: str | Path, patient_id: int, slice_index: int) -> NiftiSliceRef:
    root = Path(data_root)
    pid_str = str(int(patient_id)).zfill(3)
    return NiftiSliceRef(root / "masks" / f"{pid_str}.nii", int(slice_index))


def ref_exists(ref: Path | NiftiSliceRef) -> bool:
    return ref.path.exists() if isinstance(ref, NiftiSliceRef) else ref.exists()


def ref_name(ref: Path | NiftiSliceRef) -> str:
    if isinstance(ref, NiftiSliceRef):
        return f"{ref.path.stem}_slice{ref.slice_index + 1:03d}.png"
    return ref.name


@lru_cache(maxsize=64)
def _load_nifti(path: str):
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("nibabel is required to read CT Hemorrhage v1.3.1 NIfTI files.") from exc
    return nib.load(path)


def _window_ct(slice_image: np.ndarray, window_level: float = 40.0, window_width: float = 120.0) -> np.ndarray:
    w_min = window_level - window_width / 2
    w_max = window_level + window_width / 2
    windowed = np.clip(slice_image, w_min, w_max)
    windowed = (windowed - w_min) * (255.0 / (w_max - w_min))
    return windowed.astype(np.uint8)


def load_ct_image(ref: Path | NiftiSliceRef) -> np.ndarray:
    if isinstance(ref, NiftiSliceRef):
        nifti = _load_nifti(str(ref.path))
        image = np.asarray(nifti.dataobj[:, :, ref.slice_index], dtype=np.float32)
        return np.array(Image.fromarray(_window_ct(image)).convert("RGB"))

    return np.array(Image.open(ref).convert("RGB"))


def load_ct_mask(ref: Path | NiftiSliceRef) -> np.ndarray:
    if isinstance(ref, NiftiSliceRef):
        nifti = _load_nifti(str(ref.path))
        mask = np.asarray(nifti.dataobj[:, :, ref.slice_index], dtype=np.float32)
        return (mask > 0).astype(np.float32)

    mask = np.array(Image.open(ref).convert("L"))
    return (mask > 127).astype(np.float32)
