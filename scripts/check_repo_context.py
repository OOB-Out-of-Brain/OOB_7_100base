#!/usr/bin/env python3
"""Verify that commands are being run from the canonical OOB_7_100base repo."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


LOCAL_CANONICAL_ROOT = Path("/Users/pke03/OOB_7_100base")
DEPRECATED_LOCAL_ROOTS = [
    Path("/Users/pke03/OOB_test_7_epoch100-main"),
]


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve_from_root(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cwd = Path.cwd().resolve()
    errors: list[str] = []
    warnings: list[str] = []

    if LOCAL_CANONICAL_ROOT.exists() and root != LOCAL_CANONICAL_ROOT.resolve():
        errors.append(f"script is running from {root}, expected {LOCAL_CANONICAL_ROOT}")

    if cwd != root:
        errors.append(f"current working directory is {cwd}; run from repo root {root}")

    for deprecated_root in DEPRECATED_LOCAL_ROOTS:
        if deprecated_root.exists() and _is_relative_to(cwd, deprecated_root.resolve()):
            errors.append(f"current working directory is deprecated baseline repo: {deprecated_root}")

    for required in ("AGENTS.md", "CLAUDE.md", "config.yaml"):
        if not (root / required).exists():
            errors.append(f"required repo file missing: {required}")

    config_path = root / "config.yaml"
    if config_path.exists():
        with config_path.open() as f:
            cfg = yaml.safe_load(f)
        ct_root = _resolve_from_root(root, cfg["data"]["ct_hemorrhage_path"])
        if "1.3.1" not in ct_root.name:
            errors.append(f"CT Hemorrhage path should point to v1.3.1, got {ct_root}")
        if not (ct_root / "hemorrhage_diagnosis_raw_ct.csv").exists():
            errors.append(f"CT v1.3.1 CSV missing: {ct_root / 'hemorrhage_diagnosis_raw_ct.csv'}")
        if not (ct_root / "ct_scans").is_dir():
            errors.append(f"CT NIfTI image folder missing: {ct_root / 'ct_scans'}")
        if not (ct_root / "masks").is_dir():
            warnings.append(f"CT NIfTI mask folder missing: {ct_root / 'masks'}")

    if errors:
        print("[FAIL] Repo context check failed:")
        for error in errors:
            print(f"  - {error}")
        print("\nUse:")
        print("  cd /Users/pke03/OOB_7_100base")
        print("  ./venv/bin/python scripts/check_repo_context.py")
        return 1

    print("[OK] Canonical repo context")
    print(f"  root: {root}")
    if warnings:
        print("  warnings:")
        for warning in warnings:
            print(f"    - {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
