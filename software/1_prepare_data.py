"""
Step 1: Prepare the WM-811K wafer map dataset for training.

Reads LSWMD.pkl and extracts labeled wafer maps into:
    data/train/<class_name>/*.png
    data/val/<class_name>/*.png

Usage:
    python software/1_prepare_data.py
"""
import pickle
import numpy as np
import cv2
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import IMAGE_SIZE, CLASS_NAMES

PKL_PATH = ROOT / "LSWMD.pkl"
DATA_DIR = ROOT / "data"


# --- Pickle compat for old pandas format ---
class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        remap = {
            "pandas.indexes.base": "pandas.core.indexes.base",
            "pandas.indexes.numeric": "pandas.core.indexes.numeric",
            "pandas.indexes.range": "pandas.core.indexes.range",
            "pandas.indexes.multi": "pandas.core.indexes.multi",
        }
        return super().find_class(remap.get(module, module), name)


def main():
    assert PKL_PATH.exists(), f"Dataset not found: {PKL_PATH}"

    # Load pickle
    print(f"Loading {PKL_PATH.name} ...")
    with open(PKL_PATH, "rb") as f:
        df = CompatUnpickler(f, encoding="latin1").load()
    print(f"  {len(df)} total wafer maps")

    # Build label lookup (case-insensitive: 'none' → 'None')
    name_to_idx = {n.lower(): i for i, n in enumerate(CLASS_NAMES)}

    # Extract labeled maps
    labeled = []
    for i in range(len(df)):
        row = df.iloc[i]
        ft, wm = row.get("failureType"), row.get("waferMap")
        if ft is None or wm is None:
            continue
        if isinstance(ft, np.ndarray) and ft.size > 0:
            label_str = str(ft.flatten()[0]).strip().lower()
        else:
            continue
        if label_str not in name_to_idx:
            continue
        if not isinstance(wm, np.ndarray) or wm.ndim != 2 or min(wm.shape) < 5:
            continue
        labeled.append((wm, name_to_idx[label_str]))

    print(f"  {len(labeled)} labeled maps found")
    counts = Counter(l for _, l in labeled)
    for idx, cnt in sorted(counts.items()):
        print(f"    {CLASS_NAMES[idx]:12s}: {cnt}")

    # Save as images (80/20 train/val split, max 2000 per class)
    by_class = {}
    for wm, label in labeled:
        by_class.setdefault(label, []).append(wm)

    for split in ("train", "val"):
        for cls in CLASS_NAMES:
            (DATA_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    for label, maps in by_class.items():
        rng = np.random.RandomState(42 + label)
        rng.shuffle(maps)
        maps = maps[:2000]
        n_train = int(len(maps) * 0.8)
        for i, wm in enumerate(maps):
            # Convert wafer map to grayscale image
            img = np.zeros(wm.shape, dtype=np.uint8)
            img[wm == 1] = 128   # normal die
            img[wm == 2] = 255   # defect
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            split = "train" if i < n_train else "val"
            path = DATA_DIR / split / CLASS_NAMES[label] / f"{i:04d}.png"
            cv2.imwrite(str(path), img)
        print(f"  {CLASS_NAMES[label]:12s}: {n_train} train + {len(maps)-n_train} val")

    print(f"\nDone! Dataset saved to {DATA_DIR}/")


if __name__ == "__main__":
    main()
