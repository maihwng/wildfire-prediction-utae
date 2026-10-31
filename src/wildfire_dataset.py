"""
wildfire_dataset.py
-------------------
PyTorch Dataset and DataLoader for Spatio-Temporal Wildfire Spread Prediction.
Reads 17 dynamic continuous and mask channels from preprocessed .npz files and index.csv.
"""

import os
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# Complete list of 17 input channels in order
CHANNEL_NAMES = [
    "pr",               # 0: Precipitation
    "tmmx",             # 1: Max Temperature
    "tmmn",             # 2: Min Temperature
    "vs",               # 3: Wind Speed
    "th_sin",           # 4: Wind Direction (sin)
    "th_cos",           # 5: Wind Direction (cos)
    "rmin",             # 6: Min Relative Humidity
    "rmax",             # 7: Max Relative Humidity
    "vpd",              # 8: Vapor Pressure Deficit
    "fm100",            # 9: 100-hour Fuel Moisture
    "erc",              # 10: Energy Release Component
    "NDVI",             # 11: Normalized Difference Vegetation Index
    "EVI",              # 12: Enhanced Vegetation Index
    "fire",             # 13: Binary Active Fire Mask (Previous days)
    "log1p_frp",        # 14: Log-transformed Fire Radiative Power
    "weather_valid",    # 15: Weather Quality Mask (1=finite original)
    "vegetation_valid", # 16: Vegetation Quality Mask (1=finite original)
]


class WildfireDataset(Dataset):
    """
    Dataset for next-day wildfire prediction using spatio-temporal satellite sequences.
    
    Each sample consists of:
      - x: Input tensor of shape (T=5, C=17, H=128, W=128), dtype float32
      - y: Next-day fire mask of shape (1, H=128, W=128), dtype float32 (values in {0.0, 1.0})
      - persistence: Last-day fire mask of shape (1, H=128, W=128), dtype float32 (Persistence baseline)
      - meta: Dictionary containing tile filename, target date, year, and fire pixel counts
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        cache_size: int = 16,
        max_samples: Optional[int] = None,
        filter_active_fire: bool = False,
        random_seed: int = 42,
    ):
        """
        Args:
            root_dir: Path to processed_v1 directory containing train, val, test folders.
            split: One of 'train', 'val', 'test'.
            cache_size: Number of .npz files kept in memory to minimize disk I/O.
            max_samples: Optional limit on total samples (useful for quick experiments/debugging).
            filter_active_fire: If True, keep only sequences that contain active fire pixels.
            random_seed: Random seed for optional sampling.
        """
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)
        self.index_path = os.path.join(self.split_dir, "index.csv")

        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index file not found: {self.index_path}")

        # Load index
        df = pd.read_csv(self.index_path)

        # Filter active fires if requested
        if filter_active_fire:
            df = df[df["fire_px_target"] > 0].reset_index(drop=True)

        # Subsample if max_samples is specified
        if max_samples is not None and max_samples < len(df):
            df = df.sample(n=max_samples, random_state=random_seed).reset_index(drop=True)

        self.df = df
        self.cache_size = cache_size
        self._file_cache: OrderedDict[str, Dict[str, np.ndarray]] = OrderedDict()

    def _get_file_data(self, filename: str) -> Dict[str, np.ndarray]:
        """Fetch file arrays with LRU cache."""
        if filename in self._file_cache:
            self._file_cache.move_to_end(filename)
            return self._file_cache[filename]

        filepath = os.path.join(self.split_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        # Load npz archive into memory
        with np.load(filepath) as npz:
            data = {
                "dyn_cont": np.array(npz["dyn_cont"], dtype=np.float32),
                "fire": np.array(npz["fire"], dtype=np.float32),
                "weather_mask": np.array(npz["weather_mask"], dtype=np.float32),
                "vegetation_mask": np.array(npz["vegetation_mask"], dtype=np.float32),
            }

        if len(self._file_cache) >= self.cache_size:
            self._file_cache.popitem(last=False)

        self._file_cache[filename] = data
        return data

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        row = self.df.iloc[idx]
        filename = row["file"]
        file_data = self._get_file_data(filename)

        start = int(row["input_start"])
        end = int(row["input_end"]) + 1  # 5 time steps: start..end inclusive
        target_idx = int(row["target"])
        last_day_idx = int(row["input_end"])

        # Extract slices: (T=5, 14, 128, 128)
        dyn_cont = file_data["dyn_cont"][start:end]
        fire = file_data["fire"][start:end, np.newaxis, :, :]              # (T=5, 1, 128, 128)
        weather_mask = file_data["weather_mask"][start:end, np.newaxis, :, :]  # (T=5, 1, 128, 128)
        veg_mask = file_data["vegetation_mask"][start:end, np.newaxis, :, :]   # (T=5, 1, 128, 128)

        # Build 17 channels:
        # 0..12: dyn_cont[:, 0:13] (pr through EVI)
        # 13: fire mask
        # 14: dyn_cont[:, 13:14] (log1p_frp)
        # 15: weather_mask
        # 16: vegetation_mask
        c_0_to_12 = dyn_cont[:, 0:13, :, :]
        c_13 = fire
        c_14 = dyn_cont[:, 13:14, :, :]
        c_15 = weather_mask
        c_16 = veg_mask

        x_np = np.concatenate([c_0_to_12, c_13, c_14, c_15, c_16], axis=1)  # (5, 17, 128, 128)

        # Target mask at day 5: (1, 128, 128)
        y_np = file_data["fire"][target_idx:target_idx + 1, :, :]

        # Persistence prediction (fire mask at day 4): (1, 128, 128)
        persistence_np = file_data["fire"][last_day_idx:last_day_idx + 1, :, :]

        # Convert to PyTorch tensors
        x = torch.from_numpy(x_np)                    # float32, (5, 17, 128, 128)
        y = torch.from_numpy(y_np)                    # float32, (1, 128, 128)
        persistence = torch.from_numpy(persistence_np) # float32, (1, 128, 128)

        meta = {
            "file": filename,
            "target_date": str(row.get("target_date", "")),
            "target_year": int(row.get("target_year", 0)),
            "fire_px_target": int(row.get("fire_px_target", 0)),
            "fire_px_last": int(row.get("fire_px_last", 0)),
        }

        return x, y, persistence, meta


def get_dataloader(
    root_dir: str,
    split: str = "train",
    batch_size: int = 16,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    cache_size: int = 16,
    max_samples: Optional[int] = None,
    filter_active_fire: bool = False,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Factory function to build DataLoader for train, val, or test splits.
    """
    if shuffle is None:
        shuffle = (split == "train")

    dataset = WildfireDataset(
        root_dir=root_dir,
        split=split,
        cache_size=cache_size,
        max_samples=max_samples,
        filter_active_fire=filter_active_fire,
    )

    use_cuda = torch.cuda.is_available()
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(pin_memory and use_cuda),
        drop_last=(split == "train"),
    )
    return dataloader


if __name__ == "__main__":
    import time
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed_v1")

    print(f"Testing WildfireDataset on: {data_dir}")
    ds = WildfireDataset(root_dir=data_dir, split="train", max_samples=100)
    print(f"Dataset length: {len(ds)}")

    t0 = time.time()
    x, y, p, meta = ds[0]
    t1 = time.time()

    print(f"Sample 0 loaded in {(t1 - t0)*1000:.2f} ms")
    print(f"  Input x shape: {x.shape}, dtype: {x.dtype}")
    print(f"  Target y shape: {y.shape}, dtype: {y.dtype}, fire_px: {y.sum().item()}")
    print(f"  Persistence shape: {p.shape}, fire_px: {p.sum().item()}")
    print(f"  Metadata: {meta}")
    print("WildfireDataset verification successful!")
