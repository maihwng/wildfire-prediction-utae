"""
clean_dataset_offline.py
-------------------------
Offline Batch Preprocessing Script for Spatio-Temporal Wildfire Dataset.
Bakes the root-preprocessing directly into .npz files on disk:
  1. Zero-masks missing weather pixels (imputes to standardized mean 0.0)
  2. Zero-masks missing vegetation pixels (imputes to standardized mean 0.0)
  3. Clamps extreme FRP outlier spikes to <= 64.0 (~95th percentile)
  4. Atomic file writing with .tmp buffer to prevent any corruption
"""

import os
import sys
import glob
import time
import argparse
import numpy as np
from tqdm import tqdm

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def clean_single_npz(filepath: str, frp_max: float = 64.0) -> bool:
    """Cleans a single .npz file in-place using atomic rename."""
    tmp_path = filepath + ".tmp.npz"
    try:
        with np.load(filepath) as npz:
            data = dict(npz)

        dyn_cont = data["dyn_cont"].astype(np.float32)
        weather_mask = data["weather_mask"].astype(np.float32)[:, np.newaxis, :, :]
        veg_mask = data["vegetation_mask"].astype(np.float32)[:, np.newaxis, :, :]

        # 1. Zero-mask weather channels
        dyn_cont[:, 0:11, :, :] *= weather_mask

        # 2. Zero-mask vegetation channels
        dyn_cont[:, 11:13, :, :] *= veg_mask

        # 3. Clamp FRP outlier spike
        dyn_cont[:, 13, :, :] = np.clip(dyn_cont[:, 13, :, :], -0.02511, frp_max)

        # Store back as float16 to preserve storage efficiency
        data["dyn_cont"] = dyn_cont.astype(np.float16)

        # Save to temporary file first
        np.savez_compressed(tmp_path, **data)

        # Atomic replace
        if os.path.exists(filepath):
            os.replace(tmp_path, filepath)
        return True
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False


def main():
    parser = argparse.ArgumentParser(description="Offline Root-Cleaning for Wildfire Dataset")
    parser.add_argument("--data_dir", type=str, default="data/processed_v1", help="Path to processed_v1 directory")
    parser.add_argument("--frp_max", type=float, default=64.0, help="Upper threshold for FRP clamping")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Splits to clean")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_data_dir = os.path.join(base_dir, args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir

    print(f"Starting offline dataset root-cleaning on: {target_data_dir}")
    print(f"Splits to process: {args.splits}")
    print(f"FRP upper bound: {args.frp_max}")

    total_cleaned = 0
    start_time = time.time()

    for split in args.splits:
        split_path = os.path.join(target_data_dir, split)
        files = glob.glob(os.path.join(split_path, "*.npz"))
        print(f"\nProcessing split '{split}': {len(files)} files found.")

        for f in tqdm(files, desc=f"Cleaning {split}", dynamic_ncols=True):
            if clean_single_npz(f, frp_max=args.frp_max):
                total_cleaned += 1

    elapsed = time.time() - start_time
    print(f"\nSuccessfully cleaned {total_cleaned} files in {elapsed:.1f} seconds ({elapsed/60:.2f} minutes).")


if __name__ == "__main__":
    main()

