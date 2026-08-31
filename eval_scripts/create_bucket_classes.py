#!/usr/bin/env python3
"""
Create per-object bucket assignments for an OPV2V/V2X-Real dataset.

Expected existing files:
  buckets/classify.py
  buckets/opv2v.py

Example:
  python create_bucket_classification.py \
    --dataset-root /path/to/V2X-Real-Lidar-64/validation \
    --buckets-dir /path/to/sim2real/buckets \
    --output bucket_assignments.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import classify
import opv2v


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default="../data/v2x_real_lidar64/val/",
        help="Root to dataset, containing scenario/agent/*.bin files",
    )
    parser.add_argument(
        "--output",
        default="bucket_assignments.csv",
        help="CSV file to write to",
    )
    return parser.parse_args()

def count_valid_frames(dataset_root):
    dataset_root = Path(dataset_root)
    total = 0
    for path in dataset_root.glob("*/*/*.bin"):
        if path.with_suffix(".yaml").exists():
            total += 1

    return total

def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    output_path = Path(args.output).resolve()

    rows = []
    n_frames = 0

    total_frames = count_valid_frames(dataset_root)

    # This loops through every single frame (.bin) file with a mathcing .yaml file in the dataset
    for frame in opv2v.iter_frames(str(dataset_root)):
        rows.extend(classify.object_counts(frame))
        n_frames += 1

        if n_frames % 50 == 0:
            print(f"Classified {n_frames} frames | Progress: {((n_frames/total_frames)*100):.2f}%")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_id", 
                "index", 
                "class", 
                "n_points",
                "distance",
                "bucket"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Read {n_frames} frames")
    print(f"Wrote {len(rows)} scored objects to {output_path}")


if __name__ == "__main__":
    main()
