#!/usr/bin/env python3
"""
Simple PyTorch Dataset for OPV2V / V2X-Real frames.

This uses your existing opv2v.py file to load each frame.
It does not convert files, and it does not depend on OpenPCDet internals.
"""

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from pcdet.datasets.dataset import DatasetTemplate

import opv2v

# This dataset loads 
class OPV2VDataset(DatasetTemplate):
    def __init__(self, dataset_root):
        self.dataset_root = Path(dataset_root)
        self.frame_paths = OPV2VDataset.find_frames(self.dataset_root)

    @staticmethod
    def find_frames(path_to_dataset_root):
        """Returns a list of all frames in the dataset, represented by their .bin files"""
        frame_paths = []

        # Loops through all .bin files which contain points clouds
        for bin_path in sorted(path_to_dataset_root.glob("*/*/*.bin")):
            yaml_path = bin_path.with_suffix(".yaml") # Gets path to file with same name but .yaml extension for the labels
            if yaml_path.exists():
                frame_paths.append(bin_path) # If it exists then we add the bin path

        return frame_paths

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.frame_paths)

    def __getitem__(self, index):
        """Gets a single frame and returns its frame id, the point cloud as a numpy array (N, 4), labels as a numpy array (M, 7), and list of classes corresponding to bounding boxes"""
        frame = opv2v.load_frame(str(self.frame_paths[index]))

        input_dict = {
            "frame_id": frame["frame_id"],
            "points": frame["points"].astype(np.float32),
            "gt_boxes": frame["boxes"].astype(np.float32),
            "gt_names": np.array(frame["classes"]),
        }

        data_dict = self.prepare_data(data_dict=input_dict) # This converts the data to the OpenPCDet format
        return data_dict


def main():
    dataset = OPV2VDataset("v2x_real_lidar64_val/val")

    test_frame = dataset.__getitem__(5)
    print(test_frame)
    # print(test_frame["frame_id"])
    # print(test_frame["points"].shape)
    # print(test_frame["gt_boxes"].shape)
    # print(test_frame["gt_names"].shape)

if __name__ == "__main__":
    main()
