from pathlib import Path

import numpy as np
import torch
from pcdet.datasets.dataset import DatasetTemplate
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

import opv2v

# Inherits from OPenPCDet dataset class which allows prepoccessing
# The dataset config has info on preporccossing, I have one for pillars and one for voxels, use accordingly
class OPV2VDataset(DatasetTemplate):
    # This dict maps the class names from the opv2v dataset to the class names OpenPCDet expects
    NAME_MAP = {"vehicle":"Vehicle", "pedestrian":"Pedestrian", "truck":"Vehicle"}

    def __init__(self, model_cfg, class_names=["Car", "Pedestrian", "Truck"], training=False, logger=None):
        cfg_from_yaml_file(model_cfg, cfg) # Converts the cfg path to a cfg object

        # Initializes the OpenPCDet with the given dataset config
        super().__init__(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=training, root_path=Path(cfg.DATA_CONFIG.DATA_PATH), logger=logger)

        self.root_path = Path(cfg.DATA_CONFIG.DATA_PATH)
        self.frame_paths = self.find_frames()

    def find_frames(self):
        """Returns a list of all frames in the dataset represented by the paths to the frames .bin file"""
        frame_paths = []

        # Loops through all .bin files
        for bin_path in sorted(self.root_path.glob("*/*/*.bin")):
            yaml_path = bin_path.with_suffix(".yaml") # Finds corresponding .yaml file
            if yaml_path.exists():
                frame_paths.append(bin_path) # If it exists it gets added to the list

        return frame_paths

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.frame_paths)

    def __getitem__(self, index):
        """Gets an item and returns its id, points, labels, and class names preporccessed for OpenPCDet"""
        frame = opv2v.load_frame(str(self.frame_paths[index])) # Gets the frame from opv2v

        # Converts all class names from opv2v format to OpenPCDet format
        gt_names = np.array([OPV2VDataset.NAME_MAP[name] for name in frame["classes"]])

        # Converts it to a dictionary
        # points are a numpy array (# of points, 4)
        # gt_boxes are a numpy array (# of boxes, 7)
        # gt_names is a numpy array of all the class names corresponding to the gt_boxes
        input_dict = {
            "frame_id": frame["frame_id"],
            "points": frame["points"].astype(np.float32),
            "gt_boxes": frame["boxes"].astype(np.float32),
            "gt_names": gt_names,
        }

        # Converts to OpenPCDet format
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict


# This class loads the specific model from a checkpoint, make sure the config file for the dataset is the same as this model because it uses that to load
class ModelToTest():
    def __init__(self, path_to_checkpoint, dataset):
        self.dataset = dataset

        if cfg != None:
            logger = common_utils.create_logger()

            self.model = build_network(
                model_cfg=cfg.MODEL,
                num_class=len(cfg.CLASS_NAMES),
                dataset=dataset,
            )

            self.model.load_params_from_file(
                filename=path_to_checkpoint,
                logger=logger,
                to_cpu=True,
            )

            self.model.cuda()
            self.model.eval()
        else:
            print("ERROR: No config file set (create dataloader first)")

    def eval_frame(self, data_dict):
        with torch.no_grad():
            batch_dict = self.dataset.collate_batch([data_dict]) # This wraps an extra dimension around the frame to make it a batch
            load_data_to_gpu(batch_dict)

            pred_dicts, ret_dict = self.model.forward(batch_dict) # Runs the batch through the model
            return pred_dicts, ret_dict

def main():
    dataset = OPV2VDataset("centerpoint_custom_opv2v.yaml")
    model = ModelToTest()

    print("frame_id:", data_dict["frame_id"])
    print("pred boxes:", pred_dicts[0]["pred_boxes"].shape)
    print("pred scores:", pred_dicts[0]["pred_scores"].shape)
    print("pred labels:", pred_dicts[0]["pred_labels"].shape)

if __name__ == "__main__":
    main()