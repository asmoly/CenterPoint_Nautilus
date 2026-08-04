import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

import sys
sys.path.append('../eval_scripts')
import opv2v

# Run using python3 demo.py --cfg_file cfgs/custom_models/centerpoint_custom.yaml --ckpt ../eval_scripts/checkpoint_epoch_1.pth --data_path ../data/v2x_real_lidar64/val/2023-03-17-16-03-02_11_1/2/000003.bin
# from the tools/ drirectory (make sure val data is downloaded in the data /v2x... folder)

# Inherits from OPenPCDet dataset class which allows prepoccessing
# The dataset config has info on preporccossing, I have one for pillars and one for voxels, use accordingly
class OPV2VDataset(DatasetTemplate):
    # This dict maps the class names from the opv2v dataset to the class names OpenPCDet expects
    NAME_MAP = {"vehicle":"Vehicle", "pedestrian":"Pedestrian", "truck":"Vehicle"}

    def __init__(self, model_cfg, class_names=["Car", "Pedestrian", "Truck"], training=False, logger=None, root_path=None):
        cfg_from_yaml_file(model_cfg, cfg) # Converts the cfg path to a cfg object

        # If a root path isn't specified it just takes it from the data config
        if root_path == None:
            self.root_path = Path(cfg.DATA_CONFIG.DATA_PATH)
        else:
            self.root_path = Path(root_path)

        self.frame_paths = self.find_frames()

        # Initializes the OpenPCDet with the given dataset config
        super().__init__(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=training, root_path=self.root_path, logger=logger)

    def find_frames(self):
        """Returns a list of all frames in the dataset represented by the paths to the frames .bin file"""
        # If the root path is just a single file for testing, then that is the only file in the dataset
        if self.root_path.is_file():
            return [self.root_path]

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


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = OPV2VDataset(model_cfg=args.cfg_file, training=False, logger=logger, root_path=args.data_path)
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
