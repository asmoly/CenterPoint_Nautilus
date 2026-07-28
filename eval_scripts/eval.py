from pathlib import Path

import numpy as np
import torch
import csv
from pcdet.datasets.dataset import DatasetTemplate
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from collections import defaultdict

import opv2v

MAX_FRAMES = None
IOU_THRESH = 0.5
EVAL_OUTPUT_PATH = "eval_out.csv"


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
            return pred_dicts, ret_dict # pred_dicts contains important info, ret_dict is extra bookkeeping my OpenPCDet


def load_bucket_csv(path):
    """ 
    Loads the csv file containing all of the bucket info for each bounding box (generated my create_bucket_classes.py)
    Returns: buckets[frame_id][gt_index] = bucket_name
    """
    buckets = defaultdict(dict) # Creates an empty dict, when a non existing key is accessed it create an empty dict for that key

    with open(path, newline="") as f:
        reader = csv.DictReader(f) # Gets lines

        for row in reader:
            # Gets relavent values, for a specific frame, and a specific bouding box, get what bucket it belongs to
            frame_id = row.get("frame_id")
            gt_index = row.get("index")
            bucket = row.get("bucket")

            if frame_id is None or gt_index is None or bucket is None:
                raise ValueError("Somethings wrong with bucket csv")

            buckets[frame_id][int(gt_index)] = bucket

    return buckets

def match_predictions_to_gt(pred_boxes, pred_scores, gt_boxes, iou_thresh):
    """This function returns a set of all the ground truth bounding boxes that had a good enough match in the predicted bounding boxes"""
    matched_gt_indices = set()

    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return matched_gt_indices # If there are no bounding boxes then the set is empty

    order = torch.argsort(pred_scores, descending=True) # Returns a list of sorted indicies of scores
    # This function returns a matrix of overlap scores using intersection over union
    ious = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes[:, :7], gt_boxes[:, :7]) # Only takes first 7 values, not class id at the end

    # Loops through score indicies, starting with the highest because they are ordered
    for pred_index in order:
        pred_ious = ious[pred_index] # Gets the overlap scores with all ground truth boxes for the predicted box being analyzed

        best_iou = 0.0
        best_gt_index = None

        for gt_index, iou in enumerate(pred_ious): # Loops through gt boxes overlap with index, iou score
            if gt_index in matched_gt_indices: # If this box has alreayd been matched then it is skipped
                continue

            iou_value = float(iou.item()) # Gets the actual overlap score
            # Checks if it is bigger and sets as new best match
            if iou_value > best_iou:
                best_iou = iou_value
                best_gt_index = gt_index

        # Checks that the overlap is good enough to be considered a match based on our set threshhold
        if best_gt_index is not None and best_iou >= iou_thresh:
            matched_gt_indices.add(best_gt_index)

    return matched_gt_indices # Returns a set of all the indicis of ground truth indices that have a predicted match


def write_results(path, bucket_stats):
    rows = []

    for bucket_name in sorted(bucket_stats):
        total = bucket_stats[bucket_name]["total"]
        detected = bucket_stats[bucket_name]["detected"]
        recall = detected/total if total > 0 else 0.0

        rows.append({
            "bucket": bucket_name,
            "total_gt": total,
            "detected": detected,
            "recall": f"{recall:.4f}",
        })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["bucket", "total_gt", "detected", "recall"],
        )
        writer.writeheader()
        writer.writerows(rows)
    

def main():
    dataset = OPV2VDataset("centerpoint_custom_opv2v.yaml")
    model = ModelToTest("checkpoint_epoch_1.pth", dataset)

    bucket_lookup = load_bucket_csv("bucket_assignments.csv")
    bucket_stats = defaultdict(lambda: {"total": 0, "detected": 0}) # If accessed elemnt doesn't exist it creates the following empty dict

    num_frames = len(dataset)
    # You can set MAX_FRAMES if you want to limit eval
    if MAX_FRAMES is not None:
        num_frames = min(num_frames, MAX_FRAMES)

    for frame_index in range(num_frames):
        data_dict = dataset[frame_index] # Gets the current frame
        frame_id = data_dict["frame_id"] # Gets the frame id from the frame

        pred_dicts, _ = model.eval_frame(data_dict)
        pred = pred_dicts[0] # Removes batch dimension

        gt_boxes = data_dict["gt_boxes"] # Gets the ground truth boxes from the original frame
        # This checks that the box data only has dimensions info and not id's or anything extra
        if gt_boxes.shape[1] > 7:
            gt_boxes = gt_boxes[:, :7]

        # Returns a list of the indices of ground truth boxes with predicted matches
        matched_gt_indices = match_predictions_to_gt(pred_boxes=pred["pred_boxes"], pred_scores=pred["pred_scores"], gt_boxes=gt_boxes, iou_thresh=IOU_THRESH)

        frame_buckets = bucket_lookup.get(frame_id, {}) # Gets the the buckets for all the boxes in a specific frame ({} is default fallback value)

        # Loops through all ground truth bounding boxes
        for gt_index in range(len(gt_boxes)):
            bucket_name = frame_buckets.get(gt_index) # Gets bucket for the specific bounding box
            if bucket_name is None:
                continue

            bucket_stats[bucket_name]["total"] += 1 # Increases the total (no matter if there is a match)

            # If there is a match increase the detected count
            if gt_index in matched_gt_indices:
                bucket_stats[bucket_name]["detected"] += 1

        # 
        if frame_index % 25 == 0:
            print(f"Evaluated {frame_index}/{num_frames} frames")

    write_results(EVAL_OUTPUT_PATH, bucket_stats)
    print(f"Wrote bucket recall to {EVAL_OUTPUT_PATH}")

    

if __name__ == "__main__":
    main()