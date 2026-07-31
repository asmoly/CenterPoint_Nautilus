from pathlib import Path

import numpy as np
import torch
import csv
import argparse
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path",
        help="Which model config to use",
    )
    parser.add_argument(
            "checkpoint_path",
            help="Which model checkpoint to use",
        )
    return parser.parse_args()

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
            return pred_dicts, ret_dict, batch_dict # pred_dicts contains important info, ret_dict is extra bookkeeping my OpenPCDet, also returns batch_dict which contains gt info on gpu


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
    matched_pairs = []

    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return matched_gt_indices # If there are no bounding boxes then the set is empty

    # Makes sure the tensors are pytorch tensors
    if not torch.is_tensor(gt_boxes):
        gt_boxes = torch.from_numpy(gt_boxes).float()
    if not gt_boxes.is_cuda:
        gt_boxes = gt_boxes.cuda()
    if not pred_boxes.is_cuda:
        pred_boxes = pred_boxes.cuda()

    pred_boxes = pred_boxes.float()
    gt_boxes = gt_boxes.float()

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
            matched_pairs.append((best_gt_index, int(pred_index.item())))

    return matched_pairs # Returns a list of tuples with matching pred ind and gt indices for matched boxes


def make_bucket_confusion_matrix(gt_names, pred_names, matched_pairs, gt_buckets):
    """
    Returns: confusion[bucket][true_class][predicted_class] = count
    Creates a confusion matrix for one frame
    """
    confusion = defaultdict(lambda: defaultdict(lambda: defaultdict(int))) # Creates a nested dict, with a dict inside, then ints

    # Maps groun truth index to pred index
    gt_to_pred = {}
    for gt_index, pred_index in matched_pairs:
        gt_to_pred[gt_index] = pred_index

    # Loops through the classes of the bounding boxes with corresponding indices
    for gt_index, true_class in enumerate(gt_names):
        bucket_name = gt_buckets.get(gt_index) # Gets the bucket of the bounding box from the dict
        if bucket_name is None:
            continue

        pred_index = gt_to_pred.get(gt_index) # Gets the corresponding predicted index
        # If the gt box has a predicted match then we get the class of it, otherwise we use __missed__
        if pred_index is None: # This means that this groung truth box doesn't have a match, meaning the model didn't even predict there was an object there
            predicted_class = "__missed__"
        else:
            predicted_class = pred_names[pred_index]

        # Then we increment that element in the confusion matrix
        confusion[bucket_name][true_class][predicted_class] += 1

    return confusion

def merge_confusion(total_confusion, frame_confusion):
    """Merges one confusion matrix for a frame into the total confusion matrix"""
    for bucket_name, true_rows in frame_confusion.items(): # Loops through the buckets, bucket_name is the the key, true rows is the element corresponding to that key
        for true_class, pred_cols in true_rows.items(): # Loops through the ground truth class
            for predicted_class, count in pred_cols.items(): # Loops through the predicted class
                total_confusion[bucket_name][true_class][predicted_class] += count # Increment that element in the total confusion matrix

def stats_from_confusion(confusion):
    """Converts the bucket confusion matrix stats split by buckets"""
    rows = []

    for bucket_name in sorted(confusion): # Loops through alphabetically sorted bucket keys
        total_gt = 0
        correct = 0
        missed = 0
        wrong_class = 0

        for true_class, pred_counts in confusion[bucket_name].items(): # Loops through rows (there should only be one per bucket)
            total_gt += sum(pred_counts.values()) # Gets sum of all counts in the row
            correct += pred_counts.get(true_class, 0) # Adds all the correctly detected and classified values
            missed += pred_counts.get("__missed__", 0) # Adds boxes that weren't even detected

            for predicted_class, count in pred_counts.items(): # Loops through columns
                if predicted_class not in (true_class, "__missed__"):
                    wrong_class += count # This adds the count if it isnt the correct class, getting wrong class predictions

        detected = total_gt - missed # How many were detected (not nessessarily correct class)

        # Recals for correct class and detected, and just detected
        recall = correct/total_gt if total_gt > 0 else 0.0
        detection_recall = detected/total_gt if total_gt > 0 else 0.0

        classification_accuracy = correct/detected if detected > 0 else 0.0

        rows.append({
            "bucket": bucket_name,
            "total_gt": total_gt,
            "correct": correct,
            "missed": missed,
            "wrong_class": wrong_class,
            "recall": f"{recall:.4f}",
            "detection_recall": f"{detection_recall:.4f}",
            "classification_accuracy_on_detected": f"{classification_accuracy:.4f}",
        })

        print(f"{bucket_name}:    got: {correct}/{total_gt}    missed: {missed}    wrong class: {wrong_class}    correct recall: {recall}    detection recall: {detection_recall}    classification accuracy: {classification_accuracy}")

    return rows

def prediction_names_from_labels(pred_labels, class_names):
    """Returns a list of the predicted class names corresponding to the predicted boxes"""
    pred_names = []

    for label in pred_labels:
        pred_names.append(class_names[int(label.item()) - 1])

    return pred_names


def gt_names_from_boxes(gt_boxes, class_names):
    """Returns a list of the grount truth class name scorresponding to the predicted boxes"""
    gt_names = []

    # This is just a double check
    if gt_boxes.shape[1] <= 7: # If it doesn't contain data besides the dimensions than there are no classes
        return gt_names

    class_ids = gt_boxes[:, 7].long() # Gets the last element after the dimensions which contains the class id
    for class_id in class_ids:
        gt_names.append(class_names[int(class_id.item()) - 1]) # Gets the corresponding class name and adds it to the list

    return gt_names

def main():
    args = parse_args()
    config_path = args.config_path
    checkpoint_path = args.checkpoint_path


    dataset = OPV2VDataset(config_path)
    model = ModelToTest(checkpoint_path, dataset)

    bucket_lookup = load_bucket_csv("bucket_assignments.csv")
    bucket_stats = defaultdict(lambda: {"total": 0, "detected": 0}) # If accessed elemnt doesn't exist it creates the following empty dict

    num_frames = len(dataset)
    # You can set MAX_FRAMES if you want to limit eval
    if MAX_FRAMES is not None:
        num_frames = min(num_frames, MAX_FRAMES)

    total_confusion = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    frame_count = 0
    for frame_index in range(num_frames):
        frame_count += 1

        data_dict = dataset[frame_index] # Gets the current frame
        frame_id = data_dict["frame_id"] # Gets the frame id from the frame

        pred_dicts, _, batch_dict = model.eval_frame(data_dict)
        pred = pred_dicts[0] # Removes batch dimension
        pred_names = prediction_names_from_labels(pred["pred_labels"], cfg.CLASS_NAMES)

        gt_boxes = data_dict["gt_boxes"] # Gets the ground truth boxes from the original frame
        gt_boxes_with_labels = batch_dict["gt_boxes"][0] # Boxes but with class id as well and already loaded onto gpu
        gt_names = gt_names_from_boxes(gt_boxes_with_labels, cfg.CLASS_NAMES)
        # This checks that the box data only has dimensions info and not id's or anything extra
        if gt_boxes.shape[1] > 7:
            gt_boxes = gt_boxes[:, :7]

        # Returns a list of pairs the indices of ground truth boxes with predicted matches
        matched_pairs = match_predictions_to_gt(pred_boxes=pred["pred_boxes"], pred_scores=pred["pred_scores"], gt_boxes=gt_boxes, iou_thresh=IOU_THRESH)

        frame_buckets = bucket_lookup.get(frame_id, {}) # Gets the the buckets for all the boxes in a specific frame ({} is default fallback value)

        confusion = make_bucket_confusion_matrix(gt_names, pred_names, matched_pairs, frame_buckets)
        merge_confusion(total_confusion, confusion)

        if frame_count % 25 == 0:
            print(f"Analyzed {frame_count} frames")

    stats_from_confusion(total_confusion)
        

    

if __name__ == "__main__":
    main()