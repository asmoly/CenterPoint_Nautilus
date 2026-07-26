import numpy as np
import opv2v

#run explore.py
SPARSE_BELOW = {
    "vehicle": 191,
    "pedestrian": 32,
    "truck": 499,
}

NEAR_BELOW = 23.0
FAR_ABOVE = 33.0

#objects outside POINT_CLOUD_RANGE not considered returns true/false per box
#considerinng only centers for now for simplicity
def scored(frame):
    boxes = frame["boxes"]
    if len(boxes) == 0:
        return np.zeros(0, dtype=bool)
    low = np.array(opv2v.POINT_CLOUD_RANGE[:3])   #(-40, -40, -8)
    high = np.array(opv2v.POINT_CLOUD_RANGE[3:])  #(40, 40, 2)
    centers = boxes[:, :3]
    return np.all((centers >= low) & (centers <= high), axis=1)

def count_points_in_box(points, box):
    cx, cy, cz, dx, dy, dz, yaw = box

    #shift points so box center is origin
    shifted = points[:, :3] - np.array([cx, cy, cz])

    #box and point tilted by +yaw, undo it so easier to count points
    #rotate point around origin formula
    c = np.cos(-yaw)
    s = np.sin(-yaw)
    x = c * shifted[:, 0] - s * shifted[:, 1]
    y = s * shifted[:, 0] + c * shifted[:, 1]
    z = shifted[:, 2] #unchanged

    #point inside box if within half the size on each axis
    half_x = dx/2.0
    half_y = dy/2.0
    half_z = dz/2.0

    inside_x = (x >= -half_x) & (x <= half_x)   #between left/right walls
    inside_y = (y >= -half_y) & (y <= half_y)   #front/back walls
    inside_z = (z >= -half_z) & (z <= half_z)   #bottom/top walls

    #point inside if all 3 true
    inside = inside_x & inside_y & inside_z
    return int(inside.sum())

def object_counts(frame):
    keep = scored(frame)
    output = []
    for index, (name, box, k) in enumerate(zip(frame["classes"], frame["boxes"], keep)):
        if not k:
            continue

        n = count_points_in_box(frame["points"], box)
        if n == 0:
            continue

        distance = float(np.hypot(box[0], box[1]))
        bucket_name = bucket(name, n, distance)

        output.append({"frame_id":frame["frame_id"], 
                       "index":index, 
                       "class":name, 
                       "n_points":n,
                       "distance":distance,
                       "bucket":bucket_name})
    return output

def bucket(class_name, n_points, distance):
    cutoff = SPARSE_BELOW[class_name]
    if n_points < cutoff:
        level = "sparse"
    else:
        level = "dense"

    if distance < NEAR_BELOW:
        where = "near"
    elif distance < FAR_ABOVE:
        where = "mid"
    else:
        where = "far"

    return f"{class_name}|{level}|{where}"