import glob
import os
import numpy as np
import yaml
from opencood.data_utils import SUPER_CLASS_MAP
from opencood.utils.transformation_utils import x_to_world
from opencood.utils.transformation_utils import x1_to_x2

#flip class map around so raw name --> super class
CLASS_OF = {}
for super_name, raw_names in SUPER_CLASS_MAP.items():
    for raw_name in raw_names:
        CLASS_OF[raw_name] = super_name

#defined in starting kit config.json, objects ouyside the range don't get scored
POINT_CLOUD_RANGE = (-40.0, -40.0, -8.0, 40.0, 40.0, 2.0)


def load_frame(bin_path):
    path = os.path.normpath(bin_path)
    pieces = path.split(os.sep)
    scenario = pieces[-3]            
    agent    = pieces[-2]            
    filename = pieces[-1]

    #read points --> .bin is just float32 numbers with 4 per point in the cloud (x, y, x, and a 4th val)
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)

    #drop any 0,0,0 rows since there can't be a point at sensor's origin
    '''
    good_rows = []
    for row in points:
        x, y, z = row[0], row[1], row[2]
        if x == 0.0 and y == 0.0 and z == 0.0:
            continue                  
        good_rows.append(row)         
    points = np.array(good_rows)
    '''

    points = points[~np.all(points[:, :3] == 0.0, axis=1)]

    #read labels from matching .yaml file
    with open(path[:-4] + ".yaml") as f:
        meta = yaml.safe_load(f)

    pose = []
    for i in meta["lidar_pose"]:
        pose.append(float(i))

    world_to_sensor = x1_to_x2([0, 0, 0, 0, 0, 0], pose)  #world coords --> sensor coords

    classes = []
    raw_classes = []
    boxes = []
    unmapped = []

    for obj in (meta.get("vehicles")).values():
        name = CLASS_OF.get(obj["obj_type"])
        if name is None:
            unmapped.append(obj["obj_type"])
            continue
        classes.append(name)
        raw_classes.append(obj["obj_type"])

        #move box center from world to sensor coordinatess
        location = np.array([obj["location"][0], obj["location"][1], obj["location"][2], 1.0])
        center = world_to_sensor.dot(location)[:3]

        #extent is half, so double
        size = [2.0 * obj["extent"][0], 2.0 * obj["extent"][1], 2.0 * obj["extent"][2]]

        #box's facing direction (yaw), in radians between -pi and pi
        yaw = np.radians(obj["angle"][1] - pose[4]) #subtract by sensor's yaw gives boxes dir relative to sensor
        yaw = (yaw + np.pi) % (2.0 * np.pi) - np.pi

        boxes.append([center[0], center[1], center[2], size[0], size[1], size[2], yaw])

    return {
        "frame_id": f"{scenario}|{agent}|{filename[:-4]}",        
        "points": points,
        "boxes": np.array(boxes, dtype=float).reshape(-1, 7),
        "classes": classes,
        "raw_classes": raw_classes,
        "viewpoint": "infra" if int(agent) < 0 else "vehicle",
        "unmapped": unmapped,
    }

def iter_frames(root):
    for path in sorted(glob.glob(os.path.join(root, "*", "*", "*.bin"))):
        if os.path.exists(path[:-4] + ".yaml"): 
            yield load_frame(path)