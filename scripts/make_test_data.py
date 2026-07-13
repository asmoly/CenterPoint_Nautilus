from pathlib import Path
import numpy as np

ROOT = Path("data/custom") # Where to make the dataset
CLASSES = ["Vehicle", "Pedestrian", "Cyclist"] # List of classes for object detection

def make_frame(sample_id: str, seed: int):
    """Generates a random point cloud and labels for a given seed, also adds labeled bounding box clusters to point cloud"""
    rng = np.random.default_rng(seed) # Creates a random number generator

    # Generates a matrix of size 20000 x 4 with random values [-60 to 60, -60 to 60, -2 to 3, 0 to 4]
    # [x, y, z, intensity]
    # 20000 points in the point cloud
    points = rng.uniform(low=[-60, -60, -2, 0], high=[60, 60, 3, 1], size=(20000, 4)).astype(np.float32)

    labels = []

    for obj_i, name in enumerate(CLASSES):
        # Creates random values for the center point and yaw of the bounding box
        x = rng.uniform(-25, 25)
        y = rng.uniform(-25, 25)
        z = rng.uniform(-0.5, 0.5)
        yaw = rng.uniform(-3.14, 3.14)

        # Chooses sizes based on if its a vehicle or pedestrian
        # Doesn't really matter but adds variation to test data
        if name == "Vehicle":
            dx, dy, dz = 4.2, 1.8, 1.6
        elif name == "Pedestrian":
            dx, dy, dz = 0.8, 0.8, 1.8
        else:
            dx, dy, dz = 1.8, 0.8, 1.5

        # Appends a string with "x y z dx dy dz yaw name"
        labels.append(f"{x:.3f} {y:.3f} {z:.3f} {dx:.3f} {dy:.3f} {dz:.3f} {yaw:.3f} {name}")

        # This generates a point cloud of 800 points scatterd around the inside of the bounding box
        cluster = rng.normal(loc=[x, y, z, 0.8], scale=[dx / 5, dy / 5, dz / 5, 0.05], size=(800, 4),).astype(np.float32)

        # Adds the synthetic object to the point cloud
        points = np.vstack([points, cluster])

    # Saves the numpy array point cloud
    np.save(ROOT / "points" / f"{sample_id}.npy", points.astype(np.float32))

    # Writes the label text to the label file
    (ROOT / "labels" / f"{sample_id}.txt").write_text("\n".join(labels) + "\n")

def main():
    # Makes directories for the following folders
    for folder in ["ImageSets", "points", "labels"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)

    train_ids = [f"{i:06d}" for i in range(20)] # Creates list of 20 indexes to point clouds and labels: 000000, 000001, 000002, etc ...
    val_ids = [f"{1000 + i:06d}" for i in range(5)] # Same thing but starts from 1000

    # Writes those ids to the ImageSets files, this is used for the training loop to  determine which point clouds and labels to pull
    (ROOT / "ImageSets" / "train.txt").write_text("\n".join(train_ids) + "\n")
    (ROOT / "ImageSets" / "val.txt").write_text("\n".join(val_ids) + "\n")

    # Creates a point cloud and label for each id
    for i, sample_id in enumerate(train_ids + val_ids):
        make_frame(sample_id, i)

    print(f"Wrote fake dataset to {ROOT}")

if __name__ == "__main__":
    main()