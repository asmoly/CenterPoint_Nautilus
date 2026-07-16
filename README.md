# CenterPoint Implementation

### How to setup CenterPoint

First clone the repo (preferably to your persistent folder `git clone https://github.com/asmoly/CenterPoint_Nautilus.git`

Then go into the repo root folder and run `source install_conda.sh` if you don’t already have conda installed. Make sure to restart your terminal after this.

Next, from the repo root folder run `source setup.sh`, this will install all the necessary dependencies for OpenPCDet.

Then run, `python scripts/make_test_data.py`, verify that the `custom_dataset`appeared in the `data/` directory. This python file is a helpful file to look at to understand the dataset format. This program generates the random dataset.  It creates random point clouds and then inserts object scatters into the point clouds.

```
# For future reference this is the format the the custom dataset should be

data/custom/
  ImageSets/
    train.txt
    val.txt
  points/
    000001.npy
    000002.npy
  labels/
    000001.txt
    000002.txt

# The points files are 2D numpy arrays with each element being [x, y, z, intensity]
# The labels are text files with each line being: x y z dx dy dz heading_angle class_name
# The image sets are just text files with the ids of the point clouds you want for the train and val set

# These are some other notable config files neccessary for the custom dataset
tools\cfgs\dataset_configs\custom_dataset.yaml
tools\cfgs\custom_models\centerpoint_custom.yaml
```

Run `python -m pcdet.datasets.custom.custom_dataset create_custom_infos tools/cfgs/dataset_configs/custom_dataset.yaml` from the repo root folder. This sets up the custom dataset for training.

Then run this from the `tools/` directory

```
python train.py \
  --cfg_file cfgs/custom_models/centerpoint_custom.yaml \
  --batch_size 1 \
  --epochs 2 \
  --workers 2
```

Here you can specify the training parameters.

Checkpoints should be stored here: `CenterPoint_Nautilus/output/custom_models/centerpoint_custom/default/ckpt/`

You can test a checkpoint using:

```
python [demo.py](http://demo.py/) \
	--cfg_file cfgs/custom_models/centerpoint_custom.yaml \
	--ckpt ../output/custom_models/centerpoint_custom/default/ckpt/checkpoint_epoch_2.pth \
	--data_path ../data/custom/points/001000.npy \
	--ext .npy
```

You should see something like this pop up (This was a mostly untrained model so the output is pretty random but the trained models look much better):

![image.png](image.png)
