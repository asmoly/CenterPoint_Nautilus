# LiDAR Object Detection Framework
This is a python framework I created that allows you to train, test, and evaluate 3D detection models on custom datasets. This framework uses OpenPCDet as it`s base. In this project I already have set up CenterPoint, SECOND, voxel-rcnn, and PointPillars as well as scripts to download the v2x dataset. Some important files to note are the configuration files for the models. These can be found in tools/cfgs/... these contain info on where your dataset is located, how to parse it, and how to augment the data, as well as the model settings. These are worth taking a look at.

I ran this on a server (Nautlius) so if running in WSL or on a linux machine you may need to install some more dependencies based on the error messages.

### How to setup
1. Clone the Repo
2. Run `source install_conda.sh` from the root of the repo
3. Restart your terminal
4. Run `source setup.sh` from the repo root
5. Go to the data/ directory and run `python download_data.py train` to download the v2x train dataset (you can also use parameters "val" or "test" to download different sets of the v2x data)
6. You can also download the val data for future testing

#### Training
Training can be done in many ways, and can be configured via the config files you are referencing. This is a basic training command that can be expanded upon. From the tools/ directory run:
```
python train.py \
	--cfg_file cfgs/opv2v_models/centerpoint_custom_opv2v.yaml \
	--batch_size 16 \
	--epochs 100000 \
	--workers 2 \
```
This should launch the training, for however many epochs you run it for.
Saved checkpoints should be stored in /output/custom_models/centerpoint_custom/default/ckpt/ or a path similar to this (depending on your model name)

#### Testing
To run a test of your trained model on a specific LiDAR point cloud from the dataset and get a visualization run this command from the tools/ directory:
```
python3 demo.py \
	--cfg_file cfgs/custom_models/centerpoint_custom.yaml \
	--ckpt ../output/opv2v_models/centerpoint_custom_opv2v/default/ckpt/latest_model.pth \
	--data_path ../data/v2x_real_lidar64/val/2023-03-17-16-03-02_11_1/2/000003.bin \
```
Make sure you are referencing the correct config file and checkpoint. The data path can be the path to a specific frame from your dataset.

#### Evaluating
First, from the eval_scripts/ directory run `python create_buckets.py` this will categorize objects in the dataset into buckets based on the object type, density of it`s point cloud, and proximity to the LiDAR.
Then run:
```
python3 eval.py centerpoint_custom_opv2v.yaml ../output/opv2v_models/centerpoint_custom_opv2v/default/ckpt/latest_model.pth
```
The first parameter is the config file you are referencing, and the second parameter is the checkpoint you are evaluating. If this runs successfully it will give you a break down of the statistics of the models accuracy and precision.

Reference my YouTube video for more detailed instructions: https://www.youtube.com/watch?v=tbxvvmdVJoM
