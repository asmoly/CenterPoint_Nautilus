conda tos accept

conda create -n centerpoint python=3.8 -y
conda activate centerpoint

conda install -c conda-forge libxcrypt -y
conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11 -y
conda install -c "nvidia/label/cuda-11.6.2" cuda-toolkit -y

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.8/site-packages/torch/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

pip install --upgrade pip
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 \
  --extra-index-url https://download.pytorch.org/whl/cu116

pip install spconv-cu116
pip install -r requirements.txt
pip install open3d

python setup.py develop

python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -c "import pcdet; print('pcdet ok')"
python -c "import pcdet.ops.iou3d_nms.iou3d_nms_cuda as iou3d; print('iou3d ok')"