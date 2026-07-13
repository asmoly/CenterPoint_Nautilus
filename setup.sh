conda tos accept

conda create -n centerpoint python=3.8 -y
conda activate centerpoint

nvidia-smi

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 \
  --extra-index-url https://download.pytorch.org/whl/cu116

pip install --upgrade pip
pip install spconv-cu116
pip install -r requirements.txt
python setup.py develop

python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -c "import pcdet; print('pcdet ok')"