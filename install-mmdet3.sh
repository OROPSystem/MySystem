# conda create -n det python=3.10
# pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
# pip install jupyter tensorboard

### install as packages
# pip install openmim
# mim install mmengine
# mim install "mmdet>=3.0.0rc0"

### install as extra folders
# mkdir packages
# cd ./packages
# git clone -b dev-3.x git@github.com:open-mmlab/mmdetection.git
# git clone git@github.com:open-mmlab/mmengine.git

# cd ./mmdetection
# pip install -r requirements.txt
# python setup.py develop

# cd ../mmengine
# pip install -r requirements.txt
# python setup.py develop
