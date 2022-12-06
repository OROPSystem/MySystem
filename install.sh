pip install dill==0.3.5.1
pip install Flask==2.0.3
pip install Flask_Bootstrap==3.3.7.1
pip install Flask_Mail==0.9.1
pip install Flask_Migrate==3.1.0
pip install Flask_SQLAlchemy==2.5.1
pip install Flask_WTF==1.0.1
pip install GPUtil==1.4.0
pip install h5py==3.1.0
pip install keras==2.9.0
pip install lark==1.1.2
pip install lime==0.2.0.1
pip install matplotlib==3.5.2
pip install numpy==1.22.3
pip install pandas==1.4.2
pip install Pillow==9.2.0
pip install plotly==5.9.0
pip install psutil==5.9.1
pip install rarfile==4.0
pip install scikit_image==0.19.2
pip install scikit_learn==1.1.1
pip install scipy==1.8.1
pip install skimage==0.0
pip install streamlit==1.9.1
pip install tensorboard==2.9.0
pip install tensorflow==2.9.1
pip install tensorflow_gpu==2.5.0
pip install tensorflowjs==3.18.0
pip install timm==0.5.4
pip install traitlets==5.3.0
pip install Werkzeug==2.1.2
pip install WTForms==3.0.1
pip install mmcv-full==1.5.1

# torch版本和segmentation-models-pytorch版本会影响分割模型的兼容性
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# segmentation-models-pytorch 
pip install git+https://github.com/qubvel/segmentation_models.pytorch
pip install albumentations

# mmdet3.0
pip install openmim
mim install mmengine
mim install "mmdet>=3.0.0rc0"