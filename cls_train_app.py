import pathlib
from matplotlib.style import library
import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision
import torchinfo
import time
import timm
import random
import collections
from PIL import Image
from traitlets import default
from utils.dataset_split import DatasetSplit
from utils.train_util import Trainer


def get_streamlit_params():

    # return Dict
    params = collections.defaultdict(int)

    # Choose the DL Library
    library = st.sidebar.selectbox(label="Library", options=["PyTorch", "TensorFlow"], help="choose you library")
    params["library"] = library

    # Choose the Device 
    if library == "PyTorch":
        available_gpus = ["cpu"] + ["cuda:" + str(i) for i in range(torch.cuda.device_count())]
        device = st.sidebar.selectbox(label="Device", options=available_gpus)
        params["device"] = device

        # Choose the model
        models_path = pathlib.Path("")
        models_name = pathlib.Path(f"./user_data/test/models").rglob("*.pth")  # list models
        model = st.sidebar.selectbox(label="Model", options=[n.stem for n in models_name])
        params["model"] = model

        # Choose the View Structure
        view_structure = st.sidebar.checkbox(label="View Structure", value=False)
        params["view_structure"] = view_structure

        # Choose the dataset
        datasets_name = pathlib.Path(f"./user_data/test/datasets/image").iterdir()
        dataset = st.sidebar.selectbox(label="Dataset", options=[n.stem for n in datasets_name if n.is_dir()])
        params["dataset"] = dataset

        # Choose the split ratio
        train_scale = st.sidebar.slider(label="Train Scale", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
        val_scale = st.sidebar.slider(label="Validate Scale", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        test_scale = st.sidebar.slider(label="Test Scale", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        params["split_ratio"] = [train_scale, val_scale, test_scale]

        # Choose the image_size
        img_resize = int(st.sidebar.number_input(label="Image Resize", min_value=0, max_value=256, step=1, value=224))
        params["img_resize"] = img_resize

        # Choose the learning rate
        learning_rate = st.sidebar.slider(label="Learning Rate (x1e-4)", min_value=0.0, max_value=100.0, value=1., step=1.0)
        params["lr"] = learning_rate * 1e-4

        # Choose the batch_size
        batch_size = st.sidebar.slider(label="Batch Size", min_value=1, max_value=256, value=4, step=1)
        params["bs"] = batch_size

        # Choose the epochs
        epoch = st.sidebar.slider(label="Epochs", min_value=1, max_value=100, value=10, step=1)
        params["epoch"] = epoch

    if library == "TensorFlow":
        pass

    return params



def train_pytorch(params, train_button, stop_button):
    # ------------------------
    # Title
    # ------------------------
    user_name = "test"
    st.title("Train")

    # ------------------------
    # get params config
    # ------------------------
    model = params["model"]
    device = params["device"]
    img_resize = params["img_resize"]
    batch_size = params["bs"]
    dataset = params["dataset"]
    view_structure = params["view_structure"]
    split_ratio = params["split_ratio"]
    lr = params["lr"]
    epoch = params["epoch"]

    # ------------------------
    # load model
    # ------------------------
    # TODO 区分tensorflow和PyTorch
    # TODO 加入username变量
    model = torch.load(f"./user_data/test/models/{model}.pth", map_location=torch.device(device=device))

    # ------------------------
    # visualize model
    # ------------------------
    if view_structure:
        # TODO 通道数自适应, 有一些channel=1, 显示不了
        st.text(f"Input Shape \t\t\t\t ({batch_size}, {3}, {img_resize}, {img_resize})")
        st.text(torchinfo.summary(model, input_size=(batch_size, 3, img_resize, img_resize)))

    # ------------------------
    # load dataset transform
    # ------------------------
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    # ------------------------
    # Train
    # ------------------------
    if train_button:
        # 1. split dataset
        data_process = DatasetSplit(user_name=user_name, dataset_name=dataset)
        data_process.split_dataset(split_ratio=split_ratio)
        st.subheader("Data Split Success!!")

        # 2. create train dataset
        train_iter = data_process.create_dataloader_iterator(mode="train", batch_size=batch_size, trans=transform, shuffle=True, drop_last=True)
        valid_iter = data_process.create_dataloader_iterator(mode="valid", batch_size=batch_size, trans=None, shuffle=False, drop_last=False)
        test_iter = data_process.create_dataloader_iterator(mode="test", batch_size=batch_size, trans=None, shuffle=False, drop_last=False)

        # 3. trainer
        demo = Trainer(model, train_iter, valid_iter, epoch, lr, device=torch.device(device=device))
        st.subheader("Start Training")
        demo.train()

    # TODO : 结束训练
    if stop_button:
        pass


def train_tensorflow():
    pass


def main():
    params = get_streamlit_params()
    library = params["library"]

    train_button = st.sidebar.button(label="开始训练")
    stop_button = st.sidebar.button(label="结束训练")

    if library == "PyTorch":
        train_pytorch(params, train_button, stop_button)
    elif library == "TensorFlow":
        train_tensorflow()


if __name__ == '__main__':
    main()
