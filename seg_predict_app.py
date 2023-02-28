import streamlit as st
import pathlib
import collections
import torchinfo
import torchvision
import os
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as TR
import segmentation_models_pytorch as smp
import albumentations as albu


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
        models_name = pathlib.Path(f"./user_data/test/seg_models").rglob("*.pth")  # list models
        model = st.sidebar.selectbox(label="Model", options=[n.stem for n in models_name])
        params["model"] = model

        # Choose the dataset
        datasets_name = pathlib.Path(f"./user_data/test/seg_datasets").iterdir()
        dataset = st.sidebar.selectbox(label="Dataset", options=[n.stem for n in datasets_name if n.is_dir()])
        params["dataset"] = dataset
        
        # Choose the images
        imgs_name = pathlib.Path(f"./user_data/test/seg_datasets/{dataset}/imgs")
        imgs = st.sidebar.selectbox(label="Image", options=os.listdir(imgs_name))
        params["img"] = imgs

    if library == "TensorFlow":
        pass

    return params


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn, resize_h=256, resize_w=256):
    _transform = [
        albu.Resize(resize_h, resize_w, interpolation=cv2.INTER_CUBIC),
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def get_model(params):
    model = params["model"]
    device = params["device"]
    model_path = f"./user_data/test/seg_models/{model}.pth"
    segmentor = torch.load(model_path).to(device)
    return segmentor


def visualization(ori_img, mask_img):
    pass


def predict_pytorch(params, segmentor, predict_button):
    # ------------------------
    # Title
    # ------------------------
    user_name = "test"
    st.title("Defect Segmentation")
    st.subheader("Real-Time Segmentation")

    # ------------------------
    # get params config
    # ------------------------
    model = params["model"]
    backbone = model.split('_')[1]
    device = params["device"]
    dataset_name = params["dataset"]
    img = params["img"]

    # ------------------------
    # load model
    # ------------------------
    # for better performance, load network is out of this function.

    # ------------------------
    # load image
    # ------------------------
    img_path = f"./user_data/test/seg_datasets/{dataset_name}/imgs/{img}"
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ori_shape = (image.shape[1], image.shape[0])

    # visualization image
    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            st.image(image, caption='Original Image', use_column_width=True)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(f"{backbone}", "imagenet")
    preprocessing = get_preprocessing(preprocessing_fn)
    sample = preprocessing(image=image)
    image = sample["image"]
    
    # ------------------------
    # Predict
    # ------------------------
    if predict_button:
        start_time = time.time()

        img = torch.from_numpy(image).unsqueeze(0).to(device)
        pred = segmentor(img)
        thresh=0.5
        pred[pred >= thresh] = 1
        pred[pred < thresh] = 0
        pred = pred.detach().cpu().numpy().squeeze()
        pred = pred.astype('uint8')
        pred = pred * 255
        
        with col2:
            with st.container():
                pred = cv2.resize(pred, ori_shape, interpolation=cv2.INTER_NEAREST)
                st.image(pred, caption='Segmentation Mask', use_column_width=True)

        end_time = time.time()
        st.subheader(f"Time cost : {end_time - start_time:.4f}s")

def predict_tensorflow():
    pass


def main():
    params = get_streamlit_params()
    library = params["library"]

    # only load model once.
    segmentor = get_model(params)

    predict_button = st.sidebar.button(label="开始预测")

    if library == "PyTorch":
        predict_pytorch(params, segmentor, predict_button)
    elif library == "TensorFlow":
        predict_tensorflow()


if __name__ == '__main__':
    main()