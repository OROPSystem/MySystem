import streamlit as st
import pathlib
import collections
import os
import torch
import time
from typing import Dict, List, Optional, Tuple, Union
import mmcv
from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS
from mmdet.utils import register_all_modules
from mmdet.visualization import DetLocalVisualizer
from mmengine import Config
from mmengine.dist import master_only
from mmengine.structures import InstanceData, PixelData
from mmengine.visualization import Visualizer
from mmdet.registry import VISUALIZERS
from utils.detection.visual import DetLocalVisualizerRT


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
        models_name = pathlib.Path(f"./user_data/test/det_models").rglob("*.pth")  # list models
        model = st.sidebar.selectbox(label="Model", options=[n.stem for n in models_name])
        params["model"] = model

        # Choose the dataset
        datasets_name = pathlib.Path(f"./user_data/test/det_datasets").iterdir()
        dataset = st.sidebar.selectbox(label="Dataset", options=[n.stem for n in datasets_name if n.is_dir()])
        params["dataset"] = dataset
        
        # Choose the images
        imgs_name = pathlib.Path(f"./user_data/test/det_datasets/{dataset}/JPEGImages")
        imgs = st.sidebar.selectbox(label="Image", options=os.listdir(imgs_name))
        params["img"] = imgs

    if library == "TensorFlow":
        pass

    return params


def get_model(params):
    model = params["model"]

    device = params["device"]

    config_file = f'./utils/detection/{model}.py'
    checkpoint_file = f'./user_data/test/det_models/{model}.pth'
    cfg = Config.fromfile(config_file)
    cfg.visualizer.type="DetLocalVisualizerRT"
    register_all_modules()
    detector = init_detector(cfg, checkpoint_file, device=device)  # or device='cuda:0'

    return detector


def predict_pytorch(params, detector, predict_button):
    # ------------------------
    # Title
    # ------------------------
    user_name = "test"
    st.title("Defect Detection")
    st.subheader("Real-Time Detection")

    # ------------------------
    # get params config
    # ------------------------
    model = params["model"]
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
    img_path = f"./user_data/test/det_datasets/{dataset_name}/JPEGImages/{img}"
    image = mmcv.imread(img_path, channel_order="rgb")

    ori_shape = (image.shape[1], image.shape[0])

    # visualization image
    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            st.image(image, caption='Original Image', use_column_width=True)
    
    # ------------------------
    # Predict
    # ------------------------
    if predict_button:
        start_time = time.time()

        result = inference_detector(detector, image)
        random_name = time.time()
        detector.cfg.visualizer.name=f"visualizer_{random_name}"
        visualizer = VISUALIZERS.build(detector.cfg.visualizer)

        visualizer.dataset_meta = detector.dataset_meta
        pred = visualizer.add_datasample(
            f'result{img}',
            image,
            data_sample=result,
            draw_gt = None,
            wait_time=0,
            return_img=True
        )
        print(pred.shape)

        with col2:
            with st.container():
                # pred = cv2.resize(pred, ori_shape, interpolation=cv2.INTER_NEAREST)
                st.image(pred, caption='Detection Box', use_column_width=True)

        end_time = time.time()
        st.subheader(f"Time cost : {end_time - start_time:.4f}s")

def predict_tensorflow():
    pass


def main():
    params = get_streamlit_params()
    library = params["library"]

    # only load model once.
    detector = get_model(params)

    predict_button = st.sidebar.button(label="开始预测")

    if library == "PyTorch":
        predict_pytorch(params, detector, predict_button)
    elif library == "TensorFlow":
        predict_tensorflow()

if __name__ == '__main__':
    main()