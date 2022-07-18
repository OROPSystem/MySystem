import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
import pandas as pd
import plotly.graph_objects as go
import time

# 沿用train_app.py 117-123
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)


class PredictDataset:
    def __init__(self, user_name, dataset_name, model, default_cfg={}):
        self.user_name = user_name
        self.dataset_name = dataset_name
        self.model = model
        self.default_cfg = default_cfg

    def data_process(self, *args):
        """数据预处理
        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def tranforms(self):
        # parse transform from saved model with timm
        config = resolve_data_config(self.default_cfg, model=self.model)
        transform = create_transform(**config)
        return transform

    def create_dataloader_iterator(self, mode, batch_size, trans=None, shuffle=True, drop_last=False):
        img_path = f"./user_data/{self.user_name}/datasets/image/{self.dataset_name}/{mode}"
        dataset = datasets.ImageFolder(img_path, transform=trans)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        return dataset, dataloader


class ClassifyPredicter:
    def __init__(self, dataset_name, predict_dataset):
        self.dataset_name = dataset_name
        self.predict_dataset = predict_dataset
        self.predict_labels = []

    def predict(self, *args):
        """预测标签值并存入predict_labels.
        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def show_labels(self):
        cls2idx = self.predict_dataset.class_to_idx # lable index
        self.idx2cls = {v:k for k,v in cls2idx.items()}
        df = pd.DataFrame(data=self.idx2cls, index=["lables"])
        st.dataframe(df)

    def show_images(self):
        # privacy handle
        images = [item[0] for item in self.predict_dataset.imgs]
        st.sidebar.selectbox(label="Choose one image", options=images)
        labels = [item[1] for item in self.predict_dataset.imgs]
        privacy_images = [self.dataset_name + i.split(self.dataset_name)[1] for i in images]
        df_data = list(zip(privacy_images, labels, self.predict_labels))
        df = pd.DataFrame(data=df_data, columns=["image source", "labels", "predictions"])

        # show the different prediction
        # st.text([v for v in df["labels"]==df["predictions"]])
        color = df["labels"] == df["predictions"] # 不相同颜色记为0
        df_style = df.style.apply(lambda x: ["background-color: yellow" if not color[idx]  else "" for idx, v in enumerate(x)], subset=["predictions"], axis=0) # 标签置黄
        st.dataframe(df_style, 700, 500)

        # plot the histogram
        fig = go.Figure()
        val_label, x_label = df["labels"].value_counts().tolist(), df["labels"].value_counts().index.tolist()
        x_label = [self.idx2cls[idx] for idx in x_label]
        val_pred, x_pred = df["predictions"].value_counts().tolist(), df["predictions"].value_counts().index.tolist()
        x_pred = [self.idx2cls[idx] for idx in x_pred]
        trace1 = go.Bar(y=val_label, x=x_label, text=val_label, name="True", textposition="outside", marker=dict(color="rgb(211, 077, 058)"))
        trace2 = go.Bar(y=val_pred, x=x_pred, text=val_pred, name="Predict", textposition="outside", marker=dict(color="rgb(068, 117, 123)"))
        fig.add_traces([trace1, trace2])
        layout = go.Layout(title="真实标签与预测标签统计图")
        st.plotly_chart(fig, layout=layout)


class Predicter(ClassifyPredicter):
    def __init__(self, model, dataset_name, predict_dataset, predict_iter, device, loss_fn, show=None):
        super().__init__(dataset_name, predict_dataset)
        self.model = model
        self.predict_iter = predict_iter
        self.device = device
        self.loss_fn = loss_fn
        self.show = show
        self.count = 0

    def predict(self):
        progress_bar = st.progress(0)
        st.write(f"Testing on {self.device}")
        self.model.to(self.device)

        # metric
        predict_loss, correct = 0, 0
        num_batches = len(self.predict_iter)
        size = len(self.predict_iter.dataset)

        # predict
        with torch.no_grad():
            for inputs, targets in self.predict_iter:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                pred = self.model(inputs)

                predict_loss += self.loss_fn(pred, targets).item()
                correct      += (pred.argmax(1) == targets).type(torch.float).sum().item()
                # st.text(f"{pred.argmax(1)}, {targets}")
                self.predict_labels.extend(pred.argmax(1).tolist())
                self.count   += 1
                progress_bar.progress(self.count / (num_batches))
        predict_loss /= num_batches
        correct /= size
        st.text(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {predict_loss:>8f} \n")

        # lables
        st.subheader("Labels")
        self.show_labels()

        # Visualization
        st.subheader("Visualization")
        self.show_images()