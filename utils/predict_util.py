from tkinter import image_names
from tkinter.messagebox import NO
from matplotlib.pyplot import title
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import random
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
        print(transform)
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
        self.true_labels = []
        self.confidence = []

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
        # fix utf-8 bug.
        self.__images = [item[0].encode('utf-8', 'replace').decode('utf-8') for item in self.predict_dataset.imgs]
        st.sidebar.selectbox(label="Choose one image", options=self.__images)
        labels = [item[1] for item in self.predict_dataset.imgs]
        privacy_images = [self.dataset_name + i.split(self.dataset_name)[1] for i in self.__images]
        df_data = list(zip(privacy_images, labels, self.predict_labels, self.confidence))
        df = pd.DataFrame(data=df_data, columns=["image source", "labels", "predictions", "confidence"])

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

    def show_confidence(self):
        # plotly出图，能设置行列，但是图间隔较大，显示小，标题显示有问题

        # fig = make_subplots(rows=3, cols=3, horizontal_spacing=0, vertical_spacing=0, subplot_titles=self.confidence[:9])
        # # fig = go.Figure()
        # for i in range(9):
        #     img = Image.open(self.__images[i], mode="r")
        #     img_array = np.array(img)
        #     if img.mode == "L":
        #         img_array = np.stack([img_array, img_array, img_array], axis=-1)
        #     img_trace = go.Image(z=img_array, colormodel="rgb")
        #     fig.add_trace(img_trace, row=i//3+1, col=i%3+1)
        # layout = go.Layout(title="置信度")
        # fig.update_layout(coloraxis_showscale=False)
        # fig.update_xaxes(showticklabels=False)
        # fig.update_yaxes(showticklabels=False)
        # st.plotly_chart(fig, layout=layout)

        # streamlit出图，只能设置列数，行数只能是一行
        def create_confidence():
            col1, col2, col3 = st.columns(3)
            index = random.sample(list(range(0, len(self.predict_labels))), 3)
            idx = [self.predict_labels[i] for i in index]
            with col1:
                with st.container():
                    st.text(f"Predict: {self.idx2cls[idx[0]]}")
                    img = Image.open(self.__images[index[0]], mode="r")
                    img_array = np.array(img)
                    st.image(img_array, caption=f"Confidence: {self.confidence[index[0]]:.2f}", use_column_width="always")
            with col2:
                with st.container():
                    st.text(f"Predict: {self.idx2cls[idx[1]]}")
                    img = Image.open(self.__images[index[1]], mode="r")
                    img_array = np.array(img)
                    st.image(img_array, caption=f"Confidence: {self.confidence[index[1]]:.2f}", use_column_width="always")
            with col3:
                with st.container():
                    st.text(f"Predict: {self.idx2cls[idx[2]]}")
                    img = Image.open(self.__images[index[2]], mode="r")
                    img_array = np.array(img)
                    st.image(img_array, caption=f"Confidence: {self.confidence[index[2]]:.2f}", use_column_width="always")
        create_confidence()
        create_confidence()
        create_confidence()


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
        self.model.eval()

        # metric
        predict_loss, correct = 0, 0
        num_batches = len(self.predict_iter)
        size = len(self.predict_iter.dataset)

        # predict
        with torch.no_grad():
            for inputs, targets in self.predict_iter:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                pred = self.model(inputs)

                # Adapt Lizhaofu model
                if isinstance(pred, tuple):
                    pred = pred[1] # (feature, logit) -> logit

                predict_loss += self.loss_fn(pred, targets).item()
                correct      += (pred.argmax(1) == targets).type(torch.float).sum().item()
                # st.text(f"{pred.argmax(1)}, {targets}")
                self.predict_labels.extend(pred.argmax(1).tolist())
                # add true labels and confidence
                self.confidence.extend(torch.max(nn.Softmax(dim=1)(pred), dim=1)[0].tolist())
                self.true_labels.extend(targets.tolist())
                self.count   += 1
                progress_bar.progress(self.count / (num_batches))
        predict_loss /= num_batches

        # Metrics
        correct /= size
        precision = precision_score(self.true_labels, self.predict_labels, average='macro')
        recall = recall_score(self.true_labels, self.predict_labels, average='macro')

        st.subheader("Test Metrics:")
        st.text("Accuracy: {:.2%} \nAvg Loss: {:.4f} \n漏检率: {:.2%} \n误检率: {:.2%}".format(correct, predict_loss, 1-recall, 1-precision))

        # lables
        st.subheader("Labels")
        self.show_labels()

        # Visualization
        st.subheader("Visualization")
        self.show_images()
        # Confidence
        self.show_confidence()