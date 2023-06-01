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

import glob
import os

from itertools import islice
import matplotlib.pyplot as plt
from scipy.fft import fft
from tqdm import *
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.manifold import TSNE

def get_streamlit_params():

    # return Dict
    params = collections.defaultdict(int)

    # Choose the DL Library
    library = st.sidebar.selectbox(label="深度学习框架", options=["PyTorch", "TensorFlow"], help="choose you library")
    params["library"] = library

    # Choose the Device 
    if library == "PyTorch":
        available_gpus = ["cpu"] + ["cuda:" + str(i) for i in range(torch.cuda.device_count())]
        device = st.sidebar.selectbox(label="设备", options=available_gpus)
        params["device"] = device

        # Choose the model
        models_path = pathlib.Path("")
        models_name = pathlib.Path(f"./user_data/test/fd_models").rglob("*.pth")  # list models
        model = st.sidebar.selectbox(label="模型", options=[n.stem for n in models_name])
        params["model"] = model

        # Choose the dataset
        datasets_name = pathlib.Path(f"./user_data/test/fd_datasets").iterdir()
        dataset = st.sidebar.selectbox(label="数据集", options=[n.stem for n in datasets_name if n.is_dir()])
        params["dataset"] = dataset
        
        # Choose the load
        load_range = pathlib.Path(f"./user_data/test/fd_datasets/{dataset}")
        loads = st.sidebar.selectbox(label="负载", options=os.listdir(load_range))
        params['load'] = loads
        
        # # Choose the images
        # imgs_name = pathlib.Path(f"./user_data/test/seg_datasets/{dataset}/imgs")
        # imgs = st.sidebar.selectbox(label="Image", options=os.listdir(imgs_name))
        # params["img"] = imgs
        
        # Number per class for test
        no_test = st.sidebar.text_input(label='用于测试的每类样本数', value='300')
        params['no_test'] = int(no_test)
        
        # Signal Size
        signal_size = st.sidebar.text_input(label='信号尺寸', value='1200')
        params['signal_size'] = int(signal_size)

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
    model_path = f"./user_data/test/fd_models/{model}.pth"
    fd_model = torch.load(model_path).to(device)
    return fd_model


def visualization(ori_img, mask_img):
    pass


def predict_pytorch(params, fd_model, predict_button, data_analyze_button):
    # TODO 目前该pipeline只在HUST数据集上完成测试
    # ------------------------
    # Title
    # ------------------------
    user_name = "test"
    st.title("Fault Diagnosis")
    # ------------------------
    # get params config
    # ------------------------
    model = fd_model
    load_dir = "./user_data/test/fd_datasets/{}/{}".format(params["dataset"], params["load"])
    files = glob.glob(os.path.join(load_dir , "*.csv"))
    no_test = params['no_test']
    signal_size = params['signal_size']

    device = params['device']
    device=torch.device(device=device)
    # ------------------------
    # load model
    # ------------------------
    # for better performance, load network is out of this function.

    # ------------------------
    # Data Analysis
    # ------------------------
    if data_analyze_button:
        st.subheader("Original Data Analysis")
        
        st.markdown('#### There are totally {} conditions in the chosen dataset'.format(len(files)))
        
        for filename in files:
            condition = filename.split('/')[-1][:-4]
            st.markdown('##### Condition: {}'.format(condition))
            
            file = open(filename, "r", encoding='gb18030', errors='ignore')
            file_data = []
            for line in islice(file, 22, None):   # Skip the first 22 lines
                line = line.rstrip()
                word = line.split("\t", 5)         # Separated by \t
                file_data.append(eval(word[2]))   # Take a vibration signal in the x direction as input
            
            file_data = np.array(file_data).reshape(-1)[:signal_size]
    
            col1, col2 = st.columns(2)
            
            st.set_option('deprecation.showPyplotGlobalUse', False)
            with col1:
                plt.plot(file_data)
                plt.title('Time-Series Data')
                st.pyplot()
            
            with col2:
                file_data -= np.mean(file_data)
                fft_trans = abs(fft(file_data))
                len_fft = len(fft_trans) // 2 + 1
                plt.plot(fft_trans[:len_fft])
                plt.title('Frequency-Domain Data')
                st.pyplot()
    
    # ------------------------
    # Predict
    # ------------------------
    if predict_button:
        # Load Dataset
        train_dataset, test_dataset = [], []
        for i, path in enumerate(files):
            train, test = data_load(path, no_test, signal_size)
            
            train_dataset.append(train)
            test_dataset.append(test)
        
        train_dataset = np.concatenate(train_dataset, axis=0)
        test_dataset = np.concatenate(test_dataset, axis=0)
        
        dataloaders = dataloader_create(train_dataset, test_dataset, no_test, len(files))
        test_loader = dataloaders['test']
        
        st.subheader("Data Loading Success!!")
        
        st.subheader('Start Predicting!!')
        model.eval()

        labels, pred_labels, features = [], [], []
        for input, label in test_loader:
            input, label = input.type(torch.FloatTensor).to(device), label.to(device)

            feature, output = model(input, True)

            features.append(feature.data.cpu().numpy())

            pred = torch.max(output, 1)[1]

            labels.append(label.detach().cpu().numpy())
            pred_labels.append(pred.detach().cpu().numpy())
        
        labels = np.concatenate(labels, axis=0)
        pred_labels = np.concatenate(pred_labels, axis=0)
        features = np.concatenate(features, axis=0)
        
        C2 = confusion_matrix(labels, pred_labels)

        f1 = f1_score(labels, pred_labels, average='macro')
        recall = recall_score(labels, pred_labels, average='macro')
        precision = precision_score(labels, pred_labels, average='macro')
        accuracy = accuracy_score(labels, pred_labels)

        class_measure = classification_report(labels, pred_labels)
        
        st.markdown(('#### ******************* Classification Report ***********************'))
        st.write(class_measure)
        
        st.markdown(('#### ******************* Test Metrics ***********************'))
        st.write(f'##### f1 score: {f1*100:.2f}%')
        st.write(f'##### recall score: {recall*100:.2f}%')
        st.write(f'##### precision score: {precision*100:.2f}%')
        st.write(f'##### accuracy score: {accuracy*100:.2f}%')
        
        st.markdown(('#### ******************* Confusion Matrix ***********************'))
        plt.imshow(C2, cmap=plt.cm.Blues)
        plt.title("Normalized confusion matrix")  # title
        plt.xlabel("Predict label")
        plt.ylabel("Truth label")

        num_classes = len(files)

        for x in range(num_classes):
            for y in range(num_classes):
                value = float(C2[y, x])  # 数值处理
                plt.text(x, y, value, fontsize=10, verticalalignment='center', horizontalalignment='center')  # 写值

        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

        plt.colorbar()  # 色条
        st.pyplot()
        
        st.markdown(('#### ******************* TSNE Visualization ***********************'))
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        tsne_features = tsne.fit_transform(features)

        unique = np.unique(labels)
        colors = [plt.cm.rainbow(i / float(len(unique)-1)) for i in range(len(unique))]

        for i, u in enumerate(unique):
            xi = [tsne_features[:,0][j] for j in range(len(tsne_features[:,0])) if labels[j]==u]
            yi = [tsne_features[:,1][j] for j in range(len(tsne_features[:,0])) if labels[j]==u]
            plt.scatter(xi, yi, c=colors[i], s=7, alpha=1, label=str(u))

        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
        plt.legend(bbox_to_anchor=(0.5, -0.2), loc=8, ncol=len(unique)//2, handletextpad=0.1,\
            scatterpoints=1, labelspacing=0.1, columnspacing=0.1)
        plt.xticks([])
        plt.yticks([])
        st.pyplot()
        
        
def data_load(file_path, number_per_class_test, signal_size, number_per_class=1):
    file = open(file_path, 'r', encoding='gb18030', errors='ignore')
    file_data = []
    for line in islice(file, 22, None):    # Skip the first 22 lines
        line = line.rstrip()
        word = line.split("\t", 5)          # Separated by \t
        
        # 2 means x direction; 3 means y direction; 4 means z direction
        file_data.append(eval(word[2]))    # Take a vibration signal in the x direction as input
    
    file_data = np.array(file_data).reshape(-1)
    return dataset_create(file_data, number_per_class_test, number_per_class, signal_size)

def dataloader_create(train_dataset, test_dataset, number_per_class_test, n_class, number_per_class=1, val=False, batch_size=1):
    train_label = []
    for i in range(n_class):
        train_label += [i] * number_per_class

    train_label = np.array(train_label)

    train_dataset = train_dataset[:number_per_class*n_class, :, :]
    train_dataset = TensorDataset(torch.tensor(train_dataset,dtype=float), torch.tensor(train_label))
    
    if val:
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        data_train, data_val = random_split(train_dataset, [train_size, val_size])
    else:
        data_train = train_dataset
    
    test_label = []
    for i in range(n_class):
        test_label += [i] * number_per_class_test

    test_label = np.array(test_label)
    
    data_test = TensorDataset(torch.tensor(test_dataset,dtype=float), torch.tensor(test_label))
    
    dataloaders = {}
    dataloaders['train'] = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
    dataloaders['test'] = DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

    if val:
        dataloaders['val'] = DataLoader(data_val, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
        
    return dataloaders

def dataset_create(file_data, number_per_class_test, number_per_class, signal_size):
    start = 0
    
    enc = signal_size // 3

    for i in tqdm(range(number_per_class_test)):
        sig = file_data[start:start+signal_size].reshape(1, signal_size)
        
        sig = sig[np.newaxis, :]
        
        if i == 0:
            test =  sig
        else:
            test = np.concatenate([test, sig], axis=0)
            
        start += enc
        pass
    
    # Train dataset & Val dataset
    start += (signal_size - enc)
    for i in tqdm(range(number_per_class)):
        sig = file_data[start:start+signal_size].reshape(1, signal_size)
        
        sig = sig[np.newaxis, :]
        
        if i == 0:
            train =  sig
        else:
            train = np.concatenate([train, sig], axis=0)
            
        start += enc
        pass
    
    return train, test


#
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x1(planes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, in_channel=1, out_channel=10, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, out_channel)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, return_feature=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)

        if return_feature:
            return x, y
        else:
            return y
#
def predict_tensorflow():
    pass


def main():
    params = get_streamlit_params()
    library = params["library"]

    # only load model once.
    fd_model = get_model(params)

    data_analyze_button = st.sidebar.button(label='数据展示')
    predict_button = st.sidebar.button(label="开始预测")
    

    if library == "PyTorch":
        predict_pytorch(params, fd_model, predict_button, data_analyze_button)
    elif library == "TensorFlow":
        predict_tensorflow()


if __name__ == '__main__':
    main()