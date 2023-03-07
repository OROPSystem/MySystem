import streamlit as st
import collections
import numpy as np
import pathlib
import os
from itertools import islice
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import fftpack
import pandas as pd
import pywt
 
 
st.set_page_config(
    page_title=None,
    page_icon=None,
    layout="wide",
)


def main():
    # create three cols
    # 所选数据集及对应工况
    col1, col2 = st.columns(2)
    with col1:
        datasets_name = pathlib.Path(f"./user_data/test/fd_datasets").iterdir()
        dataset = st.selectbox(label="Dataset", options=[n.stem for n in datasets_name if n.is_dir()])
        params["dataset"] = dataset
        
        # Choose the load
    
        load_range = pathlib.Path(f"./user_data/test/fd_datasets/{dataset}")
        loads = st.selectbox(label="Load", options=os.listdir(load_range))
        params['load'] = loads
        
    with col2:    
        fault_type_range = pathlib.Path(f"./user_data/test/fd_datasets/{dataset}/{loads}")
        chosen_fault_type = st.selectbox(label="Chosen Fault Type", options=os.listdir(fault_type_range))
        params["chosen_fault_type"] = chosen_fault_type
        
        signal_size = st.number_input(label='Signal Size', value=1200)
        params['signal_size'] = signal_size
    
    col1, col2, col3 = st.columns([2,2,3], gap='small')
    
    # page in col1
    with col1:
        st.header("Data Preprocessing Type")
        clicked = False
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            button1 = st.button('FFT', use_container_width=True)
            if button1:
                params["button"] = 1 
            button2 = st.button('SDP', use_container_width=True)
            if button2:
                params["button"] = 2    
            button3 = st.button('CWT', use_container_width=True)
            if button3:
                params["button"] = 3  
            button4 = st.button('Envelop Spectrum', use_container_width=True)
            if button4:
                params["button"] = 4  
        with col1_2:
            button5 = st.button('button5', use_container_width=True)
            if button5:
                params["button"] = 5
            button6 = st.button('button6', use_container_width=True)
            if button6:
                params["button"] = 6
            button7 = st.button('button7', use_container_width=True)
            if button7:
                params["button"] = 7
            button8 = st.button('button8', use_container_width=True)
            if button8:
                params["button"] = 8
        # 将button值及时存于外部文件，以防刷新
        s = open("./user_data/preprocessing.txt", 'w') 
        s.write(str(params["button"]))
        s.close()

    params['flag'] = False
    # page in col2
    with col2:
        st.header("Parameters Setting")
        if "button" in params.keys():
            if params["button"] == 1:
                with st.form("template_input"):
                    frequency_sampling = st.number_input("frequency_sampling(*1e3 Hz)", value=25.6)
                    submit = st.form_submit_button()
                if submit:
                    params['flag'] = True
                    params["frequency_sampling"] = frequency_sampling * 1000
                           
            if params["button"] == 2:
                with st.form("template_input"):
                    submit = st.form_submit_button()
                if submit:
                    params['flag'] = True
                    
            if params["button"] == 3:
                with st.form("template_input"):
                    frequency_sampling = st.number_input("frequency_sampling(*1e3 Hz)", value=25.6)
                    submit = st.form_submit_button()
                if submit:
                    params['flag'] = True
                    params["frequency_sampling"] = frequency_sampling * 1000
            
            if params["button"] == 4:
                with st.form("template_input"):
                    frequency_sampling = st.number_input("frequency_sampling(*1e3 Hz)", value=25.6)
                    esLim_lower = st.number_input("esLim_lower", value=0)
                    esLim_upper = st.number_input("esLim_upper", value=500)
                    submit = st.form_submit_button()
                if submit:
                    params['flag'] = True
                    params["frequency_sampling"] = frequency_sampling * 1000
                    params["esLim_lower"] = esLim_lower
                    params["esLim_upper"] = esLim_upper
                    
    st.set_option('deprecation.showPyplotGlobalUse', False)   
     
    # page in col3
    with col3:
        st.header("Data Transform")
        if params['flag']:
            preprocessing(params)


##### FFT
def FFT(Fs, data):
    """
    对输入信号进行FFT
    :param Fs:    采样频率
    :param data:  待FFT的序列
    :return:
    """
    L = len(data)
    N = np.power(2, np.ceil(np.log2(L)))
    result = np.abs(fft(x=data, n=int(N))) / L * 2
    axisFreq = np.arange(int(N / 2)) * Fs / N
    result = result[range(int(N / 2))]
    plt.plot(result)
    plt.title('Frequency-Domain Data by FFT')
    st.pyplot()
    
    
##### SDP
def SDPimage(s):
    """
    # 获取 symmetrized dot pattern 结果
    # s 输入信号
    """
    for i in range(1,7):
        rs = (s - np.min(s)) / (np.max(s) - np.min(s))
        th = (72*i + ((s - np.min(s)) / (np.max(s) - np.min(s)) * 36))[2:]     
        rs = rs[:-2]
        plt.title('Frequency-Domain Data by SDP')
        plt.polar(th*(np.pi/180), rs, '.b')
    st.pyplot()
 
    
##### 连续小波变换时频图
def CWTimage(fs,s):
    """
    # 获取连续小波变换的时频图
    # s 输入信号
    """
    # freq_range = np.arange(1,int(fs/2))
    freq_range = np.linspace(1,int(fs/2),200)
    cwtmatr, freqs = pywt.cwt(s, freq_range, 'morl') 
    t = (np.arange(0, len(s))) / fs
    
    plt.contourf(t, freqs, np.abs(cwtmatr), 100)
    plt.title('Frequency-Domain Data by CWT')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.colorbar()
    st.pyplot()


##### 包络谱
def Envelopspectrum(signal, Fs, esLim_lower, esLim_upper):
    """
    # 获取信号包络谱
    # signal 输入信号
    # Fs     采样率
    # esLim  分析频率范围
    """
    signal = signal / abs(np.max(signal))
    a_signal_raw = fftpack.hilbert(signal)
    hx_raw_abs = abs(a_signal_raw)
    hx_raw_abs1 = hx_raw_abs - np.mean(hx_raw_abs)
    es_raw = fftpack.fft(hx_raw_abs1, len(signal))
    es_raw_abs = abs(es_raw)
    es_raw_abs = es_raw_abs / max(es_raw_abs)

    t1 = (np.arange(0, len(signal))) / Fs
    f1 = t1 * Fs * Fs / len(signal)
    
    plt.plot(f1, es_raw_abs)
    plt.title('Envelope spectrum')
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Acc(normalized)')
    plt.xlim([esLim_lower, esLim_upper])
    st.pyplot()


# preprocessing
def preprocessing(params):
    chosen_fault = "./user_data/test/fd_datasets/{}/{}/{}".format(params["dataset"], params["load"], params['chosen_fault_type'])
    
    file = open(chosen_fault, "r", encoding='gb18030', errors='ignore')
    file_data = []
    for line in islice(file, 22, None):   # Skip the first 22 lines
        line = line.rstrip()
        word = line.split("\t", 5)         # Separated by \t
        file_data.append(eval(word[2]))   # Take a vibration signal in the x direction as input
    
    file_data = np.array(file_data).reshape(-1)[:params['signal_size']]
    plt.plot(file_data)
    plt.title('Time-Series Data')
    st.pyplot()
    
    if params["button"] == 1:
        FFT(params["frequency_sampling"], file_data)
    if params["button"] == 2:
        SDPimage(file_data)
    if params["button"] == 3:
        CWTimage(params["frequency_sampling"], file_data)
    if params["button"] == 4:
        Envelopspectrum(file_data, params["frequency_sampling"], params["esLim_lower"], params["esLim_upper"])

# 读取外部存储的button值
s = open("./user_data/preprocessing.txt", 'r')  
button = s.read()
s.close()


if __name__ == '__main__':
    # return Dict
    params = collections.defaultdict()
    params["button"] = int(button)
    main()
