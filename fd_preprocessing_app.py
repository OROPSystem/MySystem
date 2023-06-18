import streamlit as st
import collections
import numpy as np
import pathlib
import os
from itertools import islice
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import fftpack, signal
import pandas as pd
import pywt
import stockwell
 
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
        dataset = st.selectbox(label="数据集", options=[n.stem for n in datasets_name if n.is_dir()])
        params["dataset"] = dataset
        
        # Choose the load
    
        load_range = pathlib.Path(f"./user_data/test/fd_datasets/{dataset}")
        loads = st.selectbox(label="负载", options=os.listdir(load_range))
        params['load'] = loads
        
    with col2:    
        fault_type_range = pathlib.Path(f"./user_data/test/fd_datasets/{dataset}/{loads}")
        chosen_fault_type = st.selectbox(label="选择故障类型", options=os.listdir(fault_type_range))
        params["chosen_fault_type"] = chosen_fault_type
        
        signal_size = st.number_input(label='信号长度', value=1200)
        params['signal_size'] = signal_size
    
    col1, col2 = st.columns([2,2], gap='small')
    
    # page in col1
    with col1:
        st.header("数据预处理类型")
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
            button5 = st.button('Wavelet', use_container_width=True)
            if button5:
                params["button"] = 5
            button6 = st.button('Stockwell', use_container_width=True)
            if button6:
                params["button"] = 6
            button7 = st.button('STFT', use_container_width=True)
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
        st.header("参数设置")
        if "button" in params.keys():
            if params["button"] == 1:
                st.text("当前数据预处理类型:FFT")
                with st.form("template_input"):
                    frequency_sampling = st.number_input("采样频率(*1e3 Hz):", value=25.6)
                    submit = st.form_submit_button()
                if submit:
                    params['flag'] = True
                    params["frequency_sampling"] = frequency_sampling * 1000
                           
            if params["button"] == 2:
                st.text("当前数据预处理类型:SDP")
                with st.form("template_input"):
                    submit = st.form_submit_button()
                if submit:
                    params['flag'] = True
                    
            if params["button"] == 3:
                st.text("当前数据预处理类型:CWT")
                with st.form("template_input"):
                    frequency_sampling = st.number_input("采样频率(*1e3 Hz):", value=25.6)
                    submit = st.form_submit_button()
                if submit:
                    params['flag'] = True
                    params["frequency_sampling"] = frequency_sampling * 1000
            
            if params["button"] == 4:
                st.text("当前数据预处理类型:Envelop Spectrum")
                with st.form("template_input"):
                    frequency_sampling = st.number_input("采样频率(*1e3 Hz):", value=25.6)
                    esLim_lower = st.number_input("esLim_lower", value=0.0)
                    esLim_upper = st.number_input("esLim_upper", value=500.0)
                    submit = st.form_submit_button()
                if submit:
                    params['flag'] = True
                    params["frequency_sampling"] = frequency_sampling * 1000
                    params["esLim_lower"] = esLim_lower
                    params["esLim_upper"] = esLim_upper
            
            if params["button"] == 5:
                st.text("当前数据预处理类型:Wavelet")
                with st.form("template_input"):
                    frequency_sampling = st.number_input("采样频率(*1e3 Hz):", value=25.6)
                    Fmax = st.number_input("Frequency Max", value=0.0)
                    Fmin = st.number_input("Frequency Min", value=0.0)
                    df = st.number_input("Frequency Division", value=1.0)
                    wave_name = st.text_input("Wave Name", value='morl')
                    submit = st.form_submit_button()
                if submit:
                    params['flag'] = True
                    params["frequency_sampling"] = frequency_sampling * 1000
                    params["Fmax"] = Fmax
                    params["Fmin"] = Fmin
                    params["df"] = df
                    params["wave_name"] = wave_name
                    
            if params["button"] == 6:
                st.text("当前数据预处理类型:Stockwell")
                with st.form("template_input"):
                    frequency_sampling = st.number_input("采样频率(*1e3 Hz):", value=25.6)
                    Fmax = st.number_input("Frequency Max", value=0.0)
                    Fmin = st.number_input("Frequency Min", value=0.0)
                    submit = st.form_submit_button()
                if submit:
                    params['flag'] = True
                    params["frequency_sampling"] = frequency_sampling * 1000
                    params["Fmax"] = Fmax
                    params["Fmin"] = Fmin
                    
            if params["button"] == 7:
                st.text("当前数据预处理类型:STFT")
                with st.form("template_input"):
                    frequency_sampling = st.number_input("采样频率(*1e3 Hz):", value=25.6)
                    window = st.text_input("Window", value="hann")
                    nperseg = st.number_input("n_per_seg", value=256)
                    submit = st.form_submit_button()
                if submit:
                    params['flag'] = True
                    params["frequency_sampling"] = frequency_sampling * 1000
                    params["window"] = window
                    params["nperseg"] = nperseg
                    
    st.set_option('deprecation.showPyplotGlobalUse', False)   
     
    col3 = st.container()
    # page in col3
    with col3:
        st.header("数据转换")
        if params['flag']:
            preprocessing(params)

##### button 1
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
    plt.figure(figsize=[6, 4.5])
    plt.plot(result)
    plt.title('Frequency-Domain Data by FFT')
    st.pyplot()
    
##### button 2   
##### SDP
def SDPimage(s):
    """
    # 获取 symmetrized dot pattern 结果
    # s 输入信号
    """
    plt.figure(figsize=[6, 4.5])
    for i in range(1,7):
        rs = (s - np.min(s)) / (np.max(s) - np.min(s))
        th = (72*i + ((s - np.min(s)) / (np.max(s) - np.min(s)) * 36))[2:]     
        rs = rs[:-2]
        plt.title('Frequency-Domain Data by SDP')
        plt.polar(th*(np.pi/180), rs, '.b')
    st.pyplot()
 
##### button 3    
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
    
    plt.figure(figsize=[6, 4.5])
    plt.contourf(t, freqs, np.abs(cwtmatr), 100)
    plt.title('Frequency-Domain Data by CWT')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.colorbar()
    st.pyplot()

##### button 4
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
    
    plt.figure(figsize=[6, 4.5])
    plt.plot(f1, es_raw_abs)
    plt.title('Envelope spectrum')
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Acc(normalized)')
    plt.xlim([esLim_lower, esLim_upper])
    st.pyplot()

##### button 5
##### 小波变换
def wavelet(Fs,data, Fmin=0, Fmax=0, df=1, wave_name='morl'):
    """
    对输入信号进行连续小波变换
    :param Fs:    采样频率
    :param data:    待分析的序列
    :param Fmax,Fmin: 频率范围
    :param df:  频率分辨率
    :param wave_name: 所使用的小波名称
    
    """
    if Fmax==0 and Fs>0:
        Fmax=Fs/2
    Fc=pywt.central_frequency(wave_name)
    total_scalas=int((Fmax-Fmin)/df)
    scales=(Fs*Fc)/(Fmin+df*np.arange(total_scalas, 1, -1))
    [result, freq]=pywt.cwt(data, scales, wave_name,1.0/Fs)
    
    plt.figure(figsize=[6, 4.5])
    plt.contourf(np.arange(len(data))*(1/Fs),freq,abs(result))
    st.pyplot()

##### button 6
##### Stockwell 变换
def s_transform(Fs, data, Fmax=0, Fmin=0):
    """
    对输入信号进行S(Stockwell)变换
    :param Fs:    采样频率
    :param data:    待分析的序列
    :param Fmax,Fmin: 频率范围
    """
    if Fmax==0 and Fs>0:
        Fmax=Fs/2
    df=Fs/(len(data)-1)
    Fmin_sample=int(Fmin/df)
    Fmax_sample=int(Fmax/df)
    result=stockwell.st.st(data,Fmin_sample,Fmax_sample)
    extent=(0,1/df, Fmin,Fmax)
    
    plt.figure(figsize=[6, 4.5])
    plt.contourf(abs(result),origin='lower',extent=extent)
    st.pyplot()

##### button 7
##### 短时傅里叶变换
def stft(Fs, data, window="hann", nperseg=256):
    """
    对输入信号进行短时傅里叶变换
    :param Fs:    采样频率
    :param data:  待分析的序列
    :param window: 窗口类型
    :nperseg:  每个段得长度
    """
    noverlap=nperseg-1
    freq, t, result=signal.stft(data, Fs, window=window, nperseg=nperseg, noverlap=noverlap)
    extent=(t[0], t[-1], freq[0], freq[-1])
    
    plt.figure(figsize=[6, 4.5])
    plt.contourf(abs(result), origin='lower', extent=extent)
    st.pyplot()

# preprocessing
def preprocessing(params):
    col1, col2 = st.columns(2)
    chosen_fault = "./user_data/test/fd_datasets/{}/{}/{}".format(params["dataset"], params["load"], params['chosen_fault_type'])
    
    file = open(chosen_fault, "r", encoding='gb18030', errors='ignore')
    file_data = []
    for line in islice(file, 22, None):   # Skip the first 22 lines
        line = line.rstrip()
        word = line.split("\t", 5)         # Separated by \t
        file_data.append(eval(word[2]))   # Take a vibration signal in the x direction as input
    
    file_data = np.array(file_data).reshape(-1)[:params['signal_size']]
    plt.figure(figsize=[6, 4.5])
    plt.plot(file_data)
    plt.title('Time-Series Data')
    with col1:
        st.pyplot()
    
    with col2:
        if params["button"] == 1:
            FFT(params["frequency_sampling"], file_data)
        if params["button"] == 2:
            SDPimage(file_data)
        if params["button"] == 3:
            CWTimage(params["frequency_sampling"], file_data)
        if params["button"] == 4:
            Envelopspectrum(file_data, params["frequency_sampling"], params["esLim_lower"], params["esLim_upper"])
        if params["button"] == 5:
            wavelet(params["frequency_sampling"], file_data, params["Fmin"], params["Fmax"], params["df"], params["wave_name"])
        if params["button"] == 6:
            s_transform(params["frequency_sampling"], file_data, params["Fmax"], params["Fmin"])
        if params["button"] == 7:
            stft(params["frequency_sampling"], file_data, params["window"], params["nperseg"])
        
# 读取外部存储的button值
s = open("./user_data/preprocessing.txt", 'r')  
button = s.read()
s.close()


if __name__ == '__main__':
    # return Dict
    params = collections.defaultdict()
    params["button"] = int(button)
    main()
