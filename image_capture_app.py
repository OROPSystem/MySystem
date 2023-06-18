import streamlit as st
import collections
from PIL import Image
import numpy as np
import cv2 as cv
import os

st.set_page_config(
    page_title=None,
    page_icon=None,
    layout="wide",
)

def get_streamlit_params():
    
    root_dir = "/home/user/system/MySystem/user_data/captures"
    col1, col2 = st.columns([2, 3], gap="large")
    
    with col1:
        filename = st.text_input("保存文件名")
        
    with col2:
        if filename:
            cap = st.camera_input("图像抓取")
            if cap is not None:
                img = Image.open(cap)
                img_save_path = root_dir + "/" + filename + ".jpg"
                img.save(img_save_path)
                img.close()
                
                with col1:
                    st.text("抓取成功！！！")
                    st.text(f"文件保存位置：{img_save_path}")
        else:
            st.text("请先输入保存的文件名！！")
                
            
            
def main():
    params = get_streamlit_params()

if __name__ == '__main__':
    main()