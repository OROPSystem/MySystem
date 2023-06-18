import streamlit as st
import collections
import numpy as np
import cv2 as cv

st.set_page_config(
    page_title=None,
    page_icon=None,
    layout="wide",
)

def get_streamlit_params():
    params = collections.defaultdict()
    col1, col2 = st.columns([1, 3], gap="large")
    with col1:
        filename = st.text_input("Filename", help="")
    with col2:
        cap = st.camera_input("Grab Images")
        if cap:
            st.image(cap)
            print("Got it")
            
def main():
    params = get_streamlit_params()

if __name__ == '__main__':
    main()