import streamlit as st
import collections
import numpy as np

st.set_page_config(
    page_title=None,
    page_icon=None,
    layout="wide",
)



def main():
    # create three cols
    col1, col2, col3 = st.columns([2,3,3], gap='small')
    
    # page in col1
    with col1:
        st.header("head1")
        clicked = False
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            button1 = st.button('button1')
            if button1:
                params["button"] = 1 
            button2 = st.button('button2')
            if button2:
                params["button"] = 2    
            button3 = st.button('button3')
            if button3:
                params["button"] = 3  
            button4 = st.button('button4')
            if button4:
                params["button"] = 4  
        with col1_2:
            button5 = st.button('button5')
            if button5:
                params["button"] = 5
            button6 = st.button('button6')
            if button6:
                params["button"] = 6
            button7 = st.button('button7')
            if button7:
                params["button"] = 7
            button8 = st.button('button8')
            if button8:
                params["button"] = 8

    # page in col2
    with col2:
        st.header("head2")
        if "button" in params.keys():
            if params["button"] == 1:
                with st.form("template_input"):
                    input1 = st.text_input("Input1_1")
                    input2 = st.text_input("Input1_2")
                    submit = st.form_submit_button()
                
                if submit:
                    params["input1"] = input1
                    params["input2"] = input2
                    
            if params["button"] == 2:
                with st.form("template_input"):
                    input1 = st.text_input("Input2_1")
                    input2 = st.text_input("Input2_2")
                    input3 = st.text_input("Input2_3")
                    submit = st.form_submit_button()
                
                if submit:
                    params["input1"] = input1
                    params["input2"] = input2
                    params["input3"] = input3
                    
    img1, img2 = preprocessing(params)
        
    # page in col3
    with col3:
        st.header("head3")
        st.text("chart1")
        st.image(img1)
        st.text("chart2")
        st.image(img2)
    return params["button"]
    
def preprocessing(params):
     img1 = 'static/images/loading.jpg'
     img2 = 'static/images/loading.jpg'
     return img1, img2   

button = 1 
if __name__ == '__main__':
    # return Dict
    params = collections.defaultdict()
    params["button"] = button
    main()
    print(button)

