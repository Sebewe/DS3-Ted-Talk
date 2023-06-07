import pandas as pd
import streamlit as st
import pickle
import plotly.io as pio
import numpy as np
import os
from PIL import Image

GET_DIR = lambda f: os.path.join('streamlit_resources', f)
GET_FILE = lambda f: open(GET_DIR(f), "rb")

@st.cache_resource
def load_log_model():
    return pickle.load(GET_FILE("log_model.sav"))

@st.cache_resource
def load_ovr_model():
    return pickle.load(GET_FILE("ovr_model.sav"))

@st.cache_resource
def log_model_accuracy_graph():
    return pio.from_json(GET_FILE("log_model_accuracy.json").read())

@st.cache_resource
def log_model_accuracy2_graph():
    return pio.from_json(GET_FILE("log_model_accuracy_2.json").read())

@st.cache_resource
def ovr_model_accuracy_graph():
    return pio.from_json(GET_FILE("ovr_model_accuracy.json").read())

@st.cache_resource
def ovr_model_accuracy2_graph():
    return pio.from_json(GET_FILE("ovr_model_accuracy_2.json").read())


st.set_page_config(
    page_title="Text Classification",
    page_icon=Image.open(GET_DIR('ds3.png'))
)

log_tab, ovr_tab = st.tabs(['Linear SVC', 'OVR Logistic Regression'])
with log_tab:
    @st.cache_data
    def test_classify_log(prediction_input):
        model = load_log_model()
        str_arr = pd.Series(prediction_input).to_frame()
        str_arr = str_arr.rename(columns={0: "clean_scripts"})
        prediction = model.predict(str_arr)[0]
        return prediction
    
    #this would be the explanantion about this model. 
    st.markdown("""
        # Linear SVC Model
        ### 76.1% Training Accuracy, 55.8% Test Accuracy
    """)
    st.plotly_chart(log_model_accuracy_graph())
    st.plotly_chart(log_model_accuracy2_graph())
    prediction_input = st.text_area("Input the string you'd like us to classify:", key="log_prediction_input")

    _, _, _, middle, _, _, _ = st.columns(7)
    if middle.button("classify", key="log_classify_button"):
        prediction = test_classify_log(prediction_input)
        st.markdown(f"The model predicted: **{prediction}**")


with ovr_tab:
    @st.cache_data
    def test_classify_ovr(prediction_input):
        model = load_ovr_model()
        str_arr = pd.Series(prediction_input).to_frame()
        str_arr = str_arr.rename(columns={0: "clean_scripts"})
        prediction = model.predict(str_arr)[0]
        return prediction

    #this would be the explanantion about this model. 
    st.markdown("""
        # OneVsRest Classifier + Logistic Regression
        ### 78% Training Accuracy, 58.9% Test Accuracy
    """)
    st.plotly_chart(ovr_model_accuracy_graph())
    st.plotly_chart(ovr_model_accuracy2_graph())
    prediction_input = st.text_area("Input the string you'd like us to classify:", key="ovr_prediction_input")

    _, _, _, middle, _, _, _ = st.columns(7)
    if middle.button("classify", key="ovr_classify_button"):
        prediction = test_classify_ovr(prediction_input)
        st.markdown(f"The model predicted: **{prediction}**")