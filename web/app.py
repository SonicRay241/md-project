import streamlit as st
import pandas as pd
import requests
import dotenv
import os
import time
import random

dotenv.load_dotenv()
api_url = os.getenv("API_URL")

st.set_page_config(
    page_title="Obesity Classifier",
    page_icon=":cake:",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items=None
)

st.title("Obesity Classification Prediction with XGBoost")

if "gender" not in st.session_state:
    st.session_state.gender = "Male"
    st.session_state.age = 18
    st.session_state.height = 1.80
    st.session_state.weight = 78.0
    st.session_state.family_history_with_overweight = "no"
    st.session_state.favc = "no"
    st.session_state.fcvc = 2.5
    st.session_state.ncp = 3
    st.session_state.caec = "no"
    st.session_state.smoke = "no"
    st.session_state.ch20 = 2.1
    st.session_state.scc = "yes"
    st.session_state.faf = 1.9
    st.session_state.tue = 1.4
    st.session_state.calc = "no"
    st.session_state.mtrans = "Walking"

with st.sidebar:
    st.title("Presets")

    if st.button("Preset 1"):
        st.session_state.gender = "Male"
        st.session_state.age = 19
        st.session_state.height = 1.76
        st.session_state.weight = 83.0
        st.session_state.family_history_with_overweight = "yes"
        st.session_state.favc = "yes"
        st.session_state.fcvc = 2.15
        st.session_state.ncp = 2.15
        st.session_state.caec = "Sometimes"
        st.session_state.smoke = "no"
        st.session_state.ch20 = 2.3
        st.session_state.scc = "no"
        st.session_state.faf = 1.3
        st.session_state.tue = 2.5
        st.session_state.calc = "no"
        st.session_state.mtrans = "Public_Transportation"

    if st.button("Preset 2"):
        st.session_state.gender = "Female"
        st.session_state.age = 24
        st.session_state.height = 1.58
        st.session_state.weight = 56
        st.session_state.family_history_with_overweight = "yes"
        st.session_state.favc = "no"
        st.session_state.fcvc = 2
        st.session_state.ncp = 2.15
        st.session_state.caec = "Sometimes"
        st.session_state.smoke = "no"
        st.session_state.ch20 = 1.9
        st.session_state.scc = "no"
        st.session_state.faf = 1.1
        st.session_state.tue = 1.9
        st.session_state.calc = "Sometimes"
        st.session_state.mtrans = "Public Transportation"

st.selectbox("Gender", ["Male", "Female"], key="gender")
st.number_input("Age", min_value=1, max_value=120, step=1, key="age")
st.number_input("Height (in meters)", min_value=0.1, max_value=3.0, step=0.01, key="height")
st.number_input("Weight (in kg)", min_value=1.0, max_value=300.0, step=0.1, key="weight")
st.selectbox("Family History with Overweight", ["no", "yes"], key="family_history_with_overweight")
st.selectbox("Frequent Consumption of High Calorie Food", ["no", "yes"], key="favc")
st.slider("Vegetable Consumption Ratio in Meals", min_value=1.0, max_value=3.0, step=0.01, key="fcvc")
st.number_input("Average Count of Main Meals in a Day", min_value=1.0, max_value=3.0, step=0.01, key="ncp")
st.selectbox("Frequency of Consuming Snacks Between Meals", ["no", "Sometimes", "Frequently", "Always"], key="caec")
st.selectbox("Smoking", ["no", "yes"], key="smoke")
st.number_input("Water Consumption (in litres)", min_value=1.0, max_value=3.0, step=0.01, key="ch20")
st.selectbox("Currently Tracking Calorie Intake", ["no", "yes"], key="scc")
st.slider("Physical Activity Frequency", min_value=0.0, max_value=3.0, step=0.01, key="faf")
st.slider("Time Spent with Electronics", min_value=0.0, max_value=3.0, step=0.01, key="tue")
st.selectbox("Frequency of Consuming Alcohol", ["no", "Sometimes", "Frequently", "Always"], key="calc")
st.selectbox("Main Method of Transport", ["Public Transportation", "Automobile", "Bike", "Walking", "Motorbike"], key="mtrans")

submit_btn = st.button("Classify Me!")

if submit_btn:
    with st.spinner("🧠 Thinking...", show_time = True):
        time.sleep(random.uniform(0.5, 3.0))
        
        data = st.session_state.to_dict()
        data["mtrans"] = data["mtrans"].replace(" ", "_")

        req = requests.post(api_url + "/predict", json=data)
        res = req.json()

        pred = res["prediction"].replace("_", " ")

        st.success(f"Classification Result: **{pred}**")