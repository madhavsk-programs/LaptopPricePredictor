import streamlit as st
import pickle
import numpy as np
import pandas as pd
import xgboost  # just to ensure module is available

# load model and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("💻 Laptop Price Predictor")

# UI Inputs
company = st.selectbox('Brand', df['Company'].unique())
type_ = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM (in GB)', [2,4,6,8,12,16,24,32,64])
weight = st.number_input('Weight of the Laptop')
touchscreen = st.selectbox('TouchScreen', ['No','Yes'])
ips = st.selectbox('IPS', ['No','Yes'])
screen_size = st.number_input('Screen Size (in inches)')
resolution = st.selectbox(
    'Screen Resolution',
    ['1920x1080','1366x768','1600x900','3840x2160',
     '3200x1800','2880x1800','2560x1600','2560x1440','2304x1440']
)
cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.selectbox('HDD (in GB)', [0,128,256,512,1024,2048])
ssd = st.selectbox('SSD (in GB)', [0,8,128,256,512,1024])
gpu = st.selectbox('GPU', df['Gpu brand'].unique())
os = st.selectbox('OS', df['os'].unique())

# Prediction button
if st.button('Predict Price'):
    # convert categorical to numeric where needed
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # calculate PPI
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2 + Y_res**2) ** 0.5) / screen_size

    # create input dataframe (important: DataFrame, not np.array)
    query = pd.DataFrame([[company, type_, ram, weight, touchscreen,
                           ips, ppi, cpu, hdd, ssd, gpu, os]],
                         columns=['Company','TypeName','Ram','Weight',
                                  'Touchscreen','Ips','ppi','Cpu brand',
                                  'HDD','SSD','Gpu brand','os'])
    # predict log(price)
    log_prediction = pipe.predict(query)[0]

    # convert back to original scale
    prediction = int(np.exp(log_prediction))

    st.success(f"💻 Predicted Laptop Price: ₹ {prediction}")

