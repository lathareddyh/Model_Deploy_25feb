# Load the Libraries
import numpy as np
import joblib
import streamlit as st

#Load the trained model
model = joblib.load('model_scaled.pkl')
scale = joblib.load('scaled.pkl')

#streamlit app title 

st.title('Machine Learning Model Deployment')
st.write('Enter your medical details to know about your diabetic status')

#Define the input fields
st.sidebar.header('Your Medical Records')

preg = st.sidebar.number_input('PREG', min_value=0.0, max_value=1000.0, value=50.0, step=0.1)
plas = st.sidebar.number_input('plas', min_value=0.0, max_value=1000.0, value=50.0, step=0.1)
pres = st.sidebar.number_input('pres', min_value=0.0, max_value=1000.0, value=50.0, step=0.1)
skin = st.sidebar.number_input('skin', min_value=0.0, max_value=1000.0, value=50.0, step=0.1)
test = st.sidebar.number_input('test', min_value=0.0, max_value=1000.0, value=50.0, step=0.1)
mass = st.sidebar.number_input('mass', min_value=0.0, max_value=1000.0, value=50.0, step=0.1)
pedi = st.sidebar.number_input('pedi', min_value=0.0, max_value=1000.0, value=50.0, step=0.1)
age = st.sidebar.number_input('Age', min_value=0.0, max_value=1000.0, value=50.0, step=0.1)

input_data = np.array([[preg,plas,pres,skin,test,mass,pedi,age]])
scaled_input = scale.transform(input_data)

if st.sidebar.button('Predict'):
    # Model Prediction
    prediction = model.predict(scaled_input)
    st.success(f'Prediction:{prediction[0]}')
