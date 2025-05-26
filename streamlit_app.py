import streamlit as st
import pickle

# Model load karo (agar pickle model hai)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('My ML Model App')

# Input from user
input_value = st.number_input('Enter a number')

if st.button('Predict'):
    prediction = model.predict([[input_value]])
    st.write(f'Prediction: {prediction[0]}')

