import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load ML Model
with open('ml_model.pkl', 'rb') as f:
    model_ml = pickle.load(f)

# Load Scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load DL Model
model_dl = load_model('dl_model.h5')

# Function to preprocess input
def preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked):
    sex = 1 if sex == 'male' else 0
    embarked_dict = {'C': 0, 'Q': 1, 'S': 2}
    embarked = embarked_dict.get(embarked, 2)

    input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    scaled_input = scaler.transform(input_data)
    return scaled_input

# Streamlit UI
st.title("🚢 Titanic Survival Prediction App (ML + DL)")

st.sidebar.title("Model Selection")
model_option = st.sidebar.selectbox("Choose Model", ["Machine Learning", "Deep Learning"])

pclass = st.selectbox("Passenger Class (1 = Upper, 2 = Middle, 3 = Lower)", [1, 2, 3])
sex = st.radio("Sex", ['male', 'female'])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Siblings/Spouse Aboard", min_value=0, max_value=10, value=1)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=1)
fare = st.number_input("Fare", min_value=0.0, value=50.0)
embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'])

if st.button("Predict Survival"):
    final_input = preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked)

    if model_option == "Machine Learning":
        prediction = model_ml.predict(final_input)[0]
    else:
        raw_pred = model_dl.predict(final_input)[0][0]
        prediction = 1 if raw_pred > 0.5 else 0
        st.write(f"DL Model Confidence: {raw_pred:.2%}")

    if prediction == 1:
        st.success("🎉 Passenger would have SURVIVED!")
    else:
        st.error("💀 Passenger would NOT have survived.")
