import os
import pickle
import numpy as np
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Material Transformation Predictor",
    page_icon="🔬",
    layout="centered"
)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

st.title("Material Predictor")
st.write("Enter material parameters to calculate output classification.")

if model is None:
    st.error("Model file (`model.pkl`) not found on server.")
else:
    # Input Form
    with st.form("prediction_form"):
        temp = st.number_input("Temperature (°C)", value=0.0, format="%.4f")
        pressure = st.number_input("Pressure (kPa)", value=0.0, format="%.4f")
        
        # Auto-calculate temperature x pressure interaction term
        auto_txp = temp * pressure
        temp_x_pressure = st.number_input("Temperature × Pressure", value=auto_txp, format="%.4f")
        
        fusion = st.number_input("Material Fusion Metric", value=0.0, format="%.4f")
        transformation = st.number_input("Material Transformation Metric", value=0.0, format="%.4f")
        
        submitted = st.form_submit_button("Run Prediction")

        if submitted:
            try:
                features = np.array([[temp, pressure, temp_x_pressure, fusion, transformation]])
                pred = model.predict(features)[0]
                
                st.success("Prediction Successful!")
                st.metric(label="Predicted Class", value=str(pred))
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
