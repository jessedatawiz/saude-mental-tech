import streamlit as st
import pandas as pd
import joblib

# Step 1: Load the Saved Pipeline
pipeline_path = 'pipeline.pkl'
pipeline = joblib.load(pipeline_path)

# Step 2: Create Streamlit App
st.title("Mental Health Prediction")

# Step 3: File Upload and Prediction
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    predictions = pipeline.predict(data)

    # Display the predictions or perform further operations
    st.write(predictions)