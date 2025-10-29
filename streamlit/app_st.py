import streamlit as st
import pandas as pd
import pickle
import os

# Load model and transformer
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'save_model', 'decision_tree_regressor_model.pkl')
transformer_path = os.path.join(base_dir, 'save_model', 'transformer.pkl')

try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    with open(transformer_path, 'rb') as file:
        preprocessor = pickle.load(file)
except FileNotFoundError:
    st.error("Model or transformer file not found. Please check the 'save_model' folder.")
    st.stop()

# Dropdown options from your HTML
from_options = [
    "Recife (PE)", "Florianopolis (SC)", "Brasilia (DF)", "Aracaju (SE)",
    "Salvador (BH)", "Campo Grande (MS)", "Sao Paulo (SP)", "Natal (RN)", "Rio de Janeiro (RJ)"
]

to_options = [
    "Florianopolis (SC)", "Recife (PE)", "Brasilia (DF)", "Salvador (BH)",
    "Aracaju (SE)", "Campo Grande (MS)", "Sao Paulo (SP)", "Natal (RN)", "Rio de Janeiro (RJ)"
]

flight_classes = ["firstClass", "premium", "economic"]
agencies = ["CloudFy", "Rainbow", "FlyingDrops"]

# Streamlit UI
st.title("✈️ Travel Price Prediction")
st.markdown("Enter customer and flight details to get a price estimate.")

with st.form("prediction_form"):
    from_location = st.selectbox("From", from_options)
    to_location = st.selectbox("To", to_options)
    distance = st.number_input("Distance (in km)", min_value=0.0, step=1.0)
    flight_type = st.selectbox("Flight Class", flight_classes)
    agency = st.selectbox("Agency", agencies)

    submitted = st.form_submit_button("Predict Price")

if submitted:
    input_data = {
        'from': from_location,
        'to': to_location,
        'distance': distance,
        'flightType': flight_type,
        'agency': agency
    }

    try:
        df = pd.DataFrame([input_data])
        transformed_data = preprocessor.transform(df)
        prediction = model.predict(transformed_data)[0]

        st.success(f"💰 Predicted Price: ₹{prediction:,.2f}")
        st.markdown("### Submitted Data")
        st.json(input_data)

        st.markdown("### Preprocessed Data")
        st.write(transformed_data)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
