import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import FunctionTransformer

# Must match the function used when training the saved model
def convert_to_string(X):
    return X.astype(object).where(pd.notna(X), np.nan).astype(str)

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Smart Traffic Risk Prediction System",
    page_icon="🚦",
    layout="wide"
)

# -------------------------------
# Load trained model and encoder
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "traffic_severity_model.pkl"
ENCODER_PATH = BASE_DIR / "models" / "severity_label_encoder.pkl"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder

st.title("🚦 Smart Traffic Risk Prediction System")
st.write(
    "Predict the recorded accident severity class using the "
    "trained machine learning model."
)

try:
    model, label_encoder = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files were not found. Please check that both saved files "
        "are inside the project's models folder."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not load the model files: {error}")
    st.stop()

st.success("Trained model loaded successfully!")

# -------------------------------
# Input form
# -------------------------------
st.subheader("Enter Accident Conditions")
st.caption(
    "Choose values from the dataset. Numeric category codes are shown as "
    "stored in the dataset; their meanings must be confirmed from its codebook."
)

# Load the prepared dataset to populate input choices
DATA_PATH = BASE_DIR / "dataa" / "processed" / "accident_data_prepared.csv"

try:
    df_options = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error(
        "Prepared dataset not found. Expected file: "
        "data/processed/accident_data_prepared.csv"
    )
    st.stop()

# These must match the feature columns used during model training.
categorical_features = [
    "Day_of_Week",
    "Accident_Location_A",
    "Accident_Location_A_Chainage_km_RoadSide",
    "Causes_D",
    "Road_Feature_E",
    "Road_Condition_F",
    "Weather_Conditions_H",
    "Vehicle_Type_Involved_J_V1",
    "Vehicle_Type_Involved_J_V2",
    "Accident_Month",
    "Accident_DayOfWeek",
    "Accident_Hour",
    "Is_Weekend"
]

numerical_features = [
    "Accident_Location_A_Chainage_km",
    "Accident_Year"
]

feature_columns = categorical_features + numerical_features

with st.form("prediction_form"):
    st.write("### Time and location")

    col1, col2 = st.columns(2)

    with col1:
        day_value = st.selectbox(
            "Day of week (dataset code)",
            sorted(df_options["Day_of_Week"].dropna().unique().tolist())
        )
        location_value = st.selectbox(
            "Accident location (dataset code)",
            sorted(df_options["Accident_Location_A"].dropna().unique().tolist())
        )
        chainage_value = st.number_input(
            "Location chainage (km)",
            min_value=0.0,
            value=float(df_options["Accident_Location_A_Chainage_km"].median())
        )
        roadside_value = st.selectbox(
            "Roadside category (dataset code)",
            sorted(
                df_options[
                    "Accident_Location_A_Chainage_km_RoadSide"
                ].dropna().unique().tolist()
            )
        )

    with col2:
        month_value = st.selectbox("Month", list(range(1, 13)))
        hour_value = st.slider("Hour of day", 0, 23, 12)
        year_value = st.number_input(
            "Accident year",
            min_value=1900,
            max_value=2100,
            value=int(df_options["Accident_Year"].median())
        )
        is_weekend = st.selectbox("Is weekend?", [0, 1])

    st.write("### Road, weather and vehicle conditions")
    col3, col4 = st.columns(2)

    with col3:
        cause_value = st.selectbox(
            "Cause (dataset code)",
            sorted(df_options["Causes_D"].dropna().unique().tolist())
        )
        road_feature_value = st.selectbox(
            "Road feature (dataset code)",
            sorted(df_options["Road_Feature_E"].dropna().unique().tolist())
        )
        road_condition_value = st.selectbox(
            "Road condition (dataset code)",
            sorted(df_options["Road_Condition_F"].dropna().unique().tolist())
        )
        weather_value = st.selectbox(
            "Weather (dataset code)",
            sorted(df_options["Weather_Conditions_H"].dropna().unique().tolist())
        )

    with col4:
        vehicle1_value = st.selectbox(
            "Vehicle type 1 (dataset code)",
            sorted(
                df_options[
                    "Vehicle_Type_Involved_J_V1"
                ].dropna().unique().tolist()
            )
        )

        vehicle2_options = df_options[
            "Vehicle_Type_Involved_J_V2"
        ].dropna().unique().tolist()

        vehicle2_options = sorted(vehicle2_options)
        vehicle2_options = ["Missing"] + vehicle2_options

        vehicle2_value = st.selectbox(
            "Vehicle type 2 (dataset code)",
            vehicle2_options
        )

    submitted = st.form_submit_button("Predict Severity")

if submitted:
    # Derive day-of-week code from the selected dataset value.
    # This assumes Day_of_Week and Accident_DayOfWeek use the same coding.
    # Verify this assumption against the dataset before relying on predictions.
    input_data = pd.DataFrame([{
        "Day_of_Week": day_value,
        "Accident_Location_A": location_value,
        "Accident_Location_A_Chainage_km_RoadSide": roadside_value,
        "Causes_D": cause_value,
        "Road_Feature_E": road_feature_value,
        "Road_Condition_F": road_condition_value,
        "Weather_Conditions_H": weather_value,
        "Vehicle_Type_Involved_J_V1": vehicle1_value,
        "Vehicle_Type_Involved_J_V2": vehicle2_value,
        "Accident_Month": month_value,
        "Accident_DayOfWeek": day_value,
        "Accident_Hour": hour_value,
        "Is_Weekend": is_weekend,
        "Accident_Location_A_Chainage_km": chainage_value,
        "Accident_Year": year_value
    }], columns=feature_columns)

    # Match the saved model's input feature order.
    encoded_prediction = model.predict(input_data).astype(int)
    severity_prediction = label_encoder.inverse_transform(
        encoded_prediction
    )[0]

    st.subheader("Prediction Result")
    st.metric("Predicted severity class", int(severity_prediction))
    st.info(
        "This is a predicted severity class, not a probability that an "
        "accident will occur. Confirm the meaning of each class using "
        "the dataset documentation."
    )
