import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
import plotly.express as px
import plotly.graph_objects as go
import shap
from scipy import sparse
import requests

# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Traffic Risk Prediction System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------
# Existing project paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "traffic_severity_model.pkl"
ENCODER_PATH = BASE_DIR / "models" / "severity_label_encoder.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "accident_data_prepared.csv"
LOCATION_PATH = BASE_DIR / "data" / "processed" / "location_segments.csv"

# Keep this feature schema and order aligned with model training.
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
    "Is_Weekend",
]
numerical_features = [
    "Accident_Location_A_Chainage_km",
    "Accident_Year",
]
feature_columns = categorical_features + numerical_features

# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #172b4d;
        --muted: #66758b;
        --line: #e4eaf2;
        --panel: #ffffff;
        --canvas: #f5f8fc;
        --blue: #2463eb;
        --blue-dark: #1748b8;
        --teal: #0f9f91;
    }
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background: var(--canvas);
        color: var(--ink);
    }
    .block-container {
        max-width: 1320px;
        padding-top: 1.1rem;
        padding-bottom: 3rem;
    }
    header[data-testid="stHeader"] {
        background: rgba(245,248,252,0.92);
    }
    .hero {
        background: linear-gradient(120deg, #112c58 0%, #1d4e91 58%, #247c94 100%);
        border-radius: 24px;
        padding: 30px 34px;
        color: white;
        position: relative;
        overflow: hidden;
        margin-bottom: 22px;
        box-shadow: 0 14px 34px rgba(25, 65, 117, .14);
    }
    .hero:after {
        content: "";
        position: absolute;
        right: -42px;
        top: -90px;
        width: 250px;
        height: 250px;
        border: 1px solid rgba(255,255,255,.17);
        border-radius: 50%;
        box-shadow: 0 0 0 28px rgba(255,255,255,.035),
                    0 0 0 58px rgba(255,255,255,.025);
    }
    .hero-kicker {
        color: #b8e7f1;
        text-transform: uppercase;
        letter-spacing: .15em;
        font-size: .73rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .hero h1 {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(1.75rem, 3vw, 2.65rem);
        line-height: 1.12;
        letter-spacing: -.04em;
        margin: 0 0 12px 0;
        color: #fff;
        max-width: 760px;
    }
    .hero p {
        color: #e0ebf8;
        max-width: 720px;
        font-size: 1rem;
        margin: 0;
        line-height: 1.65;
    }
    .brand-lockup {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 30px 0 18px 0;
        min-height: 70px;
    }
    .brand-icon {
        display: grid;
        place-items: center;
        width: 58px;
        height: 58px;
        flex: 0 0 58px;
        background: linear-gradient(145deg, #e5edff, #d3e2ff);
        border: 1px solid #c7d8fa;
        border-radius: 16px;
        font-size: 1.65rem;
        padding: 5px;
        box-shadow: 0 5px 14px rgba(36, 99, 235, 0.12);
    }
    .brand-name {
        color: #142b50 !important;
        font-family: 'Manrope', sans-serif;
        font-size: 1.35rem;
        font-weight: 800;
        line-height: 1.35;
        letter-spacing: -0.035em;
        opacity: 1 !important;
    }
    .brand-subtitle {
        color: #425875 !important;
        font-size: 0.9rem;
        font-weight: 600;
        line-height: 1.5;
        opacity: 1 !important;
    }
    .section-heading {
        font-family: 'Manrope', sans-serif;
        font-size: 1.3rem;
        font-weight: 800;
        color: #172b4d;
        letter-spacing: -.025em;
        margin: 8px 0 4px 0;
    }
    .section-subtitle {
        color: #6a7a90;
        margin-bottom: 18px;
        font-size: .94rem;
    }
    .surface {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 22px 24px;
        box-shadow: 0 5px 18px rgba(25, 52, 88, .035);
        margin-bottom: 16px;
    }
    .mini-label {
        color: #73829a;
        text-transform: uppercase;
        letter-spacing: .1em;
        font-size: .7rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .result-number {
        font-family: 'Manrope', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1.1;
        color: #173c75;
        margin: 6px 0 8px 0;
        overflow-wrap: anywhere;
    }
    .result-note {
        color: #53657d;
        line-height: 1.6;
        font-size: .94rem;
    }
    .status-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 6px 11px;
        background: #e8f1ff;
        color: #2458aa;
        font-size: .77rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .info-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 17px 18px;
        height: 100%;
    }
    .info-card h4 {
        margin: 0 0 7px 0;
        font-size: 1rem;
    }
    .info-card p {
        font-size: .9rem;
        line-height: 1.55;
        margin: 0;
    }
    div[data-testid="stForm"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 22px 24px 12px 24px;
        box-shadow: 0 5px 18px rgba(25, 52, 88, .035);
    }
    div[data-testid="stFormSubmitButton"] button,
    div.stButton > button {
        border-radius: 11px;
        min-height: 44px;
        font-weight: 700;
    }
    div[data-testid="stFormSubmitButton"] button {
        background: var(--blue);
        border: 1px solid var(--blue);
        color: white;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        background: var(--blue-dark);
        border-color: var(--blue-dark);
        color: white;
    }
    div[data-testid="stTabs"] button {
        font-weight: 700;
    }    
    div[data-testid="stRadio"] div[role="radiogroup"] {
        flex-wrap: nowrap !important;
        gap: 0.6rem !important;
    }
    div[data-testid="stRadio"] label {
        white-space: nowrap !important;
    }
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Dashboard specific styling */
    .dashboard-panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 18px;
        height: 100%;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .dashboard-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: 12px;
        border-bottom: 1px solid var(--line);
        padding-bottom: 8px;
    }
    div[data-testid="column"] {
        padding: 0.3rem;
    }

    /* Keep widget labels and controls readable in both themes */
    [data-testid="stAppViewContainer"], .stApp {
        background: #f5f8fc !important;
        color: #172b4d !important;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        background-color: #ffffff !important;
        border-color: #cbd5e1 !important;
        color: #172b4d !important;
    }
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] span,
    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea {
        color: #172b4d !important;
        -webkit-text-fill-color: #172b4d !important;
        opacity: 1 !important;
    }
    div[data-baseweb="select"] svg {
        fill: #53657d !important;
    }
    [data-baseweb="popover"] [role="option"] {
        background-color: #ffffff !important;
        color: #172b4d !important;
    }
    [data-baseweb="popover"] [role="option"]:hover {
        background-color: #e8f1ff !important;
    }
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within {
        border-color: #2463eb !important;
        box-shadow: 0 0 0 1px #2463eb !important;
    }
    ::selection {
        background: #c8dcff;
        color: #172b4d;
    }

    .stApp .brand-name {
        color: #142b50 !important;
        -webkit-text-fill-color: #142b50 !important;
        opacity: 1 !important;
    }
    .stApp .brand-subtitle {
        color: #425875 !important;
        -webkit-text-fill-color: #425875 !important;
        opacity: 1 !important;
    }
    .stApp .hero .hero-kicker {
        color: #b8e7f1 !important;
        -webkit-text-fill-color: #b8e7f1 !important;
    }
    .stApp .hero h1 {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    .stApp .hero p {
        color: #eaf4ff !important;
        -webkit-text-fill-color: #eaf4ff !important;
    }

    /* Highlight the Dark mode control */
    .stApp div.st-key-dark_mode {
        background: #e8f0ff !important;
        border: 1px solid #b9cff8 !important;
        border-radius: 14px !important;
        padding: 9px 14px !important;
        box-shadow: 0 5px 16px rgba(36, 99, 235, 0.15) !important;
        min-width: 150px !important;
        transition: background 0.2s ease, box-shadow 0.2s ease !important;
    }
    .stApp div.st-key-dark_mode:hover {
        background: #dce9ff !important;
        box-shadow: 0 7px 18px rgba(36, 99, 235, 0.22) !important;
    }
    .stApp div.st-key-dark_mode label,
    .stApp div.st-key-dark_mode label p {
        color: #173c75 !important;
        -webkit-text-fill-color: #173c75 !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }
    .stApp div.st-key-dark_mode [role="switch"] {
        transform: scale(1.12) !important;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .hero {
            padding: 23px 21px;
            border-radius: 18px;
        }
        .surface {
            padding: 18px;
        }
        .brand-lockup {
            padding-bottom: 8px;
        }
        div[data-testid="stForm"] {
            padding: 17px 15px 10px 15px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Load saved artifacts and prepared dataset
# ---------------------------------------------------------------------

def explain_one_prediction(model_pipeline, prediction_input, class_index):
    preprocessor = model_pipeline.named_steps["preprocessor"]
    classifier = model_pipeline.steps[-1][1]
    
    transformed = preprocessor.transform(prediction_input)
    if sparse.issparse(transformed):
        transformed = transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()
    explainer = shap.TreeExplainer(classifier)
    values = explainer.shap_values(
        transformed,
        check_additivity=False,
    )

    if isinstance(values, list):
        contributions = np.asarray(values[class_index])[0]
    else:
        values = np.asarray(values)
        if values.ndim == 3:
            contributions = values[0, :, class_index]
        elif values.ndim == 2:
            contributions = values[0]
        else:
            raise ValueError(f"Unexpected SHAP output shape: {values.shape}")

    return (
        pd.DataFrame({
            "feature": feature_names,
            "contribution": contributions,
        })
        .assign(abs_contribution=lambda frame: frame["contribution"].abs())
        .nlargest(8, "abs_contribution")
    )

# ---------------------------------------------------------------------
# Top navigation
# ---------------------------------------------------------------------
if "active_page" not in st.session_state:
    st.session_state.active_page = "Prediction"

nav_left, nav_right = st.columns(
    [1, 1.7],
    vertical_alignment="center",
    gap="small",
)

with nav_left:
    st.markdown(
        "<div class='brand-lockup'>"
        "<div class='brand-icon'>🚦</div>"
        "<div><div class='brand-name'>TrafficSense</div>"
        "<div class='brand-subtitle'>Smart Traffic Risk Prediction</div></div></div>",
        unsafe_allow_html=True,
    )

with nav_right:
    page_col, theme_col = st.columns(
        [4.5, 1],
        vertical_alignment="center",
        gap="small",
    )

    with page_col:
        selected_page = st.radio(
            "Main navigation",
            ["Prediction", "Dashboard", "Analytics", "About the project"],
            horizontal=True,
            label_visibility="collapsed",
            key="top_navigation",
        )

    with theme_col:
        dark_mode = st.toggle("Dark mode", key="dark_mode")

    st.session_state.active_page = selected_page

# Apply selected theme after navigation controls render so the toggle updates immediately.
if dark_mode:
    st.markdown("""
    <style>
    :root {
        --ink: #e5edf7;
        --muted: #bccadd;
        --line: #2b3b54;
        --panel: #131f33;
        --canvas: #0b1220;
    }
    .stApp, [data-testid="stAppViewContainer"] {
        background: var(--canvas) !important;
        color: var(--ink) !important;
    }
    [data-testid="stHeader"] {
        background: rgba(11,18,32,.96) !important;
    }
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p, [data-testid="stForm"] label, [data-testid="stForm"] label p, [data-testid="stRadio"] label, [data-testid="stRadio"] label p, [data-testid="stToggle"] label, [data-testid="stToggle"] label p {
        color: #d7e2f0 !important;
        -webkit-text-fill-color: #d7e2f0 !important;
        opacity: 1 !important;
    }
    .stApp div.st-key-dark_mode {
        background: #1b2d49 !important;
        border: 1px solid #5275a8 !important;
        box-shadow: 0 5px 16px rgba(0,0,0,.28) !important;
    }
    .stApp div.st-key-dark_mode:hover {
        background: #263d60 !important;
        border-color: #82a9eb !important;
    }
    .stApp div.st-key-dark_mode label, .stApp div.st-key-dark_mode label p {
        color: #f3f7ff !important;
        -webkit-text-fill-color: #f3f7ff !important;
    }
    .stApp .brand-name { color: #f3f7ff !important; -webkit-text-fill-color: #f3f7ff !important; }
    .stApp .brand-subtitle { color: #c0cee0 !important; -webkit-text-fill-color: #c0cee0 !important; }
    .stApp .section-heading, .stApp h2, .stApp h3, .stApp h4 { color: #edf4ff !important; -webkit-text-fill-color: #edf4ff !important; }
    .stApp .section-subtitle, .stApp .result-note, .stApp .surface p, .stApp .info-card p, .stApp [data-testid="stMarkdownContainer"] p { color: #bccadd !important; -webkit-text-fill-color: #bccadd !important; }
    .stApp .mini-label { color: #a9b8cc !important; -webkit-text-fill-color: #a9b8cc !important; }
    .stApp .result-number { color: #a9c7ff !important; -webkit-text-fill-color: #a9c7ff !important; }
    
    [data-testid="stForm"], .stApp .surface, .stApp .info-card, .dashboard-panel {
        background: var(--panel) !important;
        border-color: var(--line) !important;
        color: var(--ink) !important;
    }
    .dashboard-header { color: var(--ink); border-bottom: 1px solid var(--line); }

    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, div[data-baseweb="textarea"] > div {
        background: #0b1220 !important;
        border-color: #42536d !important;
        color: #f3f7ff !important;
    }
    div[data-baseweb="select"] input, div[data-baseweb="select"] span, div[data-baseweb="input"] input, div[data-baseweb="textarea"] textarea {
        color: #f3f7ff !important;
        -webkit-text-fill-color: #f3f7ff !important;
        opacity: 1 !important;
    }
    div[data-baseweb="select"] svg { fill: #c0cee0 !important; }
    [data-baseweb="popover"] [role="option"] { background: #131f33 !important; color: #f3f7ff !important; }
    [data-baseweb="popover"] [role="option"]:hover { background: #243752 !important; }
    
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p, [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
        color: #f3f7ff !important;
        -webkit-text-fill-color: #f3f7ff !important;
        opacity: 1 !important;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    :root {
        --ink: #172b4d;
        --muted: #66758b;
        --line: #e4eaf2;
        --panel: #ffffff;
        --canvas: #f5f8fc;
    }
    .stApp, [data-testid="stAppViewContainer"] {
        background: var(--canvas) !important;
        color: var(--ink) !important;
    }
    [data-testid="stHeader"] {
        background: rgba(245,248,252,.96) !important;
    }
    
    /* Fix for invisible KPI metrics in light mode */
    [data-testid="stMetricValue"] div, [data-testid="stMetricValue"] {
        color: #173c75 !important;
    }
    [data-testid="stMetricLabel"] p, [data-testid="stMetricLabel"] {
        color: #66758b !important;
    }
    
    /* Fix for invisible Warning/Error text in light mode */
    div[data-testid="stAlert"] div[role="alert"], div[data-testid="stAlert"] div {
        color: #172b4d !important;
    }

    .stApp div.st-key-dark_mode {
        background: #e8f0ff !important;
        border: 1px solid #b9cff8 !important;
        box-shadow: 0 5px 16px rgba(36,99,235,.15) !important;
    }
    .stApp div.st-key-dark_mode:hover {
        background: #dce9ff !important;
        border-color: #91b4f2 !important;
        box-shadow: 0 7px 18px rgba(36,99,235,.22) !important;
    }
    .stApp div.st-key-dark_mode label, .stApp div.st-key-dark_mode label p {
        color: #173c75 !important;
        -webkit-text-fill-color: #173c75 !important;
    }
    .stApp .brand-name { color: #142b50 !important; -webkit-text-fill-color: #142b50 !important; }
    .stApp .brand-subtitle { color: #425875 !important; -webkit-text-fill-color: #425875 !important; }
    .stApp .hero .hero-kicker { color: #b8e7f1 !important; -webkit-text-fill-color: #b8e7f1 !important; }
    .stApp .hero h1 { color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; }
    .stApp .hero p { color: #eaf4ff !important; -webkit-text-fill-color: #eaf4ff !important; }
    
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p, [data-testid="stForm"] label, [data-testid="stForm"] label p, [data-testid="stRadio"] label, [data-testid="stRadio"] label p {
        color: #263b59 !important;
        -webkit-text-fill-color: #263b59 !important;
        opacity: 1 !important;
    }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, div[data-baseweb="textarea"] > div {
        background: #ffffff !important;
        border-color: #cbd5e1 !important;
        color: #172b4d !important;
    }
    div[data-baseweb="select"] input, div[data-baseweb="select"] span, div[data-baseweb="input"] input, div[data-baseweb="textarea"] textarea {
        color: #172b4d !important;
        -webkit-text-fill-color: #172b4d !important;
        opacity: 1 !important;
    }
    [data-baseweb="popover"] [role="option"] { background: #ffffff !important; color: #172b4d !important; }
    [data-baseweb="popover"] [role="option"]:hover { background: #e8f1ff !important; }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Main heading
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Intelligent road safety · Decision support</div>
      <h1>Understand accident severity.<br>Make better-informed decisions.</h1>
      <p>Explore how recorded accident conditions map to a predicted severity class using your trained machine-learning model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Load saved artifacts and prepared dataset
# ---------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder

@st.cache_data
def load_prepared_dataset(path_string):
    return pd.read_csv(path_string)

def available_values(dataframe, column):
    """Return sorted non-null dataset values, preserving their stored types."""
    if column not in dataframe.columns:
        raise KeyError(
            f"Required column '{column}' is missing from the prepared dataset."
        )
    values = dataframe[column].dropna().unique().tolist()
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=lambda value: str(value))

def median_or_default(series, default=0):
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if not values.empty else float(default)

def display_value(value):
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)

WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
CATEGORY_VALUE_LABELS = {
    "Accident_Location_A": {
        1: "Urban", 2: "Rural", 3: "Unallocated",
    },
    "Accident_Location_A_Chainage_km_RoadSide": {
        1: "Left-hand side (LHS)", 2: "Right-hand side (RHS)",
    },
    # The source dataset names this column Causes_D, but its observed codes
    # match the published Nature of Accident categories.
    "Causes_D": {
        1: "Overturning",
        2: "Head-on collision",
        3: "Rear-end collision",
        4: "Brush / side-wipe collision",
        5: "Right-turn collision",
        6: "Skidding",
        8: "Overtaking vehicle",
    },
    "Road_Feature_E": {
        1: "Single-lane road",
        2: "Two-lane road",
        3: "Three or more lanes, no central divider",
        4: "Four or more lanes with a central divider",
    },
    "Road_Condition_F": {
        1: "Straight road",
        2: "Slight curve",
        3: "Sharp curve",
        4: "Flat road",
        5: "Gentle incline",
        6: "Steep incline",
        7: "Hump",
        8: "Dip",
    },
    "Weather_Conditions_H": {
        1: "Fine",
        2: "Mist / fog",
        3: "Cloudy",
        4: "Light rain",
        5: "Heavy rain",
        6: "Hail / sleet",
        7: "Snow",
        8: "Strong wind",
        9: "Dust storm",
        10: "Very hot",
        11: "Very cold",
        12: "Other extraordinary weather",
    },
    "Vehicle_Type_Involved_J_V1": {
        1: "Car / jeep / van", 2: "SUV", 3: "Bus", 4: "Mini bus",
        5: "Truck", 6: "Two-wheeler", 7: "Three-wheeler", 8: "Cycle",
        9: "Pedestrian", 10: "Tractor", 11: "Unknown", 12: "Animal",
        13: "Object", 14: "Light commercial vehicle (LCV)",
        15: "Multi-axle vehicle (MAV)",
    },
    "Vehicle_Type_Involved_J_V2": {
        1: "Car / jeep / van", 2: "SUV", 3: "Bus", 4: "Mini bus",
        5: "Truck", 6: "Two-wheeler", 7: "Three-wheeler", 8: "Cycle",
        9: "Pedestrian", 10: "Tractor", 11: "Unknown", 12: "Animal",
        13: "Object", 14: "Light commercial vehicle (LCV)",
        15: "Multi-axle vehicle (MAV)",
    },
}

def format_category(column, value):
    """Display a descriptive category label but preserve the model's code."""
    try:
        number = float(value)
        code = int(number) if number.is_integer() else None
    except (TypeError, ValueError):
        code = None

    if code is not None and column == "Day_of_Week" and 1 <= code <= 7:
        # This dataset uses 1=Monday through 7=Sunday.
        return WEEKDAY_NAMES[code - 1]
    if code is not None and column == "Accident_DayOfWeek" and 0 <= code <= 6:
        return WEEKDAY_NAMES[code]
    if code is not None and column == "Accident_Month" and 1 <= code <= 12:
        return MONTH_NAMES[code - 1]
    if code is not None and column == "Is_Weekend" and code in (0, 1):
        return "Weekend" if code == 1 else "Weekday"
    if code is not None and column == "Accident_Hour" and 0 <= code <= 23:
        return f"{code:02d}:00"

    labels = CATEGORY_VALUE_LABELS.get(column, {})
    if code in labels:
        return labels[code]
    return display_value(value)

@st.cache_data
def load_location_segments(path_string):
    segments = pd.read_csv(path_string)
    required = ["Accident_Location_A", "chainage_start_km", "chainage_end_km", "latitude", "longitude", "segment_name"]
    for column in ["Accident_Location_A", "chainage_start_km", "chainage_end_km", "latitude", "longitude"]:
        segments[column] = pd.to_numeric(segments[column], errors="coerce")
    return segments.dropna(subset=required).copy()

def find_location_segment(segments, location_code, chainage_km):
    location_code, chainage_km = pd.to_numeric(location_code, errors="coerce"), pd.to_numeric(chainage_km, errors="coerce")
    if pd.isna(location_code) or pd.isna(chainage_km):
        return None
    route = segments[segments["Accident_Location_A"] == float(location_code)]
    matches = route[(route["chainage_start_km"] <= float(chainage_km)) & (float(chainage_km) < route["chainage_end_km"])]
    if matches.empty and not route.empty:
        final_end = route["chainage_end_km"].max()
        if np.isclose(float(chainage_km), final_end):
            matches = route[route["chainage_end_km"] == final_end]
    return matches.iloc[0] if not matches.empty else None

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_current_weather(latitude, longitude):
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": float(latitude), "longitude": float(longitude),
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
            "timezone": "auto",
        },
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["current"], payload.get("current_units", {})

def describe_wmo_weather(code):
    descriptions = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle", 61: "Slight rain",
        63: "Moderate rain", 65: "Heavy rain", 71: "Slight snowfall", 95: "Thunderstorm"
    }
    return descriptions.get(int(code), f"Weather code {code}")

def map_wmo_to_dataset(wmo_code):
    """Maps Open-Meteo WMO codes to the dataset's Weather_Conditions_H codes."""
    wmo = int(wmo_code)
    if wmo in [0, 1]: return "1"         # Fine / Clear
    elif wmo in [45, 48]: return "2"     # Mist / Fog
    elif wmo in [2, 3]: return "3"       # Cloudy
    elif wmo in [51, 53, 55, 61]: return "4" # Light rain
    elif wmo in [63, 65, 80, 81, 82]: return "5" # Heavy rain
    elif wmo in [66, 67, 71, 73, 75, 77, 85, 86]: return "7" # Snow
    elif wmo in [95, 96, 99]: return "6" # Hail / Thunderstorm
    return "1" # Default to Fine

# ---------------------------------------------------------------------
# Load model, encoder, and prepared dataset
# ---------------------------------------------------------------------
try:
    model, label_encoder = load_artifacts()
except Exception as error:
    st.error(
        "Could not load the saved model or label encoder. "
        "Please check that both files exist in the models folder."
    )
    st.exception(error)
    st.stop()

try:
    df_options = load_prepared_dataset(str(DATA_PATH))
except Exception as error:
    st.error(f"Could not read the prepared dataset: {error}")
    st.stop()

missing_columns = [
    column for column in feature_columns
    if column not in df_options.columns
]

if missing_columns:
    st.error(
        "The prepared dataset is missing columns required by the current feature schema: "
        + ", ".join(f"`{column}`" for column in missing_columns)
    )
    st.stop()

try:
    option_map = {
        column: available_values(df_options, column)
        for column in categorical_features
    }
except (KeyError, TypeError) as error:
    st.error(str(error))
    st.stop()

empty_option_columns = [
    column for column, values in option_map.items()
    if not values
]
if empty_option_columns:
    st.error(
        "These required categorical columns contain no non-empty values: "
        + ", ".join(f"`{column}`" for column in empty_option_columns)
    )
    st.stop()


# ---------------------------------------------------------------------
# Prediction page
# ---------------------------------------------------------------------
if selected_page == "Prediction":
    st.markdown('<div class="section-heading">Accident conditions</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Enter the conditions to classify the recorded accident severity. Dropdown choices come from your prepared dataset.</div>', unsafe_allow_html=True)

    with st.form("prediction_form", clear_on_submit=False):
        st.markdown("#### 01 · Time and location")
        time_col1, time_col2, time_col3 = st.columns(3)

        with time_col1:
            day_value = st.selectbox(
                "Day of week", 
                option_map["Day_of_Week"], 
                format_func=lambda value: format_category("Day_of_Week", value)
            )
            month_value = st.selectbox(
                "Accident month", 
                option_map["Accident_Month"], 
                format_func=lambda value: format_category("Accident_Month", value)
            )

        with time_col2:
            location_value = st.selectbox(
                "Accident location", 
                option_map["Accident_Location_A"], 
                format_func=lambda value: format_category("Accident_Location_A", value)
            )
            hour_value = st.selectbox(
                "Accident time", 
                option_map["Accident_Hour"], 
                format_func=lambda value: format_category("Accident_Hour", value)
            )

        with time_col3:
            chainage_value = st.number_input(
                "Location chainage (km)",
                min_value=0.0,
                value=max(0.0, median_or_default(df_options["Accident_Location_A_Chainage_km"])),
                step=0.1, format="%.3f",
            )
            year_median = median_or_default(df_options["Accident_Year"], 2024)
            year_value = st.number_input(
                "Accident year",
                min_value=1900, max_value=2100,
                value=int(np.clip(round(year_median), 1900, 2100)),
                step=1,
            )

        roadside_value = st.selectbox(
            "Roadside category", 
            option_map["Accident_Location_A_Chainage_km_RoadSide"], 
            format_func=lambda value: format_category("Accident_Location_A_Chainage_km_RoadSide", value)
        )

        st.markdown("---")
        st.markdown("#### 02 · Road, weather and cause")
        road_col1, road_col2 = st.columns(2)

        with road_col1:
            cause_value = st.selectbox(
                "Accident type", 
                option_map["Causes_D"], 
                format_func=lambda value: format_category("Causes_D", value)
            )
            road_feature_value = st.selectbox(
                "Road feature", 
                option_map["Road_Feature_E"], 
                format_func=lambda value: format_category("Road_Feature_E", value)
            )
            road_condition_value = st.selectbox(
                "Road condition", 
                option_map["Road_Condition_F"], 
                format_func=lambda value: format_category("Road_Condition_F", value)
            )

        with road_col2:
            weather_value = st.selectbox(
                "Weather conditions", 
                option_map["Weather_Conditions_H"], 
                format_func=lambda value: format_category("Weather_Conditions_H", value)
            )
            weekend_value = st.selectbox(
                "Weekend status", 
                option_map["Is_Weekend"], 
                format_func=lambda value: format_category("Is_Weekend", value)
            )
            day_of_week_value = st.selectbox(
                "Accident day of week",
                option_map["Accident_DayOfWeek"],
                format_func=lambda value: format_category("Accident_DayOfWeek", value),
                help="Selected independently to preserve the feature values expected by the trained model."
            )

        st.markdown("---")
        st.markdown("#### 03 · Vehicles involved")
        vehicle_col1, vehicle_col2 = st.columns(2)

        with vehicle_col1:
            vehicle1_value = st.selectbox(
                "Vehicle type 1", 
                option_map["Vehicle_Type_Involved_J_V1"], 
                format_func=lambda value: format_category("Vehicle_Type_Involved_J_V1", value)
            )

        with vehicle_col2:
            vehicle2_value = st.selectbox(
                "Vehicle type 2",
                option_map["Vehicle_Type_Involved_J_V2"],
                format_func=lambda value: format_category("Vehicle_Type_Involved_J_V2", value),
                help="Choose a value represented in the prepared dataset.",
            )

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("Predict severity class", type="primary", use_container_width=True)
        
    if submitted:
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
            "Accident_DayOfWeek": day_of_week_value,
            "Accident_Hour": hour_value,
            "Is_Weekend": weekend_value,
            "Accident_Location_A_Chainage_km": chainage_value,
            "Accident_Year": year_value,
        }], columns=feature_columns)

        try:
            prediction_input = input_data.copy()
            for column in categorical_features:
                prediction_input[column] = prediction_input[column].astype("string").fillna("Missing").astype(str)

            raw_prediction = np.asarray(model.predict(prediction_input)).reshape(-1)
            if raw_prediction.size == 0:
                raise ValueError("The model returned no prediction.")

            encoded_prediction = int(raw_prediction[0])
            explanation = explain_one_prediction(model, prediction_input, encoded_prediction)

            severity_prediction = label_encoder.inverse_transform(
                np.asarray([encoded_prediction], dtype=int)
            )[0]

            st.session_state["last_prediction"] = {
                "severity": severity_prediction,
                "input": input_data.copy(),
                "explanation": explanation,
            }

        except Exception as error:
            st.error("Prediction could not be completed. Confirm that the selected values and feature schema match the model used during training.")
            st.exception(error)

    last_prediction = st.session_state.get("last_prediction")

    if last_prediction is not None:
        st.markdown('<div class="section-heading">Prediction result</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Result from the saved model for the submitted conditions.</div>', unsafe_allow_html=True)

        result_col, details_col = st.columns([1, 1.35], gap="large")
        with result_col:
            st.markdown(f"""
                <div class="surface">
                  <div class="mini-label">Predicted severity class</div>
                  <div class="status-pill">Model prediction</div>
                  <div class="result-number">{display_value(last_prediction["severity"])}</div>
                  <div class="result-note">This is the severity class returned for the conditions you entered.</div>
                </div>
                """, unsafe_allow_html=True)

        with details_col:
            st.markdown("""
                <div class="surface">
                  <div class="mini-label">How to interpret this result</div>
                  <div class="result-note">
                    The output represents a predicted <b>accident severity class</b>, not the probability
                    that an accident will occur. Use your dataset documentation or class codebook to
                    interpret the exact meaning of the class label.
                    <br><br>
                    This tool is intended for educational and decision-support use. It does not replace
                    official traffic-safety assessments or emergency services.
                  </div>
                </div>
                """, unsafe_allow_html=True)
            
        explanation = last_prediction.get("explanation")
        if explanation is not None and not explanation.empty:
            st.markdown("#### Factors influencing this prediction")
            st.caption("Positive SHAP values support the predicted class; negative values push away from it. These are model contributions, not proof of causation.")
            
            explanation_chart = explanation.sort_values("contribution")
            fig = px.bar(
                explanation_chart, x="contribution", y="feature", orientation="h",
                color="contribution", color_continuous_scale=["#d95f59", "#d9e2f0", "#2463eb"],
                labels={"contribution": "SHAP contribution to model score", "feature": "Encoded model feature"}
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Review submitted conditions"):
            st.dataframe(last_prediction["input"], use_container_width=True, hide_index=True)

    else:
        st.markdown("""
            <div class="surface">
              <div class="mini-label">Ready when you are</div>
              <div style="font-size:1.15rem;font-weight:800;margin-bottom:7px;">Your result will appear here</div>
              <div class="result-note">
                Complete the accident conditions above and select <b>Predict severity class</b>.
                The model's output will be shown here, along with the submitted input values.
              </div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Dashboard page
# ---------------------------------------------------------------------
elif selected_page == "Dashboard":
    
    st.markdown('<div class="section-heading">Traffic Management Dashboard</div>', unsafe_allow_html=True)
    
    if not LOCATION_PATH.exists():
        st.error(f"Location mapping file missing: {LOCATION_PATH}. Run the generate_coordinates.py script first.")
        st.stop()

    # Load locations and map accidents
    location_segments = load_location_segments(str(LOCATION_PATH))
    map_rows = []

    for row in df_options[["Accident_Location_A", "Accident_Location_A_Chainage_km", "Accident_Severity_C"]].itertuples(index=False):
        segment = find_location_segment(location_segments, row.Accident_Location_A, row.Accident_Location_A_Chainage_km)
        if segment is not None:
            map_rows.append({
                "segment_name": segment["segment_name"], 
                "latitude": segment["latitude"], 
                "longitude": segment["longitude"],
                "severity": row.Accident_Severity_C
            })

    mapped_accidents = pd.DataFrame(map_rows)

    # --- TOP ROW: KPIs ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Mapped Accidents", f"{len(mapped_accidents)}", "Historical")
    with kpi2:
        avg_sev = pd.to_numeric(mapped_accidents["severity"], errors="coerce").mean() if not mapped_accidents.empty else 0
        st.metric("Avg Severity Risk", f"{avg_sev:.2f}", "Current Data")
    with kpi3:
        st.metric("Active Segments", f"{mapped_accidents['segment_name'].nunique() if not mapped_accidents.empty else 0}", "Monitored")
    with kpi4:
        st.metric("Live API Status", "Online", "Open-Meteo")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- MIDDLE ROW: Map & Weather Panels ---
    map_col, side_col = st.columns([2.5, 1])
    
    with map_col:
        st.markdown('<div class="dashboard-panel"><div class="dashboard-header">Live Predictive Map</div>', unsafe_allow_html=True)
        
        if not mapped_accidents.empty:
            mapped_accidents["severity"] = pd.to_numeric(mapped_accidents["severity"], errors="coerce")
            map_summary = mapped_accidents.groupby(["segment_name", "latitude", "longitude"], as_index=False).agg(
                accident_count=("severity", "size"),
                avg_severity_risk=("severity", "mean")
            )

            # Map Fix: Dynamic style based on theme and explicitly setting height
            map_theme = "carto-darkmatter" if dark_mode else "carto-positron"
            
            fig_map = px.scatter_map(
                map_summary,
                lat="latitude",
                lon="longitude",
                size="accident_count",
                color="avg_severity_risk",
                hover_name="segment_name",
                color_continuous_scale=px.colors.sequential.YlOrRd,
                size_max=45,
                zoom=9,
                center={"lat": map_summary["latitude"].mean(), "lon": map_summary["longitude"].mean()},
                map_style="open-street-map"
            )
            
            fig_map.update_layout(
                height=450, # <-- This prevents the blank blue box
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_map, use_container_width=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
    with side_col:
        st.markdown('<div class="dashboard-panel"><div class="dashboard-header">Live Weather Prediction</div>', unsafe_allow_html=True)
        
        loc_options = available_values(df_options, "Accident_Location_A")
        selected_weather_loc = st.selectbox("Select Segment", loc_options, format_func=lambda value: format_category("Accident_Location_A", value), label_visibility="collapsed")
        
        selected_route = df_options[pd.to_numeric(df_options["Accident_Location_A"], errors="coerce") == float(selected_weather_loc)]
        
        if not selected_route.empty:
            route_chainage = pd.to_numeric(selected_route["Accident_Location_A_Chainage_km"], errors="coerce").dropna()
            selected_chainage = float(route_chainage.median())
            weather_segment = find_location_segment(location_segments, selected_weather_loc, selected_chainage)

            if weather_segment is not None and st.button("Predict Live Risk", type="primary", use_container_width=True):
                try:
                    # 1. Fetch Live Weather
                    current, units = fetch_current_weather(weather_segment["latitude"], weather_segment["longitude"])
                    weather_desc = describe_wmo_weather(current['weather_code'])
                    
                    st.markdown(f"""
                    <h2 style="margin:0;">{current['temperature_2m']} {units.get('temperature_2m', '°C')}</h2>
                    <p style="margin:0; font-size:1.1rem; padding-bottom: 5px;">{weather_desc}</p>
                    <hr style="margin: 10px 0;">
                    <div style="font-size:0.95rem; margin-bottom:5px;">💧 Humidity: <b>{current['relative_humidity_2m']}{units.get('relative_humidity_2m', '%')}</b></div>
                    <div style="font-size:0.95rem; margin-bottom:5px;">💨 Wind: <b>{current['wind_speed_10m']} {units.get('wind_speed_10m', 'km/h')}</b></div>
                    """, unsafe_allow_html=True)
                    
                    # 2. Build Live Data for Model
                    dataset_weather_code = map_wmo_to_dataset(current['weather_code'])
                    now = pd.Timestamp.now()
                    
                    # Helper to get the most common historical road condition for this location
                    def safe_mode(col, default="1"):
                        modes = selected_route[col].mode()
                        return str(modes.iloc[0]) if not modes.empty else str(default)

                    live_input = pd.DataFrame([{
                        "Day_of_Week": str(now.dayofweek + 1),
                        "Accident_Location_A": str(selected_weather_loc),
                        "Accident_Location_A_Chainage_km_RoadSide": safe_mode("Accident_Location_A_Chainage_km_RoadSide"),
                        "Causes_D": safe_mode("Causes_D"),
                        "Road_Feature_E": safe_mode("Road_Feature_E"),
                        "Road_Condition_F": safe_mode("Road_Condition_F"),
                        "Weather_Conditions_H": dataset_weather_code, # <-- Injected Live Weather
                        "Vehicle_Type_Involved_J_V1": safe_mode("Vehicle_Type_Involved_J_V1"),
                        "Vehicle_Type_Involved_J_V2": safe_mode("Vehicle_Type_Involved_J_V2"),
                        "Accident_Month": str(now.month),
                        "Accident_DayOfWeek": str(now.dayofweek),
                        "Accident_Hour": str(now.hour),
                        "Is_Weekend": "1" if now.dayofweek >= 5 else "0",
                        "Accident_Location_A_Chainage_km": selected_chainage,
                        "Accident_Year": now.year,
                    }], columns=feature_columns)

                    for column in categorical_features:
                        live_input[column] = live_input[column].astype("string").fillna("Missing").astype(str)

                    # 3. Predict & Display
                    live_pred_raw = np.asarray(model.predict(live_input)).reshape(-1)
                    live_severity = label_encoder.inverse_transform([int(live_pred_raw[0])])[0]

                    st.success(f"**🚨 Live Predicted Severity:** {display_value(live_severity)}")
                    
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
        else:
            st.warning("No data found for this segment.")
            
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- BOTTOM ROW: Charts ---
    bot1, bot2 = st.columns([1, 1])
    
    with bot1:
        st.markdown('<div class="dashboard-panel"><div class="dashboard-header">Historical Severity Distribution</div>', unsafe_allow_html=True)
        if not mapped_accidents.empty:
            sev_counts = mapped_accidents['severity'].value_counts().reset_index()
            sev_counts.columns = ['Severity', 'Count']
            fig_bar = px.bar(sev_counts, x='Severity', y='Count', color='Severity', color_continuous_scale="RdYlGn_r")
            fig_bar.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with bot2:
        st.markdown('<div class="dashboard-panel"><div class="dashboard-header">Accident Volume by Month</div>', unsafe_allow_html=True)
        monthly_counts = df_options.groupby("Accident_Month").size().reset_index(name="Count")
        fig_line = px.line(monthly_counts, x="Accident_Month", y="Count", markers=True)
        fig_line.update_traces(line=dict(color='#2463eb', width=3), marker=dict(size=8, color='#ff4b4b'))
        fig_line.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_line, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Analytics page
# ---------------------------------------------------------------------
elif selected_page == "Analytics":
    st.markdown('<div class="section-heading">Model performance and accident patterns</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Held-out model results and summaries of recorded accidents.</div>', unsafe_allow_html=True)
    
    metrics_path = BASE_DIR / "models" / "model_metrics.json"
    importance_path = BASE_DIR / "models" / "feature_importance.csv"
    
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as file:
            metrics = json.load(file)
        
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
        metric_col2.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
        
        class_report = metrics["classification_report"]
        class_rows = []
        
        for class_name in metrics["classes"]:
            result = class_report.get(class_name, {})
            class_rows.append({
                "Severity class": class_name,
                "Precision": result.get("precision", 0),
                "Recall": result.get("recall", 0),
                "F1 score": result.get("f1-score", 0),
                "Test rows": result.get("support", 0),
            })
            
        st.markdown("#### Scores by severity class")
        st.dataframe(
            pd.DataFrame(class_rows).style.format({
                "Precision": "{:.3f}", "Recall": "{:.3f}", "F1 score": "{:.3f}", "Test rows": "{:.0f}",
            }),
            use_container_width=True, hide_index=True,
        )
        
        st.markdown("#### Confusion matrix")
        confusion = pd.DataFrame(
            metrics["confusion_matrix"], index=metrics["classes"], columns=metrics["classes"],
        )
        st.dataframe(confusion, use_container_width=True)
        
    else:
        st.info("Run the metrics cell in 03_model_training.ipynb first. It creates models/model_metrics.json.")
        
    if importance_path.exists():
        importance = pd.read_csv(importance_path).head(15).sort_values("mean_abs_shap")
        st.markdown("#### Global SHAP feature importance")
        st.caption("This chart summarizes which encoded input features most influenced the model across the sampled test records.")
        fig = px.bar(
            importance, x="mean_abs_shap", y="feature", orientation="h",
            labels={"mean_abs_shap": "Mean absolute SHAP value", "feature": "Encoded model feature"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the SHAP cell in 03_model_training.ipynb first. It creates models/feature_importance.csv.")
        
    st.markdown("#### Recorded accidents by severity class")
    severity_counts = df_options["Accident_Severity_C"].value_counts().sort_index().rename_axis("Severity class").reset_index(name="Recorded accidents")
    fig = px.bar(severity_counts, x="Severity class", y="Recorded accidents", labels={"Severity class": "Dataset severity code"})
    st.plotly_chart(fig, use_container_width=True)

    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        monthly_counts = df_options.groupby("Accident_Month").size().reset_index(name="Recorded accidents")
        fig = px.bar(monthly_counts, x="Accident_Month", y="Recorded accidents", labels={"Accident_Month": "Month"})
        st.markdown("#### Recorded accidents by month")
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        weather_counts = df_options["Weather_Conditions_H"].value_counts().sort_index().rename_axis("Weather code").reset_index(name="Recorded accidents")
        fig = px.bar(weather_counts, x="Weather code", y="Recorded accidents")
        st.markdown("#### Recorded accidents by weather code")
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------
# About page
# ---------------------------------------------------------------------
else:
    st.markdown('<div class="section-heading">About the project</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">A machine-learning interface for exploring recorded road-accident severity.</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="surface">
          <div class="mini-label">Project overview</div>
          <h3 style="font-family:Manrope,sans-serif;margin:0 0 10px 0;">Turning accident conditions into a predicted severity class</h3>
          <p style="line-height:1.75;margin:0;">
            The Smart Traffic Risk Prediction System takes structured accident information—such as
            time, location, road conditions, weather, reported cause, and vehicles involved—and passes
            those features to a previously trained machine-learning model. The saved label encoder is
            used to translate the model's encoded output back into a severity class.
          </p>
        </div>
        """, unsafe_allow_html=True)

    card1, card2, card3 = st.columns(3, gap="medium")
    with card1:
        st.markdown("""<div class="info-card"><h4>🧾 Input conditions</h4><p>Use prepared-dataset values to describe the recorded accident, including time, location, road, weather and vehicle details.</p></div>""", unsafe_allow_html=True)
    with card2:
        st.markdown("""<div class="info-card"><h4>🧠 Saved ML model</h4><p>The application loads your existing serialized model and label encoder. It does not retrain or overwrite them.</p></div>""", unsafe_allow_html=True)
    with card3:
        st.markdown("""<div class="info-card"><h4>🚦 Interpretable output</h4><p>The result is the predicted severity class. Its real-world meaning must be confirmed from the dataset's class documentation.</p></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    st.markdown("""
        <div class="surface">
          <div class="mini-label">Important limitations</div>
          <p style="line-height:1.75;margin:0;">
            This application predicts a class from the fields supplied to the model. It does not
            establish causation, provide a live traffic feed, or calculate the likelihood of a future
            accident unless those capabilities are explicitly built and validated separately.
            Dataset category codes should be interpreted using the original data dictionary.
            Do not use the output as the sole basis for operational, legal, or emergency decisions.
          </p>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.markdown("""
    <div style="border-top:1px solid #e3e9f1;margin-top:28px;padding-top:16px;color:#8190a4;font-size:.78rem;text-align:center;">
      Smart Traffic Risk Prediction System · Educational decision-support interface
    </div>
    """, unsafe_allow_html=True)