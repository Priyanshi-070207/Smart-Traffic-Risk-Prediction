import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
import plotly.express as px
import shap
from scipy import sparse

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
# Existing project paths — kept compatible with the supplied app.py
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "traffic_severity_model.pkl"
ENCODER_PATH = BASE_DIR / "models" / "severity_label_encoder.pkl"
DATA_PATH = BASE_DIR / "dataa" / "processed" / "accident_data_prepared.csv"

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
        background: white;
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
        background: #f8fbff;
        border: 1px solid #e4edf8;
        border-radius: 15px;
        padding: 17px 18px;
        height: 100%;
    }
    .info-card h4 {
        margin: 0 0 7px 0;
        color: #18365e;
        font-size: 1rem;
    }
    .info-card p {
        color: #65758b;
        font-size: .9rem;
        line-height: 1.55;
        margin: 0;
    }
    div[data-testid="stForm"] {
        background: #fff;
        border: 1px solid #e4eaf2;
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
    label, .stMarkdown p {
        color: #344761;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {
        color: #172b4d !important;
        -webkit-text-fill-color: #172b4d !important;
        opacity: 1 !important;
    }

    /* Keep widget labels and controls readable in both themes */
    [data-testid="stAppViewContainer"], .stApp {
        background: #f5f8fc !important;
        color: #172b4d !important;
    }
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    div[data-testid="stForm"] label,
    div[data-testid="stForm"] label p,
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label p {
        color: #263b59 !important;
        opacity: 1 !important;
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
            ["Prediction", "Analytics","About the project"],
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
    .stApp, [data-testid="stAppViewContainer"] {
        background:#0b1220 !important;
        color:#e5edf7 !important;
    }

    [data-testid="stHeader"] {
        background:rgba(11,18,32,.96) !important;
    }

    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stForm"] label,
    [data-testid="stForm"] label p,
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] label p,
    [data-testid="stToggle"] label,
    [data-testid="stToggle"] label p {
        color:#d7e2f0 !important;
        -webkit-text-fill-color:#d7e2f0 !important;
        opacity:1 !important;
    }

    /* More visible Dark mode toggle in dark theme */
    .stApp div.st-key-dark_mode {
        background:#1b2d49 !important;
        border:1px solid #5275a8 !important;
        border-radius:14px !important;
        padding:10px 14px !important;
        box-shadow:0 5px 16px rgba(0,0,0,.28) !important;
        min-width:150px !important;
    }

    .stApp div.st-key-dark_mode:hover {
        background:#263d60 !important;
        border-color:#82a9eb !important;
    }

    .stApp div.st-key-dark_mode label,
    .stApp div.st-key-dark_mode label p {
        color:#f3f7ff !important;
        -webkit-text-fill-color:#f3f7ff !important;
        font-weight:700 !important;
        opacity:1 !important;
    }

    /* TrafficSense brand in dark theme */
    .stApp .brand-name {
        color:#f3f7ff !important;
        -webkit-text-fill-color:#f3f7ff !important;
    }

    .stApp .brand-subtitle {
        color:#c0cee0 !important;
        -webkit-text-fill-color:#c0cee0 !important;
    }

    .stApp .section-heading,
    .stApp h2,
    .stApp h3,
    .stApp h4 {
        color:#edf4ff !important;
        -webkit-text-fill-color:#edf4ff !important;
    }

    .stApp .section-subtitle,
    .stApp .result-note,
    .stApp .surface p,
    .stApp .info-card p,
    .stApp [data-testid="stMarkdownContainer"] p {
        color:#bccadd !important;
        -webkit-text-fill-color:#bccadd !important;
    }

    .stApp .mini-label {
        color:#a9b8cc !important;
        -webkit-text-fill-color:#a9b8cc !important;
    }

    .stApp .result-number {
        color:#a9c7ff !important;
        -webkit-text-fill-color:#a9c7ff !important;
    }

    .stApp .hero .hero-kicker {
        color:#b8e7f1 !important;
        -webkit-text-fill-color:#b8e7f1 !important;
    }

    .stApp .hero h1 {
        color:#ffffff !important;
        -webkit-text-fill-color:#ffffff !important;
    }

    .stApp .hero p {
        color:#eaf4ff !important;
        -webkit-text-fill-color:#eaf4ff !important;
    }

    [data-testid="stForm"],
    .stApp .surface,
    .stApp .info-card {
        background:#131f33 !important;
        border-color:#2b3b54 !important;
        color:#e5edf7 !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        background:#0b1220 !important;
        border-color:#42536d !important;
        color:#f3f7ff !important;
    }

    div[data-baseweb="select"] input,
    div[data-baseweb="select"] span,
    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea {
        color:#f3f7ff !important;
        -webkit-text-fill-color:#f3f7ff !important;
        opacity:1 !important;
    }

    div[data-baseweb="select"] svg {
        fill:#c0cee0 !important;
    }

    [data-baseweb="popover"] [role="option"] {
        background:#131f33 !important;
        color:#f3f7ff !important;
    }

    [data-baseweb="popover"] [role="option"]:hover {
        background:#243752 !important;
    }
    
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {
        color: #f3f7ff !important;
        -webkit-text-fill-color: #f3f7ff !important;
        opacity: 1 !important;
    }
    </style>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <style>
    .stApp, [data-testid="stAppViewContainer"] {
        background:#f5f8fc !important;
        color:#172b4d !important;
    }

    [data-testid="stHeader"] {
        background:rgba(245,248,252,.96) !important;
    }

    /* More visible Dark mode toggle in light theme */
    .stApp div.st-key-dark_mode {
        background:#e8f0ff !important;
        border:1px solid #b9cff8 !important;
        border-radius:14px !important;
        padding:9px 14px !important;
        box-shadow:0 5px 16px rgba(36,99,235,.15) !important;
        min-width:150px !important;
    }

    .stApp div.st-key-dark_mode:hover {
        background:#dce9ff !important;
        border-color:#91b4f2 !important;
        box-shadow:0 7px 18px rgba(36,99,235,.22) !important;
    }

    .stApp div.st-key-dark_mode label,
    .stApp div.st-key-dark_mode label p {
        color:#173c75 !important;
        -webkit-text-fill-color:#173c75 !important;
        font-weight:700 !important;
        opacity:1 !important;
    }

    /* Clearer TrafficSense title and subtitle in light theme */
    .stApp .brand-name {
        color:#142b50 !important;
        -webkit-text-fill-color:#142b50 !important;
    }

    .stApp .brand-subtitle {
        color:#425875 !important;
        -webkit-text-fill-color:#425875 !important;
    }

    .stApp .hero .hero-kicker {
        color:#b8e7f1 !important;
        -webkit-text-fill-color:#b8e7f1 !important;
    }

    .stApp .hero h1 {
        color:#ffffff !important;
        -webkit-text-fill-color:#ffffff !important;
    }

    .stApp .hero p {
        color:#eaf4ff !important;
        -webkit-text-fill-color:#eaf4ff !important;
    }

    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stForm"] label,
    [data-testid="stForm"] label p,
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] label p {
        color:#263b59 !important;
        -webkit-text-fill-color:#263b59 !important;
        opacity:1 !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        background:#ffffff !important;
        border-color:#cbd5e1 !important;
        color:#172b4d !important;
    }

    div[data-baseweb="select"] input,
    div[data-baseweb="select"] span,
    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea {
        color:#172b4d !important;
        -webkit-text-fill-color:#172b4d !important;
        opacity:1 !important;
    }

    [data-baseweb="popover"] [role="option"] {
        background:#ffffff !important;
        color:#172b4d !important;
    }

    [data-baseweb="popover"] [role="option"]:hover {
        background:#e8f1ff !important;
    }
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


# Validate all dropdowns can be populated before the form is shown.
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
    st.markdown(
        '<div class="section-heading">Accident conditions</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">Enter the conditions to classify the recorded accident severity. '
        'Dropdown choices come from your prepared dataset.</div>',
        unsafe_allow_html=True,
    )

    with st.form("prediction_form", clear_on_submit=False):
        st.markdown("#### 01 · Time and location")

        time_col1, time_col2, time_col3 = st.columns(3)

        with time_col1:
            day_value = st.selectbox(
                "Day of week · dataset code",
                option_map["Day_of_Week"],
                format_func=display_value,
            )

            month_value = st.selectbox(
                "Accident month",
                option_map["Accident_Month"],
                format_func=display_value,
            )

        with time_col2:
            location_value = st.selectbox(
                "Accident location · dataset code",
                option_map["Accident_Location_A"],
                format_func=display_value,
            )

            hour_value = st.selectbox(
                "Accident hour",
                option_map["Accident_Hour"],
                format_func=display_value,
            )

        with time_col3:
            chainage_value = st.number_input(
                "Location chainage (km)",
                min_value=0.0,
                value=max(
                    0.0,
                    median_or_default(
                        df_options["Accident_Location_A_Chainage_km"]
                    ),
                ),
                step=0.1,
                format="%.3f",
            )

            year_median = median_or_default(
                df_options["Accident_Year"], 2024
            )

            year_value = st.number_input(
                "Accident year",
                min_value=1900,
                max_value=2100,
                value=int(np.clip(round(year_median), 1900, 2100)),
                step=1,
            )

        roadside_value = st.selectbox(
            "Roadside category · dataset code",
            option_map["Accident_Location_A_Chainage_km_RoadSide"],
            format_func=display_value,
        )

        st.markdown("---")
        st.markdown("#### 02 · Road, weather and cause")

        road_col1, road_col2 = st.columns(2)

        with road_col1:
            cause_value = st.selectbox(
                "Cause · dataset code",
                option_map["Causes_D"],
                format_func=display_value,
            )

            road_feature_value = st.selectbox(
                "Road feature · dataset code",
                option_map["Road_Feature_E"],
                format_func=display_value,
            )

            road_condition_value = st.selectbox(
                "Road condition · dataset code",
                option_map["Road_Condition_F"],
                format_func=display_value,
            )

        with road_col2:
            weather_value = st.selectbox(
                "Weather conditions · dataset code",
                option_map["Weather_Conditions_H"],
                format_func=display_value,
            )

            weekend_value = st.selectbox(
                "Weekend indicator",
                option_map["Is_Weekend"],
                format_func=display_value,
            )

            day_of_week_value = st.selectbox(
                "Accident day-of-week code",
                option_map["Accident_DayOfWeek"],
                format_func=display_value,
                help=(
                    "Selected independently to preserve the feature values "
                    "expected by the trained model."
                ),
            )

        st.markdown("---")
        st.markdown("#### 03 · Vehicles involved")

        vehicle_col1, vehicle_col2 = st.columns(2)

        with vehicle_col1:
            vehicle1_value = st.selectbox(
                "Vehicle type 1 · dataset code",
                option_map["Vehicle_Type_Involved_J_V1"],
                format_func=display_value,
            )

        with vehicle_col2:
            vehicle2_options = option_map["Vehicle_Type_Involved_J_V2"]

            vehicle2_value = st.selectbox(
                "Vehicle type 2 · dataset code",
                vehicle2_options,
                format_func=display_value,
                help="Choose a value represented in the prepared dataset.",
            )

        st.markdown(
            "<div style='height:4px'></div>",
            unsafe_allow_html=True,
        )

        submitted = st.form_submit_button(
            "Predict severity class",
            type="primary",
            use_container_width=True,
        )
        
    if submitted:
        # Keep the exact feature names and order expected by the supplied model.
        input_data = pd.DataFrame(
            [{
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
            }],
            columns=feature_columns,
        )

        try:
            prediction_input = input_data.copy()

            for column in categorical_features:
                prediction_input[column] = (
                    prediction_input[column]
                    .astype("string")
                    .fillna("Missing")
                    .astype(str)
                )

            raw_prediction = np.asarray(
                model.predict(prediction_input)
            ).reshape(-1)

            if raw_prediction.size == 0:
                raise ValueError("The model returned no prediction.")

            encoded_prediction = int(raw_prediction[0])
            
            explanation = explain_one_prediction(
                model,
                prediction_input,
                encoded_prediction,
            )

            severity_prediction = label_encoder.inverse_transform(
                np.asarray([encoded_prediction], dtype=int)
            )[0]

            st.session_state["last_prediction"] = {
                "severity": severity_prediction,
                "input": input_data.copy(),
                "explanation": explanation,
            }

        except Exception as error:
            st.error(
                "Prediction could not be completed. Confirm that the selected values and "
                "feature schema match the model used during training."
            )
            st.exception(error)

    last_prediction = st.session_state.get("last_prediction")

    if last_prediction is not None:
        st.markdown(
            '<div class="section-heading">Prediction result</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-subtitle">Result from the saved model for the submitted conditions.</div>',
            unsafe_allow_html=True,
        )

        result_col, details_col = st.columns([1, 1.35], gap="large")

        with result_col:
            st.markdown(
                f"""
                <div class="surface">
                  <div class="mini-label">Predicted severity class</div>
                  <div class="status-pill">Model prediction</div>
                  <div class="result-number">{display_value(last_prediction["severity"])}</div>
                  <div class="result-note">
                    This is the severity class returned for the conditions you entered.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with details_col:
            st.markdown(
                """
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
                """,
                unsafe_allow_html=True,
            )
            
        explanation = last_prediction.get("explanation")

        if explanation is not None and not explanation.empty:
            st.markdown("#### Factors influencing this prediction")
            st.caption(
                "Positive SHAP values support the predicted class; negative values "
                "push away from it. These are model contributions, not proof of causation."
            )
            
            explanation_chart = explanation.sort_values("contribution")
            fig = px.bar(
                explanation_chart,
                x="contribution",
                y="feature",
                orientation="h",
                color="contribution",
                color_continuous_scale=["#d95f59", "#d9e2f0", "#2463eb"],
                labels={
                    "contribution": "SHAP contribution to model score",
                    "feature": "Encoded model feature",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Review submitted conditions"):
            st.dataframe(
                last_prediction["input"],
                use_container_width=True,
                hide_index=True,
            )

    else:
        st.markdown(
            """
            <div class="surface">
              <div class="mini-label">Ready when you are</div>
              <div style="font-family:Manrope,sans-serif;font-size:1.15rem;font-weight:800;color:#18365e;margin-bottom:7px;">
                Your result will appear here
              </div>
              <div class="result-note">
                Complete the accident conditions above and select <b>Predict severity class</b>.
                The model's output will be shown here, along with the submitted input values.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------
# Analytics page
# ---------------------------------------------------------------------

elif selected_page == "Analytics":
    st.markdown(
        '<div class="section-heading">Model performance and accident patterns</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">Held-out model results and summaries of recorded accidents.</div>',
        unsafe_allow_html=True,
    )
    
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
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1 score": "{:.3f}",
                "Test rows": "{:.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        
        st.markdown("#### Confusion matrix")
        confusion = pd.DataFrame(
            metrics["confusion_matrix"],
            index=metrics["classes"],
            columns=metrics["classes"],
        )
        st.dataframe(confusion, use_container_width=True)
        
    else:
        st.info(
            "Run the metrics cell in 03_model_training.ipynb first. "
            "It creates models/model_metrics.json."
        )
        
    if importance_path.exists():
        importance = pd.read_csv(importance_path).head(15).sort_values(
            "mean_abs_shap"
        )
        st.markdown("#### Global SHAP feature importance")
        st.caption(
            "This chart summarizes which encoded input features most influenced "
            "the model across the sampled test records."
        )
        fig = px.bar(
            importance,
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            labels={
                "mean_abs_shap": "Mean absolute SHAP value",
                "feature": "Encoded model feature",
            },
        )
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info(
            "Run the SHAP cell in 03_model_training.ipynb first. "
            "It creates models/feature_importance.csv."
        )
    st.markdown("#### Recorded accidents by severity class")
    severity_counts = (
        df_options["Accident_Severity_C"]
        .value_counts()
        .sort_index()
        .rename_axis("Severity class")
        .reset_index(name="Recorded accidents")
    )
    fig = px.bar(
        severity_counts,
        x="Severity class",
        y="Recorded accidents",
        labels={"Severity class": "Dataset severity code"},
    )
    st.plotly_chart(fig, use_container_width=True)

    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        monthly_counts = (
            df_options.groupby("Accident_Month")
            .size()
            .reset_index(name="Recorded accidents")
        )
        fig = px.bar(
            monthly_counts,
            x="Accident_Month",
            y="Recorded accidents",
            labels={"Accident_Month": "Month"},
        )
        st.markdown("#### Recorded accidents by month")
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        weather_counts = (
            df_options["Weather_Conditions_H"]
            .value_counts()
            .sort_index()
            .rename_axis("Weather code")
            .reset_index(name="Recorded accidents")
        )
        fig = px.bar(
            weather_counts,
            x="Weather code",
            y="Recorded accidents",
        )
        st.markdown("#### Recorded accidents by weather code")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# About page
# ---------------------------------------------------------------------

else:
    st.markdown(
        '<div class="section-heading">About the project</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">A machine-learning interface for exploring recorded road-accident severity.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="surface">
          <div class="mini-label">Project overview</div>
          <h3 style="font-family:Manrope,sans-serif;color:#18365e;margin:0 0 10px 0;">
            Turning accident conditions into a predicted severity class
          </h3>
          <p style="color:#5f7189;line-height:1.75;margin:0;">
            The Smart Traffic Risk Prediction System takes structured accident information—such as
            time, location, road conditions, weather, reported cause, and vehicles involved—and passes
            those features to a previously trained machine-learning model. The saved label encoder is
            used to translate the model's encoded output back into a severity class.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    card1, card2, card3 = st.columns(3, gap="medium")

    with card1:
        st.markdown(
            """
            <div class="info-card">
              <h4>🧾 Input conditions</h4>
              <p>Use prepared-dataset values to describe the recorded accident, including time, location, road, weather and vehicle details.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with card2:
        st.markdown(
            """
            <div class="info-card">
              <h4>🧠 Saved ML model</h4>
              <p>The application loads your existing serialized model and label encoder. It does not retrain or overwrite them.</p>
            </div>
            """,    
            unsafe_allow_html=True,
        )

    with card3:
        st.markdown(
            """
            <div class="info-card">
              <h4>🚦 Interpretable output</h4>
              <p>The result is the predicted severity class. Its real-world meaning must be confirmed from the dataset's class documentation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div style='height:16px'></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="surface">
          <div class="mini-label">Important limitations</div>
          <p style="color:#5f7189;line-height:1.75;margin:0;">
            This application predicts a class from the fields supplied to the model. It does not
            establish causation, provide a live traffic feed, or calculate the likelihood of a future
            accident unless those capabilities are explicitly built and validated separately.
            Dataset category codes should be interpreted using the original data dictionary.
            Do not use the output as the sole basis for operational, legal, or emergency decisions.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.markdown(
    """
    <div style="border-top:1px solid #e3e9f1;margin-top:28px;padding-top:16px;color:#8190a4;font-size:.78rem;text-align:center;">
      Smart Traffic Risk Prediction System · Educational decision-support interface
    </div>
    """,
    unsafe_allow_html=True,
)