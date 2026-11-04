# 🚦 TrafficSense – Smart Traffic Risk Prediction System

<p align="center">
  <a href="https://trafficsense-strp.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit">
  </a>
</p>

**Live Demo:** [TrafficSense – Smart Traffic Risk Prediction System](https://trafficsense-strp.streamlit.app/)

## 📌 About the Project

TrafficSense is a machine learning-based traffic risk prediction system designed to predict road accident severity. It combines historical accident data, road parameters, geographical information, and real-time weather conditions to provide traffic risk assessments through an interactive dashboard.

## ✨ Key Features

- **Accident Severity Prediction:** Uses an XGBoost machine learning model to predict accident severity.
- **Live Weather Integration:** Fetches real-time weather information using the Open-Meteo API.
- **Interactive Risk Mapping:** Visualizes highway segments and accident hotspots using Plotly maps.
- **Explainable AI:** Uses SHAP to explain factors influencing model predictions.
- **Prediction History:** Automatically stores prediction records in CSV files.
- **Custom Locations:** Allows users to add geographical road segments through the dashboard.

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Dashboard | Streamlit |
| Machine Learning | XGBoost, Scikit-learn |
| Explainable AI | SHAP |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Mapping | OpenStreetMap |
| Weather API | Open-Meteo |

## 📂 Project Structure

```text
TrafficSense/
│
├── app.py
├── generate_coordinates.py
├── requirements.txt
├── README.md
│
├── data/
├── models/
├── notebooks/
└── .devcontainer/
```

## ⚙️ Installation and Setup

**1. Clone the repository**

```bash
git clone https://github.com/Priy​​anshi-070207/TrafficSense--Smart-Traffic-Risk-Prediction-System.git

cd TrafficSense--Smart-Traffic-Risk-Prediction-System
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Generate map coordinates**

```bash
python generate_coordinates.py
```

**4. Run the application**

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

## 🔄 System Workflow

1. Historical accident data collection.
2. Data preprocessing and feature engineering.
3. XGBoost model training.
4. User input and real-time weather retrieval.
5. Accident severity prediction.
6. Interactive risk visualization.
7. SHAP-based prediction explanation.
8. Prediction logging.

## 🎯 Project Objectives

- Predict road accident severity using machine learning.
- Analyze historical accident patterns.
- Incorporate real-time weather conditions.
- Identify geographical accident hotspots.
- Improve prediction transparency through Explainable AI.

## 🚀 Future Scope

- Integration of live traffic density.
- Expansion to more highway locations.
- Improved prediction models.
- Advanced real-time traffic monitoring.

## ⚠️ Disclaimer

TrafficSense provides estimated traffic risk assessments for educational and decision-support purposes. Predictions are not guaranteed and should not replace official traffic safety information.

## 👩‍💻 Project Information

- **Project Name:** TrafficSense – Smart Traffic Risk Prediction System
- **Domain:** Machine Learning and Data Science
- **Language:** Python
- **Deployment:** Streamlit Community Cloud
- **Repository:** [GitHub](https://github.com/Priy​​anshi-070207/TrafficSense--Smart-Traffic-Risk-Prediction-System)

---

<p align="center">
  **TrafficSense – Making Roads Smarter Through Machine Learning. 🚦**
</p>
