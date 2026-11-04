# TrafficSense: Smart Traffic Risk Prediction System

TrafficSense is an interactive decision-support dashboard designed to predict the severity of road accidents. By combining historical crash data, route parameters, and real-time weather conditions, it provides granular risk assessments for specific highway segments to aid in proactive traffic management.

## Key Features

* **Live Weather Integration:** Fetches real-time temperature, wind, and precipitation data via the Open-Meteo API to dynamically adjust predictive risk models.
* **Spatial Risk Mapping:** Visualizes multi-segment highway corridors and historical accident hotspots using interactive Plotly maps.
* **Explainable AI:** Utilizes SHAP (SHapley Additive exPlanations) to provide human-readable insights into which road or weather factors most influenced a specific severity prediction.
* **Data Persistence:** Automatically logs new predictions to a CSV and allows users to append custom location segments directly from the user interface.

## Tech Stack

* **Frontend:** Streamlit, Plotly
* **Machine Learning:** XGBoost, Scikit-learn, SHAP
* **Data Processing:** Pandas, NumPy
* **External APIs:** Open-Street Maps(For Map Display),Open-Meteo (Real-time Weather)

## Project Structure

* `app.py`: The main Streamlit dashboard application.
* `generate_coordinates.py`: Script to interpolate and generate granular spatial mapping coordinates for road segments.
* `notebooks/`: Jupyter notebooks for exploratory data analysis (`01`), preprocessing (`02`), and model training (`03`).
* `models/`: Contains the serialized XGBoost model, label encoders, and evaluation metrics (`.pkl` and `.json` files).
* `data/`: Raw and processed dataset files, including prediction logs and location segments.

## Installation and Setup

1. **Clone the repository** and navigate to the project directory:
   ```bash
   git clone https://github.com/Priyanshi-070207/Smart-Traffic-Risk-Prediction.git
   cd Smart-Traffic-Risk-Prediction
   ```
2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate map coordinate segments** (required for the dashboard map to render):
   ```bash
   python generate_coordinates.py
   ```

4. **Launch the application:**
   ```bash
   streamlit run app.py
   ```
