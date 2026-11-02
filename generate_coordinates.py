import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "accident_data_prepared.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "location_segments.csv"

LOCATION_NAMES = {1: "Urban Corridor", 2: "Rural Highway", 3: "Unallocated Route"}

# Distinct base paths (lat, lon, lat_step, lon_step) near NCR/NH-48 so routes don't overlap
ROUTE_PATHS = {
    1: (28.6139, 77.2090, -0.012, -0.015),
    2: (28.4595, 77.0266, -0.018, -0.022),
    3: (28.7041, 77.1025,  0.015, -0.012),
}

df = pd.read_csv(DATA_PATH)
df["Accident_Location_A"] = pd.to_numeric(df["Accident_Location_A"], errors="coerce")
df["Accident_Location_A_Chainage_km"] = pd.to_numeric(df["Accident_Location_A_Chainage_km"], errors="coerce")

step_km = 5.0
mock_data = []

for loc in sorted(df["Accident_Location_A"].dropna().unique()):
    loc_int = int(loc)
    sub = df[df["Accident_Location_A"] == loc]["Accident_Location_A_Chainage_km"].dropna()
    max_ch = max(float(sub.max()) if not sub.empty else 50.0, step_km)

    base_lat, base_lon, d_lat, d_lon = ROUTE_PATHS.get(loc_int, (28.5, 77.1, 0.01, 0.01))
    bins = np.arange(0.0, np.ceil(max_ch / step_km) * step_km + step_km, step_km)

    for idx in range(len(bins) - 1):
        start_km, end_km = float(bins[idx]), float(bins[idx + 1])
        mock_data.append({
            "Accident_Location_A": loc_int,
            "chainage_start_km": start_km,
            "chainage_end_km": end_km,
            "latitude": round(base_lat + idx * d_lat + np.random.uniform(-0.003, 0.003), 6),
            "longitude": round(base_lon + idx * d_lon + np.random.uniform(-0.003, 0.003), 6),
            "segment_name": f"{LOCATION_NAMES.get(loc_int, f'Route {loc_int}')} ({int(start_km)}-{int(end_km)} km)",
        })

segments_df = pd.DataFrame(mock_data)
segments_df.to_csv(OUTPUT_PATH, index=False)
print(f"Generated {len(segments_df)} road segments at {OUTPUT_PATH}")