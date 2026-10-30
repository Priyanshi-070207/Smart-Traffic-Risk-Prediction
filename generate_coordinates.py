import pandas as pd
import numpy as np
from pathlib import Path

# Setup paths using 'data' to match your folder structure
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "accident_data_prepared.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "location_segments.csv"

print(f"Reading data from: {DATA_PATH}")

try:
    df = pd.read_csv(DATA_PATH)
    locations = df['Accident_Location_A'].dropna().unique()

    # Generate coordinates centered near Delhi (or your preferred base location)
    mock_data = []
    for loc in locations:
        mock_data.append({
            "Accident_Location_A": loc,
            "chainage_start_km": 0.0,
            "chainage_end_km": 9999.0, 
            "latitude": 28.6139 + np.random.uniform(-0.5, 0.5),
            "longitude": 77.2090 + np.random.uniform(-0.5, 0.5),
            "segment_name": f"Route_{int(loc)}"
        })

    # Save to the expected location
    segments_df = pd.DataFrame(mock_data)
    segments_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Successfully generated {len(segments_df)} coordinate mappings at {OUTPUT_PATH}")
    
except Exception as e:
    print(f"❌ Error: {e}")